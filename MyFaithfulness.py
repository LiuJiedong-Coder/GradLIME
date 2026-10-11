# -*- coding: utf-8 -*-
"""
MyFaithfulness.py —— 统一超像素粒度的忠实性批量评估
====================================================
指标（已删除 Sufficiency）：
    ['Correlation', 'Estimate', 'MonotonicityCorrelation',
     'IROF', 'RegionPerturbation']

沿用项目 GradLIME 的接口与取值：
    trans_A / trans_B / trans_C（ImageNet 归一化）
    batch_predict -> softmax 概率
    models.resnet18(pretrained=True), target_layers=[model.layer4]
    lime_image.LimeImageExplainer().explain_instance(...)
    lime_image_my.LimeImageExplainer():
        explain_instance_data_label(...) + explain_instance(grayscale_cam)
    random_seed=500, hide_color=0

--------------------------------------------------------------------------
修复要点
--------------------------------------------------------------------------
1. 归因图必须由拟合权重构造
   原代码用 get_image_and_mask + mark_boundaries + sum(axis=2)，
   实际编码的是「图像亮度 + 黄色边界」，权重从未写入。
   现在直接用 local_exp 写回 segments，保留符号与量级。

2. 全部指标统一到超像素粒度（见 my_metrics.py）
   12 个方法（含 IG/SA/GS/... 等稠密方法）共用同一套超像素分割：
   稠密归因在每个超像素内取均值，再按 |值| 降序逐个删除整个超像素，
   统一删 K 个。删除单元 == 归因单元，无恒定区。

3. 对照公平性
   原 LIME 用 positive_only=False、GradLIME 用 True，后者天然更干净，
   会系统性利好 IROF。现在两侧都不走 get_image_and_mask，统一取 Top-K。

4. 新增开关
   --require-correct      只保留模型预测正确的图
   --subset-size N        仅 quantus 原生后端生效（--metric-backend quantus）
   --metric-backend       superpixel（默认，推荐）| quantus（对照）
   输出：跨图配对 Wilcoxon + Holm 校正 + Cohen's d

用法
----
    python MyFaithfulness.py --method gradcam \
        --num-classes 10 --num-images-per-class 30 \
        --num-samples 1000 --num-features 20 --device cuda:0

    # 只看预测正确的图
    python MyFaithfulness.py --method gradcam --require-correct ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings

import numpy as np
import PIL
import quantus
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from my_metrics import MyMetrics
from pytorch_grad_cam import GradCAM

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

trans_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
trans_A = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    trans_norm,
])
trans_B = transforms.Compose([transforms.ToTensor(), trans_norm])
trans_C = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
])

# Sufficiency 已删除（阈值保留 60% 像素时恒为 0，无区分度）
METRICS = MyMetrics.metric_names()

EXP_METHODS = ['Lime', 'our', 'IG', 'SA', 'GS', 'IXG', 'DeCon', 'KH', 'LRP',
               'Gradient', 'CVSF', 'CVRU']


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# --------------------------------------------------------------------------- #
def attribution_from_local_exp(explanation, label, num_features=None):
    """把 local_exp 的 (超像素 id, 权重) 写回 segments，得到带符号归因图。"""
    segments = explanation.segments
    items = explanation.local_exp[label]
    if num_features is not None and num_features > 0:
        items = items[:num_features]
    lut = np.zeros(int(segments.max()) + 1, dtype=np.float32)
    for seg_id, weight in items:
        lut[int(seg_id)] = float(weight)
    return lut[segments]


def describe_attr(name, attr):
    print(f"    [{name}] min={attr.min():+.5f} max={attr.max():+.5f} "
          f"nonzero={(attr != 0).sum()} pos={(attr > 0).sum()} "
          f"neg={(attr < 0).sum()} uniq={len(np.unique(np.round(attr, 6)))}")


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m, dtype=float)
    order = np.argsort(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def paired_test(a, b, min_n=5):
    """配对 Wilcoxon + Cohen's d。a/b 为同长度数组（同一批图）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[ok]
    n = len(d)
    out = dict(n=n, mean_a=float(np.mean(a[ok])) if n else float("nan"),
               mean_b=float(np.mean(b[ok])) if n else float("nan"),
               std_a=float(np.std(a[ok], ddof=1)) if n > 1 else float("nan"),
               std_b=float(np.std(b[ok], ddof=1)) if n > 1 else float("nan"),
               mean_diff=float(np.mean(d)) if n else float("nan"),
               cohen_d=float("nan"), p_raw=1.0, p_holm=1.0)
    if n < min_n or np.allclose(d, 0):
        return out
    try:
        from scipy.stats import wilcoxon
        out["p_raw"] = float(wilcoxon(d).pvalue)
    except Exception:                                                # noqa: BLE001
        out["p_raw"] = 1.0
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if sd > 1e-12:
        out["cohen_d"] = float(np.mean(d) / sd)
    return out


# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=str, default='gradcam')
    p.add_argument('--eigen-smooth', action='store_true')
    p.add_argument('--aug-smooth', action='store_true')
    p.add_argument('--base-dir', default=r'ImageNet2012/val300')
    p.add_argument('--imagenet-json', default='imagenet/imagenet_class_index.json')
    p.add_argument('--num-samples', type=int, default=1000,
                   help='LIME 邻域样本数 N（1000 与 8000 差异<0.005）')
    p.add_argument('--num-features', type=int, default=20,
                   help='解释长度 K，同时是删除的超像素数')
    p.add_argument('--random-seed', type=int, default=500)
    p.add_argument('--num-classes', type=int, default=100)
    p.add_argument('--num-images-per-class', type=int, default=15)
    p.add_argument('--device', default="cuda")
    p.add_argument('--out-dir', default='results')
    # 新增
    p.add_argument('--require-correct', action='store_true',
                   help='只保留模型预测正确的图（black_correct==1）')
    p.add_argument('--metric-backend', choices=['superpixel', 'quantus'],
                   default='superpixel',
                   help='superpixel=统一超像素粒度（推荐）; quantus=原生口径')
    p.add_argument('--subset-size', type=int, default=10000,
                   help='仅 quantus 原生后端生效: FaithfulnessCorrelation 的'
                        '随机子集大小（默认 10000，约覆盖 6.6%）')
    p.add_argument('--softmax', dest='softmax', action='store_true', default=True)
    p.add_argument('--logits', dest='softmax', action='store_false')
    return p.parse_args()


def load_images_from_folder(folder_path):
    paths, images = [], []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            try:
                images.append(PIL.Image.open(img_path))
                paths.append(img_path)
            except Exception:                                        # noqa: BLE001
                print("Unable to load image:", img_path)
    return paths, images


# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print('device:', device)
    print('backend:', args.metric_backend,
          '| subset_size:', args.subset_size,
          '(仅 quantus 后端生效)' if args.metric_backend == 'superpixel' else '')
    print('require_correct:', args.require_correct)
    print('metrics:', METRICS)

    idx2label, cls2idx = [], {}
    if os.path.exists(args.imagenet_json):
        with open(os.path.abspath(args.imagenet_json), 'r') as f:
            class_idx = json.load(f)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    model = models.resnet18(pretrained=True).eval().to(device)

    methods = {
        "gradcam": GradCAM
    }

    def batch_predict(images):
        batch = torch.stack(tuple(trans_B(i) for i in images), dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
        return F.softmax(logits, dim=1).detach().cpu().numpy()

    classes = [d for d in os.listdir(args.base_dir)
               if os.path.isdir(os.path.join(args.base_dir, d))]
    if args.num_classes > 0:
        classes = classes[:args.num_classes]

    rows, skipped = [], 0
    t0 = time.time()
    n_done = 0

    for class_name in classes:
        label = cls2idx.get(class_name)
        class_dir = os.path.join(args.base_dir, class_name)
        paths, loaded_images = load_images_from_folder(class_dir)
        if args.num_images_per_class > 0:
            paths = paths[:args.num_images_per_class]
            loaded_images = loaded_images[:args.num_images_per_class]

        print(f'\n===== class: {class_name} (label={label}), '
              f'{len(paths)} images =====')

        for k, img_pil in enumerate(loaded_images):
            img_pil = img_pil.convert('RGB')
            input_tensor = trans_A(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_id = int(F.softmax(model(input_tensor), dim=1)
                              .argmax(dim=1).item())
            black_pred_id = np.expand_dims(np.array(pred_id), 0)

            correct = int(pred_id == label)
            if args.require_correct and not correct:
                skipped += 1
                continue

            cam_algorithm = methods[args.method]
            with cam_algorithm(model=model, target_layers=[model.layer4],
                               use_cuda=str(device).startswith('cuda')) as cam:
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=[ClassifierOutputTarget(pred_id)],
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)[0, :]

            from lime import lime_image
            explainer_org = lime_image.LimeImageExplainer()
            explanation_org = explainer_org.explain_instance(
                np.array(trans_C(img_pil)), batch_predict,
                top_labels=1, hide_color=0,
                num_samples=args.num_samples,
                num_features=args.num_features,
                random_seed=args.random_seed)

            from lime import lime_image_my
            explainer = lime_image_my.LimeImageExplainer()
            explainer.explain_instance_data_label(
                np.array(trans_C(img_pil)), batch_predict,
                top_labels=1, hide_color=0,
                num_samples=args.num_samples,
                num_features=args.num_features,
                random_seed=args.random_seed)
            explanation = explainer.explain_instance(grayscale_cam)

            lab_org = int(explanation_org.top_labels[0])
            lab_our = int(explanation.top_labels[0])
            if not (lab_org == lab_our == pred_id):
                print(f'  [warn] label mismatch: pred={pred_id} '
                      f'LIME={lab_org} Ours={lab_our}; 用 pred_id')
            lab = pred_id

            # ---- 归因图：由拟合权重构造 ----
            K = args.num_features
            attr_org = attribution_from_local_exp(explanation_org, lab, K)
            attr_our = attribution_from_local_exp(explanation, lab, K)
            describe_attr('Lime', attr_org)
            describe_attr('our ', attr_our)

            # 统一超像素分割（用 LIME 的分割，12 个方法共用）
            segments = explanation_org.segments

            x_batch = input_tensor.cpu().numpy()
            try:
                attr_IG = quantus.explain(model, input_tensor, black_pred_id,
                                          method="IntegratedGradients")
                attr_SA = quantus.explain(model, input_tensor, black_pred_id,
                                          method="Saliency")
                attr_GS = quantus.explain(model, input_tensor, black_pred_id,
                                          method="GradientShap")
                attr_IXG = quantus.explain(model, input_tensor, black_pred_id,
                                           method="InputXGradient")
                attr_DeCon = quantus.explain(model, input_tensor, black_pred_id,
                                             method="Deconvolution")
                attr_KH = quantus.explain(model, input_tensor, black_pred_id,
                                          method="KernelShap")
                attr_LRP = quantus.explain(model, input_tensor, black_pred_id,
                                           method="LRP")
                attr_G = quantus.explain(model, input_tensor, black_pred_id,
                                         method="Gradient")
                attr_CVSF = quantus.explain(model, input_tensor, black_pred_id,
                                            method="Control Var. Sobel Filter")
                attr_CVRU = quantus.explain(model, input_tensor, black_pred_id,
                                            method="Control Var. Random Uniform")
            except Exception as exc:                                 # noqa: BLE001
                print(f'  [warn] quantus.explain 失败: {exc}')
                continue

            attrs = [attr_org, attr_our, attr_IG, attr_SA, attr_GS,
                     attr_IXG, attr_DeCon, attr_KH, attr_LRP, attr_G,
                     attr_CVSF, attr_CVRU]

            row = {'img': os.path.basename(paths[k]), 'class': class_name,
                   'pred': pred_id, 'black_correct': correct}
            for metric in METRICS:
                for name, attr in zip(EXP_METHODS, attrs):
                    a4 = np.asarray(attr, dtype=np.float32)
                    if a4.ndim == 2:
                        a4 = a4.reshape(1, 1, *a4.shape)
                    try:
                        s = MyMetrics.compute_Faithfulness(
                            model, x_batch, black_pred_id, a4, metric, device,
                            segments=segments, num_features=K,
                            softmax=args.softmax,
                            backend=args.metric_backend,
                            subset_size=args.subset_size)
                        row[f'{metric}__{name}'] = float(s)
                    except Exception as exc:                          # noqa: BLE001
                        row[f'{metric}__{name}'] = float('nan')
                        print(f'  [warn] {metric}/{name}: '
                              f'{type(exc).__name__}: {exc}')
            rows.append(row)
            n_done += 1
            print(f'  [{n_done}] {row["img"]} pred={pred_id} correct={correct} '
                  f'Corr(Lime)={row.get("Correlation__Lime", float("nan")):+.4f} '
                  f'Corr(our)={row.get("Correlation__our", float("nan")):+.4f}')

    if not rows:
        print('没有成功评估的图像')
        return

    # ---- 逐图 CSV ----
    csv_path = os.path.join(args.out_dir, 'faithfulness_scores.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\n[info] 逐图分数 -> {csv_path}')

    # ---- 各方法均值 ----
    sum_path = os.path.join(args.out_dir, 'faithfulness_summary.csv')
    with open(sum_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric'] + EXP_METHODS)
        for metric in METRICS:
            vals = []
            for name in EXP_METHODS:
                col = [r[f'{metric}__{name}'] for r in rows]
                col = [c for c in col if not np.isnan(c)]
                vals.append(f'{np.mean(col):.4f}' if col else 'nan')
            w.writerow([metric] + vals)
    print(f'[info] 汇总均值 -> {sum_path}')

    print('\n===== 各指标均值 =====')
    print(f"{'metric':<26}" + ''.join(f'{m:>9}' for m in EXP_METHODS))
    for metric in METRICS:
        line = f'{metric:<26}'
        for name in EXP_METHODS:
            col = [r[f'{metric}__{name}'] for r in rows]
            col = [c for c in col if not np.isnan(c)]
            line += f'{np.mean(col):>9.4f}' if col else f'{"nan":>9}'
        print(line)

    # ---- 配对 Wilcoxon：our vs Lime ----
    print('\n===== 配对检验：our - Lime =====')
    tests = {}
    for metric in METRICS:
        a = np.array([r[f'{metric}__Lime'] for r in rows], dtype=float)
        b = np.array([r[f'{metric}__our'] for r in rows], dtype=float)
        tests[metric] = paired_test(a, b)
    adj = holm_bonferroni([tests[m]['p_raw'] for m in METRICS])
    for metric, pa in zip(METRICS, adj):
        tests[metric]['p_holm'] = float(pa)

    print(f"{'metric':<26}{'n':>4}{'Lime':>16}{'our':>16}{'diff':>10}"
          f"{'p_raw':>9}{'p_holm':>9}{'d':>7}")
    print('-' * 97)
    for metric in METRICS:
        t = tests[metric]
        ma = f"{t['mean_a']:.4f}±{t['std_a']:.4f}" if t['n'] > 1 else f"{t['mean_a']:.4f}"
        mb = f"{t['mean_b']:.4f}±{t['std_b']:.4f}" if t['n'] > 1 else f"{t['mean_b']:.4f}"
        print(f"{metric:<26}{t['n']:>4}{ma:>16}{mb:>16}{t['mean_diff']:>+10.4f}"
              f"{t['p_raw']:>9.4f}{t['p_holm']:>9.4f}{t['cohen_d']:>+7.3f}")
    print('-' * 97)

    tpath = os.path.join(args.out_dir, 'faithfulness_paired_tests.csv')
    with open(tpath, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'n', 'mean_Lime', 'std_Lime', 'mean_our',
                    'std_our', 'mean_diff', 'cohen_d', 'p_raw', 'p_holm'])
        for metric in METRICS:
            t = tests[metric]
            w.writerow([metric, t['n'], t['mean_a'], t['std_a'], t['mean_b'],
                        t['std_b'], t['mean_diff'], t['cohen_d'],
                        t['p_raw'], t['p_holm']])
    print(f'[info] 配对检验 -> {tpath}')

    print(f'\n[info] 图像数={n_done}（因 require_correct 跳过 {skipped} 张）'
          f'  用时={time.time()-t0:.1f}s')
    if n_done < 30:
        print('[reminder] n<30，配对检验功效很低；论文口径建议 >=300 张。')


if __name__ == '__main__':
    main()
