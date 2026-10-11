# -*- coding: utf-8 -*-
"""
MyComplexity.py —— 解释复杂性评估（结构与 MyFaithfulness.py 对齐）
==================================================================
指标：['Sparseness', 'Complexity', 'EffectiveComplexity']

方向性（重要，与原脚本不同）：
    Sparseness          Gini 系数，越高越稀疏 = 越简单（越高越好）
    Complexity          熵，越高越复杂（越低越好）
    EffectiveComplexity 达到 (1-eps) 累积归因质量所需单元数（越低越好）
                        eps 由 --ec-threshold 控制，默认 0.1（即 90% 质量）

    注意：EC 是**整数计数**。当权重分布接近均匀时（本数据中两个方法的熵
    均已达到 log(K) 的 93%），任何阈值下两侧差异都只有 1 个单元左右，
    分辨率上限约为 ±1。它适合作为辅助指标，不应作为主要证据；
    Sparseness（连续量，本数据 d≈0.48）分辨力高得多。

--------------------------------------------------------------------------
本次修复
--------------------------------------------------------------------------
A. quantus.explain 的方法名与单点故障（本次报错的直接原因）
   报错有两种形态，都源于"方法名 + 异常传播"：
     (1) ValueError: The selected control_var_sobel_filter XAI method is not
         in the list of supported ... ['Control Var. Sobel Filter', ...]
         -> quantus 内置列表用的是 Title Case 写法，小写下划线名不被接受。
     (2) KernelShap (KH) 在 captum/_core/lime.py 的 _convert_output_shape 抛错，
         未被捕获，直接中断整个 run（traceback 停在 attr_KH 那行）。
   修复：
     * _NAME_CANDIDATES 按候选顺序尝试，成功即缓存 _RESOLVED；
     * 每个稠密方法**各自** try/except，任一方法失败只把该方法停用
       （DISABLED），不再 continue 掉整张图，也不会中断 run；
     * 方法中途失败时，回溯把已写入的该列全部置 NaN，保证列对齐。

B. 归因图必须由拟合权重构造（最关键的方法学修复）
   原代码：
       temp, mask = explanation.get_image_and_mask(..., positive_only=True, ...)
       img_boundry = mark_boundaries(temp/255.0, mask)
       attributions = img_boundry.sum(axis=2)
   问题：LIME 源码 else 分支只把选中超像素的某一个通道拉到 255，权重数值
   从未写入；mark_boundaries 再叠加黄色描边。sum(axis=2) 编码的是
   「图像亮度 + 边界线」，量级排序为 亮背景(2.54) > 边界线(2.00) > 选中区。
   修复：用 local_exp 的 (超像素, 权重) 写回，保留符号与量级，区外置 0。

C. 复杂性必须在超像素粒度上计算（新增，默认开启）
   若在像素粒度上算熵：LIME 类归因在超像素内是常数，像素级 p_i 与该超像素
   的**面积**成正比，于是熵主要由"被点亮的面积有多大"决定，几乎与权重分布
   无关 —— 测的是面积而非解释的集中度。
   现在默认 --granularity superpixel：
       稠密归因  -> 每个超像素内取均值
       LIME/our  -> 直接用拟合权重（本身就是一个超像素一个值）
   得到长度 M 的向量后再算三个指标，与忠实性实验的删除单元一致。
   用 --granularity pixel 可回到像素粒度作对照。

D. 与忠实性实验同源
   --base-dir 默认值改回 ...\\val（原为 val1，与 MyFaithfulness 不一致，
   会导致两个实验不在同一批图像上）；新增 --require-correct，
   保证复杂性表与忠实性表可交叉引用。

E. EffectiveComplexity 阈值可配置（本次修改）
   原实现 eps=1e-5（99.999% 质量）。在 K=20 截断下，归因恰好有 20 个非零
   单元，而最小单元占比远大于 1e-5，因此 EC 恒等于 20 —— 完全饱和、零分辨力
   （实测两侧均为 20.0 ± 0.0）。
   现在 --ec-threshold 可配置，默认 0.1（即 90% 质量，文献常用取值），
   并支持一次给多个阈值（如 --ec-threshold 0.1 0.3 0.5），
   每个阈值生成独立指标列，便于说明结论不依赖单一阈值。

F. 统计输出
   原脚本往 .txt 追加原始字典，且 computer_avg_score 用 d[k][sk][0] 取标量，
   分数为标量时会崩。现在输出三个 CSV：逐图、各方法均值、配对检验
   （Wilcoxon + Holm + Cohen d），并显式标注每个指标的方向性。

用法
----
    python MyComplexity.py --method gradcam \
        --num-classes 100 --num-images-per-class 15 \
        --num-samples 1000 --num-features 20 \
        --require-correct --device cuda:0
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
from torchvision import models, transforms

from my_metrics import superpixel_values
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

# 越高越好(True) / 越低越好(False)
METRICS = ['Sparseness', 'Complexity', 'EffectiveComplexity']
HIGHER_BETTER = {'Sparseness': True, 'Complexity': False,
                 'EffectiveComplexity': False}
EC_EPS = {}          # 指标名 -> eps，在 main() 中按 --ec-threshold 填充
BASE_METRICS = ['Sparseness', 'Complexity']

EXP_METHODS = ['Lime', 'our', 'IG', 'SA', 'GS', 'IXG', 'DeCon', 'KH', 'LRP',
               'Gradient', 'CVSF', 'CVRU']
DENSE_KEYS = ['IG', 'SA', 'GS', 'IXG', 'DeCon', 'KH', 'LRP', 'Gradient',
              'CVSF', 'CVRU']

MODEL = None          # batch_predict 需要


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# --------------------------------------------------------------------------- #
# 归因图构造（与 MyFaithfulness.py 完全一致）
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
# quantus.explain 方法名解析 + 单点故障隔离
# --------------------------------------------------------------------------- #
# quantus 内置列表用 Title Case（"Control Var. Sobel Filter"）；
# 个别构建用小写下划线。按候选顺序尝试，成功即缓存。
_NAME_CANDIDATES = {
    'IG':       ["IntegratedGradients"],
    'SA':       ["Saliency"],
    'GS':       ["GradientShap"],
    'IXG':      ["InputXGradient"],
    'DeCon':    ["Deconvolution"],
    'KH':       ["KernelShap"],
    'LRP':      ["LRP"],
    'Gradient': ["Gradient"],
    'CVSF':     ["Control Var. Sobel Filter", "control_var_sobel_filter"],
    'CVRU':     ["Control Var. Random Uniform", "control_var_random_uniform"],
}

_RESOLVED = {}      # key -> 已验证可用的方法名
DISABLED = set()    # 已永久停用的方法


def explain_dense(model, input_tensor, black_pred_id, key):
    """返回 (attr, used_name)。失败返回 (None, None)，不抛异常。"""
    names = ([_RESOLVED[key]] if key in _RESOLVED
             else _NAME_CANDIDATES[key])
    for name in names:
        try:
            attr = quantus.explain(model, input_tensor, black_pred_id,
                                   method=name)
            _RESOLVED[key] = name
            return attr, name
        except Exception:                                            # noqa: BLE001
            continue
    return None, None


# --------------------------------------------------------------------------- #
# 复杂性指标（纯 numpy，定义显式，可独立验证）
# --------------------------------------------------------------------------- #
def _as_1d(a):
    return np.abs(np.asarray(a, dtype=np.float64)).reshape(-1)


def sparseness_gini(a):
    """Gini 系数（Chalasani et al., 2020）。越高越稀疏 = 越简单。"""
    x = np.sort(_as_1d(a))
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(((2 * i - n - 1) * x).sum() / (n * total))


def complexity_entropy(a):
    """Shannon 熵（Bhatt et al., 2020）。越高越复杂。"""
    x = _as_1d(a)
    if x.sum() <= 0:
        return float("nan")
    nz = x[x > 0]
    p = nz / nz.sum()
    return float(-(p * np.log(p)).sum())


def effective_complexity(a, eps=0.1):
    """达到 (1-eps) 累积归因质量所需单元数（Nguyen et al., 2020）。越小越简单。

    eps=0.1 即 90% 质量，为文献常用取值。
    eps=1e-5（quantus 默认）在 K=20 截断下会恒等于非零单元数，完全饱和。
    """
    x = np.sort(_as_1d(a))[::-1]
    total = x.sum()
    if total <= 0:
        return float("nan")
    cum = np.cumsum(x) / total
    idx = np.searchsorted(cum, 1.0 - eps)
    return float(min(idx + 1, x.size))


NP_IMPL = {'Sparseness': sparseness_gini,
           'Complexity': complexity_entropy,
           'EffectiveComplexity': effective_complexity}


# --------------------------------------------------------------------------- #
# 统计（与 MyFaithfulness.py 一致）
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m, dtype=float)
    order = np.argsort(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def paired_test(a, b, min_n=5):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[ok]
    n = len(d)
    out = dict(n=n, mean_a=float(np.mean(a[ok])) if n else float('nan'),
               mean_b=float(np.mean(b[ok])) if n else float('nan'),
               std_a=float(np.std(a[ok], ddof=1)) if n > 1 else float('nan'),
               std_b=float(np.std(b[ok], ddof=1)) if n > 1 else float('nan'),
               mean_diff=float(np.mean(d)) if n else float('nan'),
               cohen_d=float('nan'), p_raw=1.0, p_holm=1.0)
    if n < min_n or np.allclose(d, 0):
        return out
    try:
        from scipy.stats import wilcoxon
        out['p_raw'] = float(wilcoxon(d).pvalue)
    except Exception:                                                # noqa: BLE001
        out['p_raw'] = 1.0
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if sd > 1e-12:
        out['cohen_d'] = float(np.mean(d) / sd)
    return out


# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=str, default='gradcam')
    p.add_argument('--eigen-smooth', action='store_true')
    p.add_argument('--aug-smooth', action='store_true')
    p.add_argument('--base-dir', default=r'ImageNet2012/val300')
    p.add_argument('--imagenet-json', default='imagenet/imagenet_class_index.json')
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--num-features', type=int, default=20)
    p.add_argument('--random-seed', type=int, default=500)
    p.add_argument('--num-classes', type=int, default=100)
    p.add_argument('--num-images-per-class', type=int, default=15)
    p.add_argument('--device', default="cuda")
    p.add_argument('--out-dir', default='results')
    p.add_argument('--require-correct', action='store_true',
                   help='只保留模型预测正确的图（与忠实性实验保持一致）')
    p.add_argument('--granularity', choices=['superpixel', 'pixel'],
                   default='superpixel',
                   help='superpixel=在超像素单元上计算(默认, 与删除协议一致); '
                        'pixel=像素粒度对照')
    p.add_argument('--metric-backend', choices=['numpy', 'quantus'],
                   default='numpy')
    p.add_argument('--ec-threshold', type=float, nargs='+', default=[0.1],
                   help='EffectiveComplexity 的 eps：达到 (1-eps) 累积归因质量'
                        '所需单元数。默认 0.1 = 90%% 质量。'
                        '给多个值可一次生成多个指标列，例如 0.1 0.3 0.5。'
                        '注意 1e-5（quantus 默认）在 K=20 下会饱和。')
    p.add_argument('--log-every', type=int, default=10)
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


def flat(a):
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    return a


# --------------------------------------------------------------------------- #
def main():
    global MODEL
    args = get_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print('device:', device)
    print('granularity:', args.granularity,
          '| metric backend:', args.metric_backend)
    print('require_correct:', args.require_correct)

    # ---- 按 --ec-threshold 构建指标列表（原地修改 METRICS，后续循环无需改动）----
    thr = sorted(set(float(t) for t in args.ec_threshold))
    for t in thr:
        if not (0.0 < t < 1.0):
            raise ValueError(f'--ec-threshold 必须在 (0,1) 内，收到 {t}')
    if len(thr) == 1:
        ec_names = [('EffectiveComplexity', thr[0])]
    else:
        ec_names = [(f'EffectiveComplexity@{int(round((1 - t) * 100))}', t)
                    for t in thr]
    EC_EPS.update(dict(ec_names))
    HIGHER_BETTER.update({n: False for n, _ in ec_names})
    METRICS[:] = BASE_METRICS + [n for n, _ in ec_names]
    print('EffectiveComplexity 阈值:',
          ', '.join(f'{n}=eps {e:g}（{100 * (1 - e):.0f}% 质量）'
                    for n, e in ec_names))
    print('方向:', {m: ('越高越好' if HIGHER_BETTER[m] else '越低越好')
                    for m in METRICS})

    cls2idx = {}
    if os.path.exists(args.imagenet_json):
        with open(os.path.abspath(args.imagenet_json), 'r') as f:
            class_idx = json.load(f)
            cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    model = models.resnet18(pretrained=True).eval().to(device)
    MODEL = model

    def batch_predict(images):
        batch = torch.stack(tuple(trans_B(i) for i in images), dim=0).to(device)
        with torch.no_grad():
            logits = MODEL(batch)
        return F.softmax(logits, dim=1).detach().cpu().numpy()

    classes = [d for d in os.listdir(args.base_dir)
               if os.path.isdir(os.path.join(args.base_dir, d))]
    if args.num_classes > 0:
        classes = classes[:args.num_classes]

    rows, skipped = [], 0
    t0 = time.time()
    n_done = 0
    use_sp = (args.granularity == 'superpixel')

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
                pred_id = int(F.softmax(MODEL(input_tensor), dim=1)
                              .argmax(dim=1).item())
            black_pred_id = np.expand_dims(np.array(pred_id), 0)

            correct = int(pred_id == label)
            if args.require_correct and not correct:
                skipped += 1
                continue

            with GradCAM(model=model, target_layers=[model.layer4],
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

            lab = pred_id
            K = args.num_features
            attr_org = attribution_from_local_exp(explanation_org, lab, K)
            attr_our = attribution_from_local_exp(explanation, lab, K)
            segments = explanation_org.segments

            # ---- 稠密方法：逐个隔离，失败只停用该方法 ----
            dense, newly = {}, set()
            for key in DENSE_KEYS:
                if key in DISABLED:
                    continue
                try:
                    a, _used = explain_dense(model, input_tensor,
                                             black_pred_id, key)
                except Exception as exc:                             # noqa: BLE001
                    a = None
                    print(f'  [warn] {key} 异常: {type(exc).__name__}: {exc}')
                if a is None:
                    if key not in DISABLED:
                        DISABLED.add(key)
                        newly.add(key)
                        print(f'  [warn] 方法 {key} 不可用，'
                              f'该列将从所有图像中移除')
                    continue
                dense[key] = flat(a)

            # 中途停用的方法：把已写入的行回溯置 NaN，保持列对齐
            if newly:
                for r in rows:
                    for m in METRICS:
                        for key in newly:
                            r[f'{m}__{key}'] = float('nan')

            # ---- 统一到超像素单元（默认）----
            if use_sp:
                vals = {'Lime': superpixel_values(attr_org, segments)[1],
                        'our':  superpixel_values(attr_our, segments)[1]}
                for key, a in dense.items():
                    vals[key] = superpixel_values(a, segments)[1]
            else:
                vals = {'Lime': attr_org, 'our': attr_our}
                vals.update(dense)

            avail = [m for m in EXP_METHODS if m in vals]
            row = {'img': os.path.basename(paths[k]), 'class': class_name,
                   'pred': pred_id, 'black_correct': correct}
            for metric in METRICS:
                for name in avail:
                    try:
                        if metric in EC_EPS:
                            val = effective_complexity(vals[name],
                                                       eps=EC_EPS[metric])
                        else:
                            val = NP_IMPL[metric](vals[name])
                        row[f'{metric}__{name}'] = float(val)
                    except Exception as exc:                         # noqa: BLE001
                        row[f'{metric}__{name}'] = float('nan')
                        print(f'  [warn] {metric}/{name}: {exc}')
            rows.append(row)
            n_done += 1
            if n_done % args.log_every == 0 or n_done == 1:
                _k = ec_names[0][0]
                print(f'  [{n_done}] {row["img"]} '
                      f'EC(Lime)={row.get(f"{_k}__Lime", float("nan")):.1f} '
                      f'EC(our)={row.get(f"{_k}__our", float("nan")):.1f}')

    if not rows:
        print('没有成功评估的图像')
        return

    FINAL_METHODS = [m for m in EXP_METHODS if m not in DISABLED]
    if DISABLED:
        print(f'[info] 以下方法不可用，已从表中移除: {sorted(DISABLED)}')

    csv_path = os.path.join(args.out_dir, 'complexity_scores.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\n[info] 逐图分数 -> {csv_path}')

    sum_path = os.path.join(args.out_dir, 'complexity_summary.csv')
    with open(sum_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'direction', 'ec_eps'] + FINAL_METHODS)
        for metric in METRICS:
            vals = []
            for name in FINAL_METHODS:
                col = [r.get(f'{metric}__{name}', float('nan')) for r in rows]
                col = [c for c in col if not np.isnan(c)]
                vals.append(f'{np.mean(col):.4f}' if col else 'nan')
            w.writerow([metric,
                        'higher_better' if HIGHER_BETTER[metric] else 'lower_better',
                        EC_EPS.get(metric, '')]
                       + vals)
    print(f'[info] 汇总均值 -> {sum_path}')

    print('\n===== 各指标均值 =====')
    print(f"{'metric':<22}" + ''.join(f'{m:>9}' for m in FINAL_METHODS))
    for metric in METRICS:
        line = f'{metric:<22}'
        for name in FINAL_METHODS:
            col = [r.get(f'{metric}__{name}', float('nan')) for r in rows]
            col = [c for c in col if not np.isnan(c)]
            line += f'{np.mean(col):>9.4f}' if col else f'{"nan":>9}'
        print(line)

    # ---- 配对检验 ----
    print('\n===== 配对检验：our - Lime =====')
    tests = {}
    for metric in METRICS:
        a = np.array([r.get(f'{metric}__Lime', np.nan) for r in rows], dtype=float)
        b = np.array([r.get(f'{metric}__our', np.nan) for r in rows], dtype=float)
        tests[metric] = paired_test(a, b)
    for metric, pa in zip(METRICS, holm_bonferroni([tests[m]['p_raw'] for m in METRICS])):
        tests[metric]['p_holm'] = float(pa)

    print(f"{'metric':<22}{'n':>4}{'Lime':>16}{'our':>16}{'diff':>10}"
          f"{'p_holm':>10}{'d':>7}{'方向':>10}")
    print('-' * 95)
    for metric in METRICS:
        t = tests[metric]
        ma = f"{t['mean_a']:.4f}±{t['std_a']:.4f}" if t['n'] > 1 else f"{t['mean_a']:.4f}"
        mb = f"{t['mean_b']:.4f}±{t['std_b']:.4f}" if t['n'] > 1 else f"{t['mean_b']:.4f}"
        if t['p_holm'] >= 0.05:
            verdict = '无显著差异'
        else:
            better = (t['mean_diff'] > 0) if HIGHER_BETTER[metric] else (t['mean_diff'] < 0)
            verdict = 'our 更优' if better else 'Lime 更优'
        print(f"{metric:<22}{t['n']:>4}{ma:>16}{mb:>16}{t['mean_diff']:>+10.4f}"
              f"{t['p_holm']:>10.4f}{t['cohen_d']:>+7.3f}{verdict:>10}")
    print('-' * 95)

    tpath = os.path.join(args.out_dir, 'complexity_paired_tests.csv')
    with open(tpath, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'n', 'mean_Lime', 'std_Lime', 'mean_our',
                    'std_our', 'mean_diff', 'cohen_d', 'p_raw', 'p_holm',
                    'higher_better'])
        for metric in METRICS:
            t = tests[metric]
            w.writerow([metric, t['n'], t['mean_a'], t['std_a'], t['mean_b'],
                        t['std_b'], t['mean_diff'], t['cohen_d'],
                        t['p_raw'], t['p_holm'], HIGHER_BETTER[metric]])
    print(f'[info] 配对检验 -> {tpath}')
    print(f'\n[info] 图像数={n_done}（require_correct 跳过 {skipped} 张）'
          f'  用时={time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
