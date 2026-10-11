# -*- coding: utf-8 -*-
"""
MyVisualizeMulti.py —— ImageNet2012 验证集 × 多黑盒模型的可视化对比

参考 my_difmodel.py 的构图（行 = 黑盒模型，列 = Original / Activation / LIME / GradLIME），
改为从 ImageNet2012 验证集批量读取样本，由 --samples N 控制生成多少张结果图并落盘。

与单模型版 MyVisualize.py 的区别：
  * 一张结果图内横向对比多个黑盒模型（默认 5 个，均为 224 输入）
  * 模型只加载一次、解释完即释放，避免多模型常驻显存
  * 先用一次轻量前向筛选样本（--require-correct），再进入耗时的解释阶段
  * 支持两种版式：--layout model（行=模型）/ sample（行=样本）

用法示例：
    python MyVisualizeMulti.py --samples 5
    python MyVisualizeMulti.py --samples 10 --models resnet18 densenet121 \
        --require-correct --device cuda:0
    python MyVisualizeMulti.py --samples 3 --layout sample --num-samples 500

输出：
    <out-dir>/NNNN_<wnid>_<basename>.png      每个样本一张多模型对比图
    <out-dir>/manifest.csv                    逐样本 × 逐模型的预测记录
    <out-dir>/grid_all.png                    总览拼接图（可选）
"""

import os
import csv
import json
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

import matplotlib
matplotlib.use('Agg')            # 无显示环境必须提前设置
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM

from skimage.segmentation import mark_boundaries

warnings.filterwarnings('ignore')

IMG_EXT = ('.jpeg', '.jpg', '.png', '.bmp')

# 均为 224 输入的模型，分割与解释尺度一致，故作为默认集合
DEFAULT_MODELS = ['mobilenet_v2', 'alexnet', 'googlenet', 'resnet18', 'densenet121']
ALL_MODELS = DEFAULT_MODELS + ['resnet50', 'vgg16', 'inception_v3']
# "--models all / ALL_MODELS / ALL" 视为全部模型，免去逐个列名
MODEL_ALIASES = {'all', 'ALL', 'ALL_MODELS', 'all_models', 'ALL_MODELS'}


class ClassifierOutputTarget:
    """pytorch_grad_cam 的目标包装：取指定类别的分数。"""
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if isinstance(model_output, (tuple, list)):   # googlenet / inception 训练分支
            model_output = model_output[0]
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# ------------------------------------------------------------------ 模型构建
def build_model(name, device):
    """返回 (model, target_layers, img_size)。目标层沿用 my_difmodel.py 的设定。"""
    name = name.lower()
    if name == 'resnet18':
        m = models.resnet18(pretrained=True)
        return m, [m.layer4[-1]], 224
    if name == 'resnet50':
        m = models.resnet50(pretrained=True)
        return m, [m.layer4[-1]], 224
    if name == 'vgg16':
        m = models.vgg16(pretrained=True)
        return m, [m.features[-1]], 224
    if name == 'densenet121':
        m = models.densenet121(pretrained=True)
        return m, [m.features[-1]], 224
    if name == 'mobilenet_v2':
        m = models.mobilenet_v2(pretrained=True)
        return m, [m.features[17]], 224          # my_difmodel.py
    if name == 'alexnet':
        m = models.alexnet(pretrained=True)
        return m, [m.features[11]], 224
    if name == 'googlenet':
        m = models.googlenet(pretrained=True)
        return m, [m.inception5b], 224
    if name == 'inception_v3':
        m = models.inception_v3(pretrained=True)
        return m, [m.Mixed_7c], 299              # 299 输入，分割尺度与其余模型不同
    raise ValueError('未知模型: %s' % name)


def make_transforms(img_size):
    """trans_A: 模型输入张量；trans_B: LIME 邻域图张量；trans_C: 显示/分割用 PIL 图。"""
    trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    resize_to = 342 if img_size == 299 else img_size + 32
    trans_A = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        trans_norm,
    ])
    trans_B = transforms.Compose([transforms.ToTensor(), trans_norm])
    trans_C = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.CenterCrop(img_size),
    ])
    return trans_A, trans_B, trans_C


def forward_logits(model, batch):
    """统一处理返回 tuple 的模型（googlenet / inception 的训练分支）。"""
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# ------------------------------------------------------------------ 参数
def get_args():
    p = argparse.ArgumentParser(
        description='ImageNet2012 多黑盒模型可视化对比（LIME / GradLIME / CAM）')

    p.add_argument('--base-dir', default=r'F:\Databases\ImageNet2012/val',
                   help='ImageNet2012 验证集根目录（子目录为类别 wnid）')
    p.add_argument('--imagenet-json', default='imagenet/imagenet_class_index.json',
                   help='类别索引 json，用于取真实标签')

    p.add_argument('--samples', type=int, default=5,
                   help='生成多少张结果图（0 表示全部）')
    p.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                   choices=ALL_MODELS + sorted(MODEL_ALIASES),
                   help='参与对比的黑盒模型。可用 all / ALL_MODELS 表示全部 8 个；'
                        '默认 5 个均为 224 输入，分割尺度一致；'
                        'inception_v3 为 299 输入，分割尺度与其余不同')
    p.add_argument('--start', type=int, default=0, help='从第几个样本开始')
    p.add_argument('--shuffle', action='store_true',
                   help='按固定种子打乱样本顺序后再截取')
    p.add_argument('--random-seed', type=int, default=500,
                   help='分割与 LIME 的随机种子')

    p.add_argument('--require-correct', action='store_true',
                   help='只保留预测正确的样本（此时 --samples 指有效样本数）')
    p.add_argument('--correct-by', default='all', choices=['primary', 'all', 'any'],
                   help='正确性判定方式：基准模型 / 全部模型 / 任一模型')

    p.add_argument('--method', default='gradcam', help='CAM 方法')
    p.add_argument('--aug-smooth', action='store_true', help='CAM 测试时增强平滑')
    p.add_argument('--eigen-smooth', action='store_true', help='CAM 主成分平滑')

    p.add_argument('--num-samples', type=int, default=1000,
                   help='LIME / GradLIME 的邻域采样数 N')
    p.add_argument('--num-features', type=int, default=20, help='解释长度 K')
    p.add_argument('--positive-only', action='store_true',
                   help='只显示正向归因区（默认同时显示正负）')
    p.add_argument('--activation-mode', default='overlay', choices=['overlay', 'heat'],
                   help='Activation 列：叠加到原图 / 仅热力图')

    p.add_argument('--layout', default='model', choices=['model', 'sample'],
                   help='model: 每张图行=模型；sample: 每张图行=样本')
    p.add_argument('--grid-max', type=int, default=0,
                   help='总览拼接图最多含多少张（0 表示不生成）')
    p.add_argument('--out-dir', default='visual_result', help='结果图输出目录')
    p.add_argument('--device', default='cuda:0', help='cuda:0 / cpu')

    return p.parse_args()


# ------------------------------------------------------------------ 样本
def collect_samples(base_dir, shuffle, seed, start):
    """返回 [(wnid, abs_path), ...]，按类别目录与文件名排序，保证可复现。"""
    classes = [d for d in sorted(os.listdir(base_dir))
               if os.path.isdir(os.path.join(base_dir, d))]
    items = []
    for wnid in classes:
        cdir = os.path.join(base_dir, wnid)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(IMG_EXT):
                items.append((wnid, os.path.join(cdir, fn)))
    if shuffle:
        rng = np.random.RandomState(seed)
        items = [items[i] for i in rng.permutation(len(items))]
    return items[start:]


def cam_to_uint8(cam):
    cam = np.asarray(cam, dtype=np.float32)
    mn, mx = float(cam.min()), float(cam.max())
    if mx - mn < 1e-12:
        return np.zeros(cam.shape, dtype=np.uint8)
    return ((cam - mn) / (mx - mn) * 255.0).astype(np.uint8)


# ------------------------------------------------------------------ 阶段 1：轻量预测筛选
def scan_predictions(args, items, device, cls2idx, model_names):
    """逐个模型加载，对所有候选样本做一次前向，返回 preds[model] = [int, ...]。"""
    preds = {}
    for name in model_names:
        model, _, img_size = build_model(name, device)
        model = model.eval().to(device)
        trans_A, _, _ = make_transforms(img_size)

        plist = []
        bs = 32
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            batch = []
            for _, path in chunk:
                try:
                    img = Image.open(path).convert('RGB')
                except Exception:
                    img = Image.new('RGB', (img_size, img_size))
                batch.append(trans_A(img))
            with torch.no_grad():
                logits = forward_logits(model, torch.stack(batch).to(device))
                plist.extend(logits.argmax(dim=1).detach().cpu().tolist())

        preds[name] = plist
        del model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        print('[scan] %-14s 完成 %d 张' % (name, len(plist)))
    return preds


def select_indices(args, items, preds, cls2idx, model_names):
    """按 --require-correct / --correct-by 与 --samples 选出样本下标。"""
    primary = 'resnet18' if 'resnet18' in model_names else model_names[0]
    keep = []
    for i, (wnid, _) in enumerate(items):
        label = cls2idx.get(wnid, -1)
        if args.require_correct and label >= 0:
            ok = {'primary': preds[primary][i] == label,
                  'all': all(preds[m][i] == label for m in model_names),
                  'any': any(preds[m][i] == label for m in model_names)}[args.correct_by]
            if not ok:
                continue
        keep.append(i)
    if args.samples > 0:
        keep = keep[:args.samples]
    return keep, primary


# ------------------------------------------------------------------ 阶段 2：解释
def explain_one_model(args, name, device, use_cuda, samples_pil, cam_methods):
    """对一批 PIL 图，用单个模型依次算出 CAM / LIME / GradLIME。

    返回 list[(cam_u8, lime_img, our_img, pred_id)]，与 samples_pil 等长。
    模型在本函数内加载、结束时释放，避免多模型常驻显存。
    """
    from lime import lime_image, lime_image_my

    model, target_layers, img_size = build_model(name, device)
    model = model.eval().to(device)
    trans_A, trans_B, trans_C = make_transforms(img_size)

    def batch_predict(images):
        batch = torch.stack(tuple(trans_B(i) for i in images), dim=0).to(device)
        with torch.no_grad():
            logits = forward_logits(model, batch)
        return F.softmax(logits, dim=1).detach().cpu().numpy()

    cam_algorithm = cam_methods[args.method]
    out = []

    for img_pil in samples_pil:
        input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = forward_logits(model, input_tensor)
            pred_id = int(F.softmax(logits, dim=1).argmax(dim=1).item())

        # ---- CAM
        with cam_algorithm(model=model, target_layers=target_layers,
                           use_cuda=use_cuda) as cam:
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[ClassifierOutputTarget(pred_id)],
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)[0, :]
        cam_u8 = cam_to_uint8(grayscale_cam)

        # ---- 分割与解释统一用 224/299 的 PIL 图
        img_arr = np.array(trans_C(img_pil))

        # ---- 原始 LIME
        explainer_org = lime_image.LimeImageExplainer()
        explanation_org = explainer_org.explain_instance(
            img_arr, batch_predict, top_labels=1, hide_color=0,
            num_samples=args.num_samples, num_features=args.num_features,
            random_seed=args.random_seed)

        # ---- GradLIME：先取数据与标签，再传入 CAM
        explainer = lime_image_my.LimeImageExplainer()
        explainer.explain_instance_data_label(
            img_arr, batch_predict, top_labels=1, hide_color=0,
            num_samples=args.num_samples, num_features=args.num_features,
            random_seed=args.random_seed)
        explanation = explainer.explain_instance(grayscale_cam)

        pos_only = args.positive_only
        nf = args.num_features
        _, mask_o = explanation_org.get_image_and_mask(
            explanation_org.top_labels[0], positive_only=pos_only,
            num_features=nf, hide_rest=False)
        img_boundry_org = mark_boundaries(img_arr / 255.0, mask_o)

        _, mask_u = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=pos_only,
            num_features=nf, hide_rest=False)
        img_boundry_our = mark_boundaries(img_arr / 255.0, mask_u)

        out.append((cam_u8, img_boundry_org, img_boundry_our, pred_id))

    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ 绘图
def _draw_row(axes_row, img_rgb, cam_u8, lime_img, our_img, row_label,
              activation_mode, show_titles):
    panels = [
        ('Original', img_rgb, None),
        ('Activation', None, cam_u8),
        ('LIME', lime_img, None),
        ('GradLIME', our_img, None),
    ]
    for ax, (name, arr, cam) in zip(axes_row, panels):
        if name == 'Activation':
            if activation_mode == 'overlay':
                ax.imshow(img_rgb)
                ax.imshow(cam, cmap='jet', alpha=0.5, vmin=0, vmax=255)
            else:
                ax.imshow(cam, cmap='jet', vmin=0, vmax=255)
        else:
            ax.imshow(arr)
        if show_titles:
            ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes_row[0].set_ylabel(row_label, fontsize=9, rotation=90, labelpad=6)


def save_figure_model_layout(out_path, img_rgb, per_model, model_names,
                             title, activation_mode):
    """行 = 模型，列 = Original / Activation / LIME / GradLIME。"""
    n = len(model_names)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.1 * n))
    axes = np.atleast_2d(axes)
    for r, name in enumerate(model_names):
        cam_u8, lime_img, our_img, pred = per_model[name]
        _draw_row(axes[r], img_rgb, cam_u8, lime_img, our_img,
                  '%s\npred=%d' % (name, pred), activation_mode, show_titles=(r == 0))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_figure_sample_layout(out_path, model_name, rows, activation_mode):
    """行 = 样本，列 = Original / Activation / LIME / GradLIME（单模型多样本）。"""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.1 * n))
    axes = np.atleast_2d(axes)
    for r, (base, img_rgb, cam_u8, lime_img, our_img, pred) in enumerate(rows):
        _draw_row(axes[r], img_rgb, cam_u8, lime_img, our_img,
                  '%s\npred=%d' % (base[:18], pred), activation_mode,
                  show_titles=(r == 0))
    fig.suptitle('%s' % model_name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def make_grid(paths, out_path):
    if not paths:
        return
    imgs = [Image.open(p).convert('RGB') for p in paths]
    w = max(im.width for im in imgs)
    h = sum(im.height for im in imgs)
    canvas = Image.new('RGB', (w, h), 'white')
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(out_path)
    print('[save] 总览图 -> %s (%dx%d)' % (out_path, canvas.width, canvas.height))


# ------------------------------------------------------------------ 主流程
def main():
    args = get_args()
    t0 = time.time()

    device = torch.device(args.device if torch.cuda.is_available()
                          or str(args.device).startswith('cpu') else 'cpu')
    use_cuda = str(device).startswith('cuda')
    print('device:', device)

    os.makedirs(args.out_dir, exist_ok=True)

    cls2idx, idx2label = {}, []
    if os.path.exists(args.imagenet_json):
        with open(os.path.abspath(args.imagenet_json), 'r') as f:
            class_idx = json.load(f)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}
        print('已加载类别索引，共 %d 类' % len(cls2idx))
    else:
        print('[warn] 未找到 %s，将无法判断预测正确性' % args.imagenet_json)

    items = collect_samples(args.base_dir, args.shuffle, args.random_seed, args.start)
    if not items:
        print('[error] %s 下未找到样本' % args.base_dir)
        return
    print('目录样本总数: %d' % len(items))

    # 展开 all / ALL_MODELS 等别名；去重并保持 ALL_MODELS 中的顺序
    if any(m.lower() in {a.lower() for a in MODEL_ALIASES} for m in args.models):
        model_names = list(ALL_MODELS)
    else:
        model_names = [m for m in ALL_MODELS if m in set(args.models)]
        # 保留用户未通过校验的拼写（理论上不会发生），仅作兜底
        for m in args.models:
            if m not in model_names and m not in MODEL_ALIASES:
                model_names.append(m)
    if not model_names:
        model_names = list(DEFAULT_MODELS)
    print('参与对比的模型(%d): %s' % (len(model_names), ', '.join(model_names)))
    print('预计前向次数: %d 图 × %d 模型 × 2 × %d 采样 ≈ %d 次'
          % (min(args.samples, len(items)) if args.samples > 0 else len(items),
             len(model_names), args.num_samples,
             (min(args.samples, len(items)) if args.samples > 0 else len(items))
             * len(model_names) * 2 * args.num_samples))

    # ---- 阶段 1：轻量预测筛选
    scan_pool = items
    if args.samples > 0 and (args.require_correct or args.shuffle):
        scan_pool = items[:max(args.samples * 12, args.samples + 200)]
    elif args.samples > 0:
        scan_pool = items[:args.samples]
    print('[阶段1] 预测扫描候选 %d 张 ...' % len(scan_pool))
    preds = scan_predictions(args, scan_pool, device, cls2idx, model_names)

    keep_idx, primary = select_indices(args, scan_pool, preds, cls2idx, model_names)
    if not keep_idx:
        print('[error] 没有符合条件的样本（可放宽 --correct-by 或去掉 --require-correct）')
        return
    print('[阶段1] 选中 %d 张（正确性基准模型: %s，判定方式: %s）'
          % (len(keep_idx), primary,
             args.correct_by if args.require_correct else '不过滤'))

    # ---- 载入选中样本
    sel_paths = [scan_pool[i] for i in keep_idx]
    samples_pil, bases, wnids = [], [], []
    for wnid, path in sel_paths:
        try:
            samples_pil.append(Image.open(path).convert('RGB'))
        except Exception as e:
            print('[skip] 读取失败 %s: %s' % (path, e))
            continue
        bases.append(os.path.splitext(os.path.basename(path))[0])
        wnids.append(wnid)
    if not samples_pil:
        print('[error] 所有样本读取失败')
        return

    cam_methods = {
        "gradcam": GradCAM
    }

    # ---- 阶段 2：逐模型解释
    results = {}      # results[model] = [(cam, lime, our, pred), ...]
    for name in model_names:
        t1 = time.time()
        results[name] = explain_one_model(args, name, device, use_cuda,
                                          samples_pil, cam_methods)
        print('[阶段2] %-14s 完成 %d 张，用时 %.1fs' % (name, len(results[name]),
                                                        time.time() - t1))

    # ---- 阶段 3：出图
    saved_paths, rows = [], []
    for i, (wnid, base) in enumerate(zip(wnids, bases)):
        img_pil = samples_pil[i]
        per_model = {}
        for name in model_names:
            cam_u8, lime_img, our_img, pred = results[name][i]
            per_model[name] = (cam_u8, lime_img, our_img, pred)

        gt = cls2idx.get(wnid, -1)
        pred0 = per_model[primary][3]
        tag = 'correct' if gt == pred0 else 'wrong'
        title = '%s | gt=%s(%s) | %s pred=%d %s' % (
            base, wnid, str(gt), primary, pred0,
            ('[%s]' % tag) if gt >= 0 else '')

        out_name = '%04d_%s_%s.png' % (i, wnid, base)
        out_path = os.path.join(args.out_dir, out_name)
        save_figure_model_layout(out_path, np.array(img_pil), per_model,
                                 model_names, title, args.activation_mode)
        saved_paths.append(out_path)

        rec = {'idx': i, 'file': sel_paths[i][1], 'wnid': wnid,
               'label': gt, 'figure': out_name}
        for name in model_names:
            rec['pred_%s' % name] = per_model[name][3]
            rec['ok_%s' % name] = int(per_model[name][3] == gt)
        rows.append(rec)
        print('[%d/%d] %s -> %s' % (i + 1, len(wnids), base, out_name))

        # 可选：每个模型一张"行=样本"的图
        if args.layout == 'sample':
            for name in model_names:
                cam_u8, lime_img, our_img, pred = per_model[name]
                sp = os.path.join(args.out_dir, 'by_model',
                                  '%s_%04d_%s.png' % (name, i, base))
                os.makedirs(os.path.dirname(sp), exist_ok=True)
                save_figure_sample_layout(
                    sp, name,
                    [(base, np.array(img_pil), cam_u8, lime_img, our_img, pred)],
                    args.activation_mode)

    # ---- manifest
    fields = ['idx', 'file', 'wnid', 'label', 'figure']
    for name in model_names:
        fields += ['pred_%s' % name, 'ok_%s' % name]
    manifest_path = os.path.join(args.out_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print('[save] 清单 -> %s' % manifest_path)

    if args.grid_max > 0 and saved_paths:
        make_grid(saved_paths[:args.grid_max],
                  os.path.join(args.out_dir, 'grid_all.png'))

    print('\n完成：%d 张结果图，%d 个模型，用时 %.1fs，输出 %s'
          % (len(saved_paths), len(model_names), time.time() - t0,
             os.path.abspath(args.out_dir)))


if __name__ == '__main__':
    main()
