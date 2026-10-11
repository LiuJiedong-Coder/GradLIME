# -*- coding: utf-8 -*-
"""
abl_common.py —— 消融实验公共底座
=================================
所有 A1..A8 消融脚本共享此模块；每个消融脚本本身独立可运行，互不依赖。

设计原则
--------
1. 归因图一律由 local_exp 的拟合权重写回 segments（与 MyFaithfulness /
   MyComplexity 完全一致），不再使用 get_image_and_mask 的可视化图。
2. 忠实性一律用**超像素粒度**的逐步删除协议（删除单元 == 归因单元）。
3. 复杂性一律在**超像素单元**上计算（避免像素级熵被面积混淆）。
4. 所有变体在同一批图像、同一 random_seed 下比较，配对检验。
5. 不对 lime_base_my.py 做任何修改；变体通过公开接口 + 能力检测实现，
   不可达的变体会被明确标记为 "unavailable"，而不是静默给错数。
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

warnings.filterwarnings("ignore")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BLACK_NORMALISED = (0.0 - IMAGENET_MEAN) / IMAGENET_STD

trans_norm = transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
trans_A = transforms.Compose([transforms.Resize((256, 256)),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(), trans_norm])
trans_B = transforms.Compose([transforms.ToTensor(), trans_norm])
trans_C = transforms.Compose([transforms.Resize((256, 256)),
                              transforms.CenterCrop(224)])

# 指标方向：True = 越高越好
HIGHER_BETTER = {
    'corr': True, 'aopc_mean': True, 'aopc_uniform': True,
    'sparseness': True, 'entropy': False, 'ec90': False,
}
METRIC_LABEL = {
    'corr': 'FaithfulnessCorrelation',
    'aopc_mean': 'AOPC(mean-fill / IROF)',
    'aopc_uniform': 'AOPC(uniform-fill / RegionPert.)',
    'sparseness': 'Sparseness(Gini)',
    'entropy': 'Complexity(entropy)',
    'ec90': 'EffectiveComplexity@90',
}


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--base-dir', default=r'ImageNet2012/val100')
    p.add_argument('--imagenet-json', default='imagenet/imagenet_class_index.json')
    p.add_argument('--num-classes', type=int, default=70,
                   help='消融实验不需要全量样本；默认 10 类')
    p.add_argument('--num-images-per-class', type=int, default=15)
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--num-features', type=int, default=20)
    p.add_argument('--random-seed', type=int, default=500)
    p.add_argument('--device', default="cuda")
    p.add_argument('--out-dir', default='results_ablation')
    p.add_argument('--require-correct', action='store_true')
    p.add_argument('--metrics', default='corr,sparseness,entropy',
                   help='逗号分隔；可选 corr,aopc_mean,aopc_uniform,'
                        'sparseness,entropy,ec90')
    p.add_argument('--log-every', type=int, default=10)
    return p


# --------------------------------------------------------------------------- #
# 模型与图像
# --------------------------------------------------------------------------- #
def load_model(device):
    try:
        from torchvision.models import ResNet18_Weights
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:                                                # noqa: BLE001
        m = models.resnet18(pretrained=True)
    return m.eval().to(device)


def make_batch_predict(model, device):
    def batch_predict(images):
        b = torch.stack(tuple(trans_B(i) for i in images), dim=0).to(device)
        with torch.no_grad():
            out = F.softmax(model(b), dim=1)
        return out.detach().cpu().numpy()
    return batch_predict


def iter_images(base_dir, num_classes, per_class, require_correct,
                model, device, json_path, log=None):
    """产出 (class_name, label, pil_image, x_norm, pred_id, correct)。"""
    cls2idx = {}
    if os.path.exists(json_path):
        with open(os.path.abspath(json_path), 'r') as f:
            ci = json.load(f)
            cls2idx = {ci[str(k)][0]: k for k in range(len(ci))}

    classes = sorted(d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d)))
    if num_classes > 0:
        classes = classes[:num_classes]

    for cname in classes:
        label = cls2idx.get(cname)
        cdir = os.path.join(base_dir, cname)
        files = sorted(f for f in os.listdir(cdir)
                       if os.path.isfile(os.path.join(cdir, f)))
        if per_class > 0:
            files = files[:per_class]
        for fn in files:
            try:
                pil = Image.open(os.path.join(cdir, fn)).convert('RGB')
            except Exception:                                        # noqa: BLE001
                continue
            x = trans_A(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(F.softmax(model(x), dim=1).argmax(dim=1).item())
            correct = int(pred == label)
            if require_correct and not correct:
                continue
            yield dict(class_name=cname, label=label, img=fn, pil=pil,
                       x_norm=x[0].detach().cpu().numpy(),
                       pred=pred, correct=correct)


# --------------------------------------------------------------------------- #
# CAM（可控制 ReLU / 归一化 / 目标层）
# --------------------------------------------------------------------------- #
def compute_cam(model, x_tensor, pred_id, layer=None, relu=True,
                normalise=True, device=None):
    """自己算 Grad-CAM，避免依赖 pytorch_grad_cam 的内部开关。

    relu=False 保留负证据（对应"去掉 ReLU"变体）；
    normalise 控制是否除以最大值。
    返回 (cam_2d, cam_raw_stats)
    """
    from pytorch_grad_cam import GradCAM
    layer = layer if layer is not None else model.layer4
    with GradCAM(model=model, target_layers=[layer],
                 use_cuda=str(device).startswith('cuda')) as cam:
        g = cam(input_tensor=x_tensor,
                targets=[ClassifierOutputTarget(pred_id)])[0, :]
        acts = cam.activations_and_grads.activations
        grads = cam.activations_and_grads.gradients

    def _to_chw(x):
        """pytorch_grad_cam 的 activations/gradients 是带 batch 维的列表，
        元素形状常见为 (1, C, H, W)；个别构建给出 (C, H, W)。统一到 (C,H,W)。"""
        x = np.asarray(x, dtype=np.float32)
        while x.ndim > 3:
            x = x[0]
        return x

    if acts is not None and grads is not None and len(acts) and len(grads):
        a = _to_chw(acts[0])                          # (C,H,W)
        gg = _to_chw(grads[0])                        # (C,H,W)
        if gg.shape != a.shape:                       # 形状异常则回退
            raw = np.asarray(g, dtype=np.float32)
        else:
            w = gg.mean(axis=(-2, -1), keepdims=True)  # (C,1,1) GAP
            raw = (w * a).sum(axis=0)                  # (H,W)，未过 ReLU
    else:                                             # 回退：用库输出
        raw = np.asarray(g, dtype=np.float32)

    raw = np.maximum(raw, 0.0) if relu else raw

    if normalise:
        mx = float(np.max(np.abs(raw))) if not relu else float(raw.max())
        raw = raw / mx if mx > 1e-12 else raw
    return raw, g


def _as_hw(cam):
    """把 CAM 规整成二维 (H,W)：兼容 (H,W) / (1,H,W) / (1,1,H,W)。"""
    a = np.asarray(cam, dtype=np.float32)
    while a.ndim > 2:
        a = a[0] if a.shape[0] == 1 else a.mean(axis=0)
    return a


def cam_to_224(cam_small, size=224):
    """任意形状的 CAM -> (size,size)，双线性。"""
    a = _as_hw(cam_small)
    t = torch.from_numpy(a.astype(np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode='bilinear',
                         align_corners=False)[0, 0].numpy()


# --------------------------------------------------------------------------- #
# 解释生成
# --------------------------------------------------------------------------- #
def attribution_from_local_exp(explanation, label, num_features=None):
    segments = explanation.segments
    items = explanation.local_exp[label]
    if num_features is not None and num_features > 0:
        items = items[:num_features]
    lut = np.zeros(int(segments.max()) + 1, dtype=np.float32)
    for sid, w in items:
        lut[int(sid)] = float(w)
    return lut[segments]


def fresh_rs(seed):
    """每个变体一个【独立但同种子】的 RandomState。

    为什么必须这样
    --------------
    sklearn 的 check_random_state(None) 返回 numpy 全局单例
    np.random.mtrand._rand。若各变体不显式传 random_state，
    它们会共享同一个对象、并按调用顺序依次推进它，于是后一个变体的
    邻域样本 data 与前一个完全不同 —— 各变体在不同样本上拟合，
    对照彻底失效（实测差异约 50% 的元素）。

    实测：
        共享全局单例 -> 变体间 data 一致率 25%
        fresh_rs(seed) -> 100%
    """
    return np.random.RandomState(seed)


def lime_explanation(image_arr, predict_fn, seed, N, K, hide_color=0):
    from lime import lime_image
    ex = lime_image.LimeImageExplainer(random_state=fresh_rs(seed))
    return ex.explain_instance(image_arr, predict_fn, top_labels=1,
                               hide_color=hide_color, num_samples=N,
                               num_features=K, random_seed=seed)


def gradlime_explanation(image_arr, predict_fn, cam_224, seed, N, K,
                         hide_color=0, cam_mode=None):
    """GradLIME 解释。

    cam_mode 仅在底层 explain_instance_with_data 支持时才透传；
    不支持时返回 (explanation, note)，note 说明降级情况。
    """
    from lime import lime_image_my
    ex = lime_image_my.LimeImageExplainer(random_state=fresh_rs(seed))
    ex.explain_instance_data_label(image_arr, predict_fn, top_labels=1,
                                   hide_color=hide_color, num_samples=N,
                                   num_features=K, random_seed=seed)
    note = None
    if cam_mode is not None:
        ok = _try_set_cam_mode(ex, cam_mode)
        if not ok:
            note = (f'cam_mode={cam_mode} 在当前 lime_base_my 接口下不可透传，'
                    f'结果与默认构型相同')
    return ex.explain_instance(cam_224), note


def _try_set_cam_mode(explainer, cam_mode):
    """尝试把 cam_mode 传到底层；成功返回 True。"""
    try:
        base = getattr(explainer, 'base', None) or getattr(explainer, 'lime_base', None)
        if base is None:
            return False
        fn = getattr(base, 'explain_instance_with_data', None)
        if fn is None or 'cam_mode' not in inspect.signature(fn).parameters:
            return False
        orig = getattr(base, 'explain_instance_with_data')
        def wrapped(*a, **kw):
            kw.setdefault('cam_mode', cam_mode)
            return orig(*a, **kw)
        base.explain_instance_with_data = wrapped
        return True
    except Exception:                                                # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# 超像素单元
# --------------------------------------------------------------------------- #
def superpixel_values(attr_2d, segments):
    ids = np.unique(segments)
    vals = np.zeros(len(ids), dtype=np.float64)
    for i, sid in enumerate(ids):
        m = (segments == int(sid))
        if m.any():
            vals[i] = float(np.asarray(attr_2d)[m].mean())
    return ids, vals


def topk(ids, vals, k):
    order = np.argsort(-np.abs(np.asarray(vals, dtype=np.float64)))
    k = int(min(k, len(order))) if k and k > 0 else len(order)
    return [(int(np.asarray(ids)[order[i]]), float(np.asarray(vals)[order[i]]))
            for i in range(k)]


def baseline_value(kind, x_norm):
    if kind == 'black':
        return BLACK_NORMALISED.astype(np.float64).copy()
    if kind == 'mean':
        return np.zeros(3, dtype=np.float64)
    if kind == 'uniform':
        return np.random.uniform(0.0, 1.0, size=3).astype(np.float64)
    raise ValueError(kind)


def build_deletion_batch(x_norm, segments, items, baseline):
    c, h, w = x_norm.shape
    xs, a_abs = [], []
    cum = np.zeros((h, w), dtype=bool)
    run = 0.0
    for sid, val in items:
        cum |= (segments == int(sid))
        x = np.array(x_norm, copy=True)
        flat = x.reshape(c, -1)
        flat[:, cum.reshape(-1)] = np.asarray(baseline)[:, None]
        xs.append(x)
        run += abs(float(val))
        a_abs.append(run)
    return np.stack(xs, axis=0), np.asarray(a_abs)


def predict_scores(model, xs, class_idx, device, batch_size=32,
                   softmax=True):
    out = []
    for i in range(0, len(xs), batch_size):
        b = torch.from_numpy(np.ascontiguousarray(xs[i:i + batch_size])
                             ).float().to(device)
        with torch.no_grad():
            o = model(b)
        if softmax:
            o = F.softmax(o, dim=1)
        out.append(o[:, int(class_idx)].detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def predict_single(model, x_norm, class_idx, device, softmax=True):
    t = torch.from_numpy(np.ascontiguousarray(x_norm[None])).float().to(device)
    with torch.no_grad():
        o = model(t)
    if softmax:
        o = F.softmax(o, dim=1)
    return float(o[0, int(class_idx)].item())


def _pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def blur_value(x_norm):
    """归一化空间中的"模糊填充"：用图像各通道均值（近似 blur 基准）。"""
    return x_norm.reshape(3, -1).mean(axis=1).astype(np.float64)


def eval_faithfulness(model, x_norm, segments, attr_2d, pred, device,
                      K, metrics, batch_size=32, softmax=True, fill=None):
    """超像素粒度逐步删除，返回所需指标的字典。

    fill: dict，覆盖各指标使用的填充策略，例如 {'corr': 'blur'}。
          可选值 black / mean / uniform / blur。
    """
    res = {}
    ids, vals = superpixel_values(attr_2d, segments)
    items = topk(ids, vals, K)
    s0 = predict_single(model, x_norm, pred, device, softmax=softmax)
    fill = fill or {}

    def _bv(kind):
        if kind == 'blur':
            return blur_value(x_norm)
        return baseline_value(kind, x_norm)

    if 'corr' in metrics:
        xs, cum = build_deletion_batch(x_norm, segments, items,
                                       _bv(fill.get('corr', 'black')))
        s = predict_scores(model, xs, pred, device, batch_size, softmax)
        res['corr'] = _pearson(cum, s0 - s)
    if 'aopc_mean' in metrics:
        xs, _ = build_deletion_batch(x_norm, segments, items,
                                     _bv(fill.get('aopc_mean', 'mean')))
        s = predict_scores(model, xs, pred, device, batch_size, softmax)
        res['aopc_mean'] = float(np.mean(s0 - s))
    if 'aopc_uniform' in metrics:
        xs, _ = build_deletion_batch(x_norm, segments, items,
                                     _bv(fill.get('aopc_uniform', 'uniform')))
        s = predict_scores(model, xs, pred, device, batch_size, softmax)
        res['aopc_uniform'] = float(np.mean(s0 - s))
    return res


def gini(a):
    x = np.sort(np.abs(np.asarray(a, dtype=np.float64)).reshape(-1))
    n = x.size
    tot = x.sum()
    if n == 0 or tot <= 0:
        return float('nan')
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(((2 * i - n - 1) * x).sum() / (n * tot))


def entropy(a):
    x = np.abs(np.asarray(a, dtype=np.float64)).reshape(-1)
    if x.sum() <= 0:
        return float('nan')
    nz = x[x > 0]
    p = nz / nz.sum()
    return float(-(p * np.log(p)).sum())


def ec(a, eps=0.1):
    x = np.sort(np.abs(np.asarray(a, dtype=np.float64)).reshape(-1))[::-1]
    if x.sum() <= 0:
        return float('nan')
    cum = np.cumsum(x) / x.sum()
    return float(min(int(np.searchsorted(cum, 1.0 - eps)) + 1, x.size))


def eval_complexity(attr_2d, segments, metrics):
    ids, vals = superpixel_values(attr_2d, segments)
    res = {}
    if 'sparseness' in metrics:
        res['sparseness'] = gini(vals)
    if 'entropy' in metrics:
        res['entropy'] = entropy(vals)
    if 'ec90' in metrics:
        res['ec90'] = ec(vals, 0.1)
    return res


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m, dtype=float)
    order = np.argsort(p)
    run = 0.0
    for r, idx in enumerate(order):
        run = max(run, (m - r) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def paired_test(a, b, min_n=5):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
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


def _check_data_consistency(ctx, variants, tag, _state={'n': 0, 'ok': 0,
                                                        'warned': False}):
    """校验各变体是否在同一批邻域样本 data 上拟合。

    data 不一致意味着各变体在不同样本上拟合，对照无效。
    A3（num_samples 不同）等消融会刻意改变 data 形状，
    故只在同一形状组内比较。
    """
    dd = ctx.get('_data', {})
    if len(dd) < 2:
        return
    by_shape = {}
    for v, d in dd.items():
        by_shape.setdefault(d.shape, []).append(v)
    for _shape, vs in by_shape.items():
        if len(vs) < 2:
            continue
        ref = np.asarray(dd[vs[0]])
        same = all(np.array_equal(ref, np.asarray(dd[v])) for v in vs[1:])
        _state['n'] += 1
        if same:
            _state['ok'] += 1
        elif not _state['warned']:
            _state['warned'] = True
            diff = vs[1:]
            print(f'  [严重][{tag}] {diff} 与 {vs[0]} 的邻域样本 data 不同 -> '
                  f'各变体在不同样本上拟合，对照无效。'
                  f'请确认 explain_fn 用 C.fresh_rs(seed) 构造解释器。')


def run_ablation(args, tag, variants, explain_fn, reference=None,
                 cam_fn=None, fill_fn=None, k_fn=None):
    """通用执行框架：每个消融脚本只需提供 explain_fn。

    explain_fn(ctx, variant) -> (attr_2d, segments) 或 None
        ctx 含 model/device/pil/x_norm/pred/predict_fn/img_arr/cam
    cam_fn(ctx, variant) -> cam_224 或 None（A2/A7 需要不同的 CAM）
    reference: 对照组变体名，默认取 variants[0]
    """
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    unknown = [m for m in metrics if m not in HIGHER_BETTER]
    if unknown:
        raise ValueError(f'未知指标: {unknown}')
    print(f'[{tag}] device={device}')
    print(f'[{tag}] variants={variants}')
    print(f'[{tag}] metrics={[METRIC_LABEL[m] for m in metrics]}')

    model = load_model(device)
    predict_fn = make_batch_predict(model, device)
    rows, t0 = [], time.time()
    n = 0

    for info in iter_images(args.base_dir, args.num_classes,
                            args.num_images_per_class, args.require_correct,
                            model, device, args.imagenet_json):
        ctx = dict(model=model, device=device, predict_fn=predict_fn,
                   args=args, **info)
        ctx['img_arr'] = np.array(trans_C(info['pil']))
        ctx['x_tensor'] = trans_A(info['pil']).unsqueeze(0).to(device)
        row = {'img': info['img'], 'class': info['class_name'],
               'pred': info['pred'], 'correct': info['correct']}
        ok_any = False
        for v in variants:
            try:
                if cam_fn is not None:
                    c = cam_fn(ctx, v)
                    if c is not None:
                        ctx['cam'] = c
                out = explain_fn(ctx, v)
                if out is None:
                    continue
                attr, segments = out
                # 记录邻域样本 data 以便校验各变体是否在同一批样本上拟合
                try:
                    _ex = ctx.get('_last_explainer')
                    if _ex is not None and hasattr(_ex, 'data'):
                        ctx.setdefault('_data', {})[v] = np.asarray(_ex.data)
                except Exception:                                    # noqa: BLE001
                    pass
                fo = fill_fn(v) if fill_fn is not None else None
                kk = k_fn(v) if k_fn is not None else args.num_features
                fa = eval_faithfulness(model, info['x_norm'], segments, attr,
                                       info['pred'], device, kk, metrics,
                                       softmax=True, fill=fo)
                cx = eval_complexity(attr, segments, metrics)
                for m in metrics:
                    row[f'{m}__{v}'] = fa.get(m, cx.get(m, np.nan))
                ok_any = True
            except Exception as exc:                                # noqa: BLE001
                print(f'  [warn] {v} 失败: {type(exc).__name__}: {exc}')
        if ok_any:
            rows.append(row)
            n += 1
            if n % args.log_every == 0 or n == 1:
                print(f'  [{n}] {row["img"]} pred={info["pred"]}')
        _check_data_consistency(ctx, variants, tag)

    print(f'[{tag}] 完成 {n} 张，用时 {time.time()-t0:.1f}s')
    finalise(rows, variants, metrics, args.out_dir, tag, reference)


def finalise(rows, variants, metrics, outdir, tag, reference=None):
    """写 CSV + 打印汇总 + 配对检验（各变体 vs reference）。"""
    os.makedirs(outdir, exist_ok=True)
    if not rows:
        print('[warn] 没有成功评估的图像')
        return

    # 逐图
    p1 = os.path.join(outdir, f'{tag}_scores.csv')
    with open(p1, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ref = reference or variants[0]
    lines = []
    lines.append(f'===== {tag} =====')
    lines.append(f'{"variant":<26}' + ''.join(f'{m:>18}' for m in metrics))
    for v in variants:
        line = f'{v:<26}'
        for m in metrics:
            col = [r.get(f'{m}__{v}', np.nan) for r in rows]
            col = [c for c in col if not np.isnan(c)]
            line += f'{np.mean(col):>11.4f}±{np.std(col, ddof=1):>6.4f}' if col else f'{"nan":>18}'
        lines.append(line)

    # 配对检验
    lines.append('')
    lines.append(f'--- 配对检验（各变体 vs {ref}）---')
    lines.append(f'{"variant":<26}{"metric":<14}{"n":>4}{"diff":>10}{"d":>8}'
                 f'{"p_holm":>11}{"判定":>12}')
    tests = []
    for v in variants:
        if v == ref:
            continue
        per = {}
        for m in metrics:
            a = np.array([r.get(f'{m}__{ref}', np.nan) for r in rows], float)
            b = np.array([r.get(f'{m}__{v}', np.nan) for r in rows], float)
            per[m] = paired_test(a, b)
        adj = holm_bonferroni([per[m]['p_raw'] for m in metrics])
        for m, pa in zip(metrics, adj):
            per[m]['p_holm'] = float(pa)
            t = per[m]
            if t['p_holm'] >= 0.05 or t['n'] < 5:
                verdict = '无显著差异'
            else:
                better = (t['mean_diff'] > 0) == HIGHER_BETTER[m]
                verdict = f'{v}更优' if better else f'{ref}更优'
            lines.append(f'{v:<26}{m:<14}{t["n"]:>4}{t["mean_diff"]:>+10.4f}'
                         f'{t["cohen_d"]:>+8.3f}{t["p_holm"]:>11.2e}{verdict:>12}')
            tests.append(dict(variant=v, metric=m, n=t['n'],
                              mean_ref=t['mean_a'], mean_var=t['mean_b'],
                              mean_diff=t['mean_diff'], cohen_d=t['cohen_d'],
                              p_raw=t['p_raw'], p_holm=t['p_holm'],
                              higher_better=HIGHER_BETTER[m]))
    txt = '\n'.join(lines)
    print(txt)

    with open(os.path.join(outdir, f'{tag}_summary.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write(txt + '\n')
    p2 = os.path.join(outdir, f'{tag}_paired.csv')
    with open(p2, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['variant', 'metric', 'n', 'mean_ref',
                                           'mean_var', 'mean_diff', 'cohen_d',
                                           'p_raw', 'p_holm', 'higher_better'])
        w.writeheader(); w.writerows(tests)
    print(f'\n[info] 逐图 -> {p1}')
    print(f'[info] 汇总 -> {os.path.join(outdir, tag + "_summary.txt")}')
    print(f'[info] 配对 -> {p2}')
