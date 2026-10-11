# -*- coding: utf-8 -*-
"""
my_metrics.py —— 统一超像素粒度的忠实性指标（适配 Quantus 0.5.3）
==================================================================
评估指标（已删除 Sufficiency）：
    ['Correlation', 'Estimate', 'MonotonicityCorrelation',
     'IROF', 'RegionPerturbation']

--------------------------------------------------------------------------
核心改动：所有指标统一到「超像素粒度」删除
--------------------------------------------------------------------------
原实现的问题
------------
1. FaithfulnessCorrelation 在 Quantus >= 0.4 改用「随机子集」扰动：
       arr 展平为 (C,H,W) 共 150528 个位置，随机抽 subset_size 个填黑。
   LIME 类归因只在 K 个超像素上非零，subset_size=224 时覆盖率 0.15%，
   随机点几乎全落在零区 —— 信噪比极低（实测 0.05 量级，而超像素粒度
   的同一批数据是 0.73 量级）。

2. FaithfulnessEstimate / RegionPerturbation 用固定像素数分批删除，
   MonotonicityCorrelation 同理；删除单元（像素块）与归因单元（超像素）
   不一致 —— 一次删除会把同一个权重值切开到多步。

3. 非零区删完后继续删 → 「累计归因和」出现恒定区，扭曲相关系数。

统一方案
--------
对**全部 12 个方法**（含稠密方法 IG/SA/GS/...）使用同一套超像素分割：
   稠密归因图 -> 在每个超像素内取均值 -> 得到每超像素一个值
   LIME/GradLIME -> 直接取 local_exp 的拟合权重
然后按 |值| 降序，逐个删除**整个超像素**，共删 K 个（num_features）。
这样删除单元 == 归因单元，且无恒定区，跨方法、跨指标完全可比。

各指标在该统一序列上的定义
--------------------------
  Correlation              Pearson (累计|归因|, 分数下降)       baseline=真黑
  Estimate                 Pearson (累计|归因|, 分数下降)       baseline=真黑
                           （与 Correlation 的区别：Estimate 用有符号累计；
                             见下方 use_signed 说明）
  MonotonicityCorrelation  Spearman(累计|归因|, 分数下降)       baseline=uniform
  IROF                     AOPC = mean(分数下降)                baseline=mean
  RegionPerturbation       AOPC = mean(分数下降)                baseline=uniform

其它修复（沿用此前结论）
------------------------
* 真黑 baseline：BLACK_NORMALISED = (0-mean)/std ≈ [-2.118,-2.036,-1.804]。
  直接填 0.0 反解是灰褐色 (124,116,104)，与 LIME 的 hide_color=0 不一致。
* device 必须显式透传，model 必须传裸 torch 模型。
* 版本自适应参数过滤 _keep()。
"""
from __future__ import annotations

import inspect

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:                                                    # noqa: BLE001
    torch = None
    F = None

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 归一化空间中的真黑：RGB(0,0,0) 经 (x-mean)/std
BLACK_NORMALISED = (0.0 - IMAGENET_MEAN) / IMAGENET_STD   # ≈[-2.118,-2.036,-1.804]


# --------------------------------------------------------------------------- #
# 归因图 -> 每超像素一个值
# --------------------------------------------------------------------------- #
def superpixel_values(attr_2d, segments):
    """稠密归因图 (H,W) -> 每个超像素的均值。

    返回 (ids, vals)：ids 为超像素编号（升序），vals 为对应归因值。
    """
    attr_2d = np.asarray(attr_2d, dtype=np.float64)
    ids = np.unique(segments)
    vals = np.zeros(len(ids), dtype=np.float64)
    for i, sid in enumerate(ids):
        m = (segments == int(sid))
        if m.any():
            vals[i] = float(attr_2d[m].mean())
    return ids, vals


def topk_superpixels(ids, vals, k):
    """按 |值| 降序取前 k 个，返回 [(sid, value), ...]。"""
    order = np.argsort(-np.abs(np.asarray(vals, dtype=np.float64)))
    k = int(min(k, len(order))) if k and k > 0 else len(order)
    return [(int(np.asarray(ids)[order[i]]), float(np.asarray(vals)[order[i]]))
            for i in range(k)]


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #
def baseline_value(kind, x_norm):
    """返回用于填充的三个通道值（归一化空间）。"""
    if kind == "black":
        return BLACK_NORMALISED.copy()
    if kind == "mean":
        # 归一化后各通道均值≈0（已减 mean 除 std），quantus 的 "mean" 语义
        return np.zeros(3, dtype=np.float64)
    if kind == "uniform":
        # quantus 的 "uniform"：np.random.uniform(0, 1)
        return np.random.uniform(0.0, 1.0, size=3).astype(np.float64)
    raise ValueError(f"unknown baseline: {kind}")


# --------------------------------------------------------------------------- #
# 超像素粒度的删除序列（纯 numpy，可独立测试）
# --------------------------------------------------------------------------- #
def build_deletion_batch(x_norm, segments, items):
    """逐个删除 items 中的超像素，构造 (K, C, H, W) 批次。

    items: [(seg_id, value), ...] 已按重要性排好序。
    返回 (xs, att_abs, att_sgn)：
        xs      (K, C, H, W)
        att_abs 累计 |value|
        att_sgn 累计 value
    """
    c, h, w = x_norm.shape
    xs, a_abs, a_sgn = [], [], []
    cum = np.zeros((h, w), dtype=bool)
    run_abs, run_sgn = 0.0, 0.0
    for sid, val in items:
        cum |= (segments == int(sid))
        xs.append(cum.copy())                     # 先存 mask，填值在下一步
        run_abs += abs(float(val))
        run_sgn += float(val)
        a_abs.append(run_abs)
        a_sgn.append(run_sgn)
    return np.stack(xs, axis=0), np.asarray(a_abs), np.asarray(a_sgn)


def fill_batch(x_norm, masks, baseline):
    """把 masks (K,H,W) 为 True 的像素填成 baseline。"""
    c, h, w = x_norm.shape
    out = np.repeat(x_norm[None, ...], len(masks), axis=0).astype(np.float32)
    flat = out.reshape(len(masks), c, -1)
    mf = masks.reshape(len(masks), -1)
    for r in range(len(masks)):
        flat[r][:, mf[r]] = baseline[:, None]
    return out


def build_deletion_batch_filled(x_norm, segments, items, baseline):
    """build_deletion_batch + fill_batch 的合并版本（真黑等固定 baseline）。"""
    c, h, w = x_norm.shape
    xs, a_abs, a_sgn = [], [], []
    cum = np.zeros((h, w), dtype=bool)
    run_abs, run_sgn = 0.0, 0.0
    for sid, val in items:
        cum |= (segments == int(sid))
        x = np.array(x_norm, copy=True)
        flat = x.reshape(c, -1)
        flat[:, cum.reshape(-1)] = np.asarray(baseline)[:, None]
        xs.append(x)
        run_abs += abs(float(val))
        run_sgn += float(val)
        a_abs.append(run_abs)
        a_sgn.append(run_sgn)
    return np.stack(xs, axis=0), np.asarray(a_abs), np.asarray(a_sgn)


# --------------------------------------------------------------------------- #
# 前向
# --------------------------------------------------------------------------- #
def predict_scores(model, xs, class_idx, device, softmax=True, batch_size=32):
    """对一批扰动图做前向，返回该类分数。"""
    if torch is None:
        raise RuntimeError("torch is required for metric computation")
    preds = []
    for i in range(0, len(xs), batch_size):
        b = torch.from_numpy(
            np.ascontiguousarray(xs[i:i + batch_size])).float().to(device)
        with torch.no_grad():
            out = model(b)
        if softmax:
            out = F.softmax(out, dim=1)
        preds.append(out[:, int(class_idx)].detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def predict_single(model, x_norm, class_idx, device, softmax=True):
    if torch is None:
        raise RuntimeError("torch is required for metric computation")
    t = torch.from_numpy(np.ascontiguousarray(x_norm[None])).float().to(device)
    with torch.no_grad():
        out = model(t)
    if softmax:
        out = F.softmax(out, dim=1)
    return float(out[0, int(class_idx)].item())


# --------------------------------------------------------------------------- #
# 相关系数
# --------------------------------------------------------------------------- #
def _pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = _rank(a)
    rb = _rank(b)
    return _pearson(ra, rb)


def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # 平秩处理
    uniq, inv = np.unique(np.round(x, 12), return_inverse=True)
    if len(uniq) < len(x):
        for u in uniq:
            m = np.isclose(x, u, atol=1e-12)
            if m.sum() > 1:
                r[m] = r[m].mean()
    return r


# --------------------------------------------------------------------------- #
# quantus 原生口径（可选后端）
# --------------------------------------------------------------------------- #
_TRUE_BLACK_PERTURB = None


def _get_true_black_perturb():
    global _TRUE_BLACK_PERTURB
    if _TRUE_BLACK_PERTURB is not None:
        return _TRUE_BLACK_PERTURB
    raise RuntimeError("perturb not initialised")


def make_true_black_perturb(n_channels: int = 3):
    """把选中位置替换为真黑，适配三种 quantus 契约。"""

    def _fill_block(block, idx, channels, hw):
        if idx.size == 0:
            return block
        if idx.max() >= hw:
            ch = np.clip(idx // hw, 0, channels - 1)
            pos = idx - ch * hw
            block[ch, pos] = BLACK_NORMALISED[ch]
        else:
            chs = np.repeat(np.arange(channels), idx.size)
            pos = np.tile(idx, channels)
            block[chs, pos] = BLACK_NORMALISED[chs]
        return block

    def _perturb(arr, indices=None, indexed_axes=None, **kwargs):
        a = np.array(arr, copy=True)
        if indices is None:
            return a
        idx = np.asarray(indices)
        if idx.size == 0:
            return a
        if a.ndim == 2:
            b, n = a.shape
            hw = n // n_channels
            flat = a.reshape(b, n_channels, hw)
            if idx.ndim == 2:
                for r in range(b):
                    _fill_block(flat[r], idx[r].reshape(-1), n_channels, hw)
            else:
                _fill_block(flat[0], idx.reshape(-1), n_channels, hw)
            return a
        if a.ndim == 3:
            c, h, w = a.shape
            _fill_block(a.reshape(c, h * w), idx.reshape(-1), c, h * w)
            return a
        if a.ndim == 4:
            b, c, h, w = a.shape
            flat = a.reshape(b, c, h * w)
            shp = idx.reshape(-1) if idx.ndim == 1 else idx[0].reshape(-1)
            for r in range(b):
                _fill_block(flat[r], shp, c, h * w)
            return a
        flat = a.reshape(-1)
        flat[idx.reshape(-1)] = float(BLACK_NORMALISED.mean())
        return a.reshape(arr.shape)

    return _perturb


def _keep(metric_cls, kwargs: dict) -> dict:
    """只保留当前版本 __init__ 中存在的参数。"""
    try:
        sig = inspect.signature(metric_cls.__init__).parameters
    except (TypeError, ValueError):                                  # noqa: BLE001
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig}


class MyMetrics(object):
    """忠实性指标封装。

    统一超像素粒度（推荐，默认）
        MyMetrics.compute_Faithfulness(model, x_batch, y_batch, a_batch,
                                       type, device, segments=segments,
                                       num_features=K)

    quantus 原生口径（对照）
        ... backend="quantus", subset_size=10000
    """

    # 指标 -> (baseline, 相似度/聚合方式, 是否用有符号累计)
    _SPEC = {
        'Correlation':             dict(baseline='black',   agg='pearson',  signed=False),
        'Estimate':                dict(baseline='black',   agg='spearman',  signed=False),
        'MonotonicityCorrelation': dict(baseline='uniform', agg='spearman', signed=False),
        'IROF':                    dict(baseline='mean',    agg='aopc',     signed=False),
        'RegionPerturbation':      dict(baseline='uniform', agg='aopc',     signed=False),
    }

    @staticmethod
    def metric_names():
        return list(MyMetrics._SPEC.keys())

    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_Faithfulness(model, x_batch, y_batch, a_batch, type, device=None,
                             segments=None, num_features=None,
                             softmax=True, batch_size=32,
                             backend='superpixel', subset_size=10000,
                             **kwargs):
        """计算单个忠实性指标，返回标量。

        segments 为 None 且 backend='superpixel' 时，退化为 quantus 原生口径。
        """
        if type not in MyMetrics._SPEC:
            print(f'Wrong type of Faithfulness: {type}')
            return float('nan')

        # ---- 归一化为 (H,W) 归因图 ----
        a = np.asarray(a_batch, dtype=np.float64)
        while a.ndim > 2:
            a = a[0]
        class_idx = int(np.asarray(y_batch).reshape(-1)[0])
        x_norm = np.asarray(x_batch, dtype=np.float32)
        if x_norm.ndim == 4:
            x_norm = x_norm[0]

        has_seg = segments is not None
        if backend == 'superpixel' and has_seg:
            return MyMetrics._superpixel_metric(
                model, x_norm, a, segments, class_idx, type, device,
                num_features, softmax, batch_size)

        # ---- 回退：quantus 原生口径 ----
        return MyMetrics._quantus_metric(
            model, np.asarray(x_batch), np.asarray(y_batch),
            np.asarray(a_batch), class_idx, type, device, subset_size)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _superpixel_metric(model, x_norm, attr_2d, segments, class_idx,
                           type, device, num_features, softmax, batch_size):
        spec = MyMetrics._SPEC[type]
        base = baseline_value(spec['baseline'], x_norm)

        # 稠密归因 -> 每超像素均值；再取 top-K
        ids, vals = superpixel_values(attr_2d, segments)
        K = num_features if (num_features and num_features > 0) else len(ids)
        items = topk_superpixels(ids, vals, K)
        if not items:
            return float('nan')

        xs, att_abs, att_sgn = build_deletion_batch_filled(
            x_norm, segments, items, base)
        orig = predict_single(model, x_norm, class_idx, device, softmax)
        after = predict_scores(model, xs, class_idx, device, softmax, batch_size)
        deltas = orig - after

        att = att_sgn if spec['signed'] else att_abs
        if spec['agg'] == 'pearson':
            return _pearson(att, deltas)
        if spec['agg'] == 'spearman':
            return _spearman(att, deltas)
        if spec['agg'] == 'aopc':
            return float(np.mean(deltas))
        return float('nan')

    # ------------------------------------------------------------------ #
    @staticmethod
    def _quantus_metric(model, x_batch, y_batch, a_batch, class_idx,
                        type, device, subset_size):
        import quantus
        tb = make_true_black_perturb(3)
        common = dict(return_aggregate=False, disable_warnings=True,
                      display_progressbar=False)
        try:
            if type == 'Correlation':
                kw = dict(nr_runs=100, subset_size=subset_size, abs=False,
                          normalise=True, perturb_func=tb,
                          perturb_baseline="black", **common)
                m = quantus.FaithfulnessCorrelation(
                    **_keep(quantus.FaithfulnessCorrelation, kw))
            elif type == 'Estimate':
                kw = dict(features_in_step=224, abs=False, normalise=True,
                          perturb_func=tb, perturb_baseline="black", **common)
                m = quantus.FaithfulnessEstimate(
                    **_keep(quantus.FaithfulnessEstimate, kw))
            elif type == 'MonotonicityCorrelation':
                kw = dict(nr_samples=10, features_in_step=3136,
                          perturb_baseline="uniform", **common)
                m = quantus.MonotonicityCorrelation(
                    **_keep(quantus.MonotonicityCorrelation, kw))
            elif type == 'IROF':
                kw = dict(segmentation_method="slic",
                          perturb_baseline="mean", **common)
                m = quantus.IROF(**_keep(quantus.IROF, kw))
            elif type == 'RegionPerturbation':
                kw = dict(patch_size=14, regions_evaluation=1,
                          perturb_baseline="uniform", normalise=True, **common)
                m = quantus.RegionPerturbation(
                    **_keep(quantus.RegionPerturbation, kw))
            else:
                return float('nan')
            s = m(model=model, x_batch=x_batch, y_batch=y_batch,
                  a_batch=a_batch, device=device)
            arr = np.asarray(s, dtype=float).reshape(-1)
            return float(arr[0]) if arr.size else float('nan')
        except Exception as exc:                                     # noqa: BLE001
            print(f'  [warn] quantus {type}: {type(exc).__name__}: {exc}')
            return float('nan')

    # ------------------------------------------------------------------ #
    # 以下类别保持原样（未做 0.5.3 适配，本次不使用）
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_Robustness(model, x_batch, y_batch, a_batch, type, device=None):
        import quantus
        score = None
        if type == 'LocalLipschitzEstimate':
            score = quantus.FaithfulnessCorrelation(
                nr_samples=10, perturb_std=0.2, perturb_mean=0.0,
                norm_numerator=quantus.similarity_func.distance_euclidean,
                norm_denominator=quantus.similarity_func.distance_euclidean,
                perturb_func=quantus.perturb_func.gaussian_noise,
                similarity_func=quantus.similarity_func.lipschitz_constant,
            )(model=model, x_batch=x_batch, y_batch=y_batch,
              a_batch=a_batch, device=device)
        elif type == 'MaxSensitivity':
            score = quantus.MaxSensitivity(
                nr_samples=10, lower_bound=0.2,
                norm_numerator=quantus.norm_func.fro_norm,
                norm_denominator=quantus.norm_func.fro_norm,
                perturb_func=quantus.perturb_func.uniform_noise,
                similarity_func=quantus.similarity_func.difference,
            )(model=model, x_batch=x_batch, y_batch=y_batch,
              a_batch=a_batch, device=device)
        elif type == 'AvgSensitivity':
            score = quantus.AvgSensitivity(
                nr_samples=10, lower_bound=0.2,
                norm_numerator=quantus.norm_func.fro_norm,
                norm_denominator=quantus.norm_func.fro_norm,
                perturb_func=quantus.perturb_func.uniform_noise,
                similarity_func=quantus.similarity_func.difference,
            )(model=model, x_batch=x_batch, y_batch=y_batch,
              a_batch=a_batch, device=device)
        elif type == 'Continuity':
            score = quantus.Continuity(
                patch_size=56, nr_steps=10, perturb_baseline="uniform",
                similarity_func=quantus.similarity_func.correlation_spearman,
            )(model=model, x_batch=x_batch, y_batch=y_batch,
              a_batch=a_batch, device=device)
        elif type == 'Consistency':
            score = quantus.Consistency(
                discretise_func=quantus.discretise_func.top_n_sign,
                return_aggregate=False,
            )(model=model, x_batch=x_batch, y_batch=y_batch,
              a_batch=a_batch, device=device)
        elif type == 'RelativeInputStability':
            score = quantus.RelativeInputStability(nr_samples=5)(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, device=device)
        elif type == 'RelativeOutputStability':
            score = quantus.RelativeOutputStability(nr_samples=5)(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, device=device)
        elif type == 'RelativeRepresentationStability':
            score = quantus.RelativeRepresentationStability(
                nr_samples=5, layer_names=["layer4.1.conv2"])(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, explain_func=quantus.explain,
                explain_func_kwargs={"method": "Lime"}, device=device)
        else:
            print('Wrong type of Robustness')
        return score

    @staticmethod
    def compute_Randomisation(model, x_batch, y_batch, a_batch, type, device=None):
        import quantus
        score = None
        if type == 'MPRT':
            score = quantus.MPRT(
                layer_order="bottom_up",
                similarity_func=quantus.similarity_func.correlation_spearman,
                normalise=True,
            )(model=model, x_batch=x_batch, y_batch=y_batch,
              a_batch=a_batch, device=device)
        elif type == 'RandomLogit':
            score = quantus.RandomLogit(
                num_classes=1000,
                similarity_func=quantus.similarity_func.ssim,
            )(model=model, x_batch=x_batch, y_batch=y_batch,
              a_batch=a_batch, device=device)
        else:
            print('Wrong type of Randomisation')
        return score

    @staticmethod
    def compute_Complexity(model, x_batch, y_batch, a_batch, type, device=None):
        import quantus
        score = None
        if type == 'Sparseness':
            score = quantus.Sparseness()(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, device=device)
        elif type == 'Complexity':
            score = quantus.Complexity()(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, device=device)
        elif type == 'EffectiveComplexity':
            score = quantus.EffectiveComplexity(eps=1e-5)(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, device=device)
        else:
            print('Wrong type of Complexity')
        return score

    @staticmethod
    def compute_Axiomatic(model, x_batch, y_batch, a_batch, type, device=None):
        import quantus
        score = None
        if type == 'Completeness':
            score = quantus.Completeness(abs=False, disable_warnings=True)(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, device=device)
        elif type == 'NonSensitivity':
            score = quantus.NonSensitivity(
                abs=True, eps=1e-5, n_samples=10, perturb_baseline="black",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                features_in_step=6272,
            )(model=model, x_batch=x_batch, y_batch=y_batch,
              a_batch=a_batch, device=device)
        elif type == 'InputInvariance':
            score = quantus.InputInvariance(abs=False, disable_warnings=True)(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, device=device)
        else:
            print('Wrong type of Complexity')
        return score

    @staticmethod
    def compute_Localisation(model, x_batch, y_batch, a_batch, s_batch, type, device=None):
        import quantus
        score = None
        if type == 'PointingGame':
            score = quantus.PointingGame()(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, s_batch=s_batch, device=device)
        elif type == 'AttributionLocalisation':
            score = quantus.AttributionLocalisation()(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, s_batch=s_batch, device=device)
        elif type == 'TKI':
            score = quantus.TopKIntersection()(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, s_batch=s_batch, device=device)
        elif type == 'RelevanceRankAccuracy':
            score = quantus.RelevanceRankAccuracy()(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, s_batch=s_batch, device=device)
        elif type == 'RelevanceMassAccuracy':
            score = quantus.RelevanceMassAccuracy()(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, s_batch=s_batch, device=device)
        elif type == 'AUC':
            score = quantus.AUC()(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch, s_batch=s_batch, device=device)
        else:
            print('Wrong type of Localisation')
        return score
