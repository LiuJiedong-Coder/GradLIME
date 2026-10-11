# -*- coding: utf-8 -*-
"""
A4 —— 分割算法与粒度消融（协议敏感性）
======================================
变体
    quickshift          LIME 默认（对照组）
    slic_m50            SLIC，目标 50 个超像素
    slic_m100           SLIC，目标 100 个超像素
    slic_m200           SLIC，目标 200 个超像素
    felzenszwalb        Felzenszwalb 图分割

为什么要做
    超像素数 M 与解释长度 K 共同决定删除比例 K/M，而此前的分析表明
    该比例会显著影响忠实性指标（K=20 且 M≈33 时删除近六成图像，
    指标接近饱和）。本消融量化分割选择的影响。

注意
    分割算法通过 explain_instance 的 segmentation_fn 传入；
    若底层 lime_image_my 不支持该参数，相关变体会被跳过并给出提示。
    脚本会同时记录每张图的实际超像素数 M，便于核对 K/M。

用法
    python A4_segmentation.py --num-classes 20 --num-images-per-class 10 \
        --require-correct --device cuda:0
"""
from __future__ import annotations

import inspect

import numpy as np
from skimage.segmentation import felzenszwalb, slic

import abl_common as C

VARIANTS = ['quickshift', 'slic_m50', 'slic_m100', 'slic_m200', 'felzenszwalb']
REFERENCE = 'quickshift'


def _seg_fn(kind):
    def f(img):
        if kind == 'quickshift':
            from skimage.segmentation import quickshift
            return quickshift(img, kernel_size=4, max_dist=200, ratio=0.2)
        if kind == 'felzenszwalb':
            return felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
        n = int(kind.split('_m')[1])
        return slic(img, n_segments=n, compactness=10, sigma=1,
                    start_label=0)
    return f


def cam_fn(ctx, variant):
    small, _ = C.compute_cam(ctx['model'], ctx['x_tensor'], ctx['pred'],
                             relu=True, normalise=True, device=ctx['device'])
    return C.cam_to_224(small, 224)


def explain_fn(ctx, variant):
    from lime import lime_image_my
    a = ctx['args']
    ex = lime_image_my.LimeImageExplainer(
        random_state=C.fresh_rs(a.random_seed))
    kw = dict(top_labels=1, hide_color=0, num_samples=a.num_samples,
              num_features=a.num_features, random_seed=a.random_seed)
    if variant != 'quickshift':
        try:
            sig = inspect.signature(ex.explain_instance_data_label)
            if 'segmentation_fn' in sig.parameters:
                kw['segmentation_fn'] = _seg_fn(variant)
            else:
                C.NOTES.add(f'底层不支持 segmentation_fn，'
                            f'{variant} 与 quickshift 相同')
        except Exception:                                            # noqa: BLE001
            pass
    ctx['_last_explainer'] = ex   # 供 data 一致性校验
    ex.explain_instance_data_label(ctx['img_arr'], ctx['predict_fn'], **kw)
    exp = ex.explain_instance(ctx['cam'])
    C.M_VALUES.setdefault(variant, []).append(int(exp.segments.max()) + 1)
    return (C.attribution_from_local_exp(exp, ctx['pred'], a.num_features),
            exp.segments)


def main():
    args = C.base_parser('A4: 分割消融').parse_args()
    C.NOTES = set()
    C.M_VALUES = {}
    C.run_ablation(args, 'A4_segmentation', VARIANTS, explain_fn,
                   reference=REFERENCE, cam_fn=cam_fn)
    if C.M_VALUES:
        print('\n[超像素数 M]')
        for v in VARIANTS:
            if v in C.M_VALUES:
                arr = np.array(C.M_VALUES[v])
                print(f'  {v:<14} mean={arr.mean():.1f} '
                      f'min={arr.min()} max={arr.max()} '
                      f'K/M={args.num_features/arr.mean():.2f}')
    if C.NOTES:
        print('\n[notes]')
        for n in sorted(C.NOTES):
            print('  -', n)


if __name__ == '__main__':
    main()
