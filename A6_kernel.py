# -*- coding: utf-8 -*-
"""
A6 —— 核宽度 σ 消融（回应审稿意见 4：距离度量与核函数）
======================================================
变体
    sigma=0.25 / 0.75 / 1.5 / 3.0
    LIME 库默认 kernel_width=0.25

为什么要做
    论文 Eq.(12) 的核函数写的是二值码空间上的余弦距离，但实现用的是
    LIME 原生核。审稿意见 4 追问核宽的作用。此前分析显示：在二值码空间
    上余弦距离约 0.29，exp(-d²/2σ²) 在多个 σ 下均接近 1，核近似为常数。
    本消融验证"σ 是否为惰性超参"。

注意
    kernel_width 通过 LimeImageExplainer(kernel_width=...) 传入；
    若底层 lime_image_my 不接受该参数，脚本会记录不可用并给出提示。

用法
    python A6_kernel.py --num-classes 20 --num-images-per-class 10 \
        --require-correct --device cuda:0
"""
from __future__ import annotations

import inspect

import numpy as np

import abl_common as C

VARIANTS = ['sigma0.25', 'sigma0.75', 'sigma1.5', 'sigma3.0']
REFERENCE = 'sigma0.25'
_SIG = {'sigma0.25': 0.25, 'sigma0.75': 0.75, 'sigma1.5': 1.5,
        'sigma3.0': 3.0}


def cam_fn(ctx, variant):
    small, _ = C.compute_cam(ctx['model'], ctx['x_tensor'], ctx['pred'],
                             relu=True, normalise=True, device=ctx['device'])
    return C.cam_to_224(small, 224)


def _explainer(kernel_width, seed):
    from lime import lime_image_my
    # 关键：每个变体用独立但同种子的 RandomState。
    # 若共用 check_random_state(None) 返回的 numpy 全局单例，
    # 各变体的邻域样本 data 会完全不同，对照失效。
    rs = C.fresh_rs(seed)      # 独立但同种子：保证变体间 data 一致
    try:
        sig = inspect.signature(lime_image_my.LimeImageExplainer.__init__)
        if 'kernel_width' in sig.parameters:
            return lime_image_my.LimeImageExplainer(
                kernel_width=kernel_width, random_state=rs)
        C.NOTES.add('LimeImageExplainer 不接受 kernel_width，'
                    'σ 变体与默认相同')
    except Exception:                                                # noqa: BLE001
        pass
    return lime_image_my.LimeImageExplainer(random_state=rs)


def explain_fn(ctx, variant):
    a = ctx['args']
    ex = _explainer(_SIG[variant], a.random_seed)
    ctx['_last_explainer'] = ex   # 供 data 一致性校验
    ex.explain_instance_data_label(ctx['img_arr'], ctx['predict_fn'],
                                   top_labels=1, hide_color=0,
                                   num_samples=a.num_samples,
                                   num_features=a.num_features,
                                   random_seed=a.random_seed)
    exp = ex.explain_instance(ctx['cam'])
    return (C.attribution_from_local_exp(exp, ctx['pred'], a.num_features),
            exp.segments)


def main():
    args = C.base_parser('A6: 核宽度消融').parse_args()
    C.NOTES = set()
    C.run_ablation(args, 'A6_kernel', VARIANTS, explain_fn,
                   reference=REFERENCE, cam_fn=cam_fn)
    if C.NOTES:
        print('\n[notes]')
        for n in sorted(C.NOTES):
            print('  -', n)


if __name__ == '__main__':
    main()
