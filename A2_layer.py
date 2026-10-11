# -*- coding: utf-8 -*-
"""
A2 —— 目标激活层消融（回应审稿意见 5：单层 vs 多层的矛盾）
==========================================================
变体
    layer1 / layer2 / layer3 / layer4      ResNet-18 的四个 stage
    layer4                                 默认构型（对照组）
    multi                                  四个 stage 的 CAM 平均后归一化

为什么要做
    论文声称"深层特征更有效"，但方法描述的是"某一层"。
    审稿意见 5 指出这个矛盾。本消融直接给出逐层对比，
    并检验 multi-layer 是否优于任一单层。

注意
    不同 stage 的特征图分辨率不同（7x7 至 56x56）。本脚本先各自上采样到
    224x224 再聚合，保证同一套超像素上的激活向量可比。

用法
    python A2_layer.py --num-classes 10 --num-images-per-class 10 \
        --require-correct --device cuda:0
"""
from __future__ import annotations

import numpy as np

import abl_common as C

VARIANTS = ['layer1', 'layer2', 'layer3', 'layer4', 'multi']
REFERENCE = 'layer4'

_LAYER = {'layer1': 'layer1', 'layer2': 'layer2',
          'layer3': 'layer3', 'layer4': 'layer4'}


def cam_fn(ctx, variant):
    m = ctx['model']
    if variant == 'multi':
        acc = None
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            small, _ = C.compute_cam(m, ctx['x_tensor'], ctx['pred'],
                                     layer=getattr(m, name), relu=True,
                                     normalise=False, device=ctx['device'])
            up = C.cam_to_224(small, 224)
            acc = up if acc is None else acc + up
        acc /= 4.0
        mx = float(acc.max())
        return acc / mx if mx > 1e-12 else acc
    small, _ = C.compute_cam(m, ctx['x_tensor'], ctx['pred'],
                             layer=getattr(m, _LAYER[variant]), relu=True,
                             normalise=True, device=ctx['device'])
    return C.cam_to_224(small, 224)


def explain_fn(ctx, variant):
    from lime import lime_image_my
    a = ctx['args']
    ex = lime_image_my.LimeImageExplainer(
        random_state=C.fresh_rs(a.random_seed))
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
    args = C.base_parser('A2: 目标层消融').parse_args()
    C.run_ablation(args, 'A2_layer', VARIANTS, explain_fn,
                   reference=REFERENCE, cam_fn=cam_fn)


if __name__ == '__main__':
    main()
