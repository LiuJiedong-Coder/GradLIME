# -*- coding: utf-8 -*-
"""
A3 —— 邻域样本数 N 消融（效率论据）
==================================
变体
    N =  500 / 1000 / 4000 / 8000
    N = 1000 为默认构型（对照组）

为什么要做
    论文需要说明 N 的选取是否有依据，以及更少的采样是否足够。
    若 N 从 500 到 8000 指标几乎不变，则可采用更小的 N 以节省算力，
    这本身是一个可报告的效率结论。

用法
    python A3_nsamples.py --num-classes 20 --num-images-per-class 10 \
        --require-correct --device cuda:0

    注意：--num-samples 在此脚本中被变体覆盖，仅作为默认构型值。
"""
from __future__ import annotations

import numpy as np

import abl_common as C

VARIANTS = ['N500', 'N1000', 'N8000']
REFERENCE = 'N1000'
_NS = {'N500': 500, 'N1000': 1000, 'N8000': 8000}


def cam_fn(ctx, variant):
    small, _ = C.compute_cam(ctx['model'], ctx['x_tensor'], ctx['pred'],
                             relu=True, normalise=True, device=ctx['device'])
    return C.cam_to_224(small, 224)


def explain_fn(ctx, variant):
    from lime import lime_image_my
    a = ctx['args']
    ex = lime_image_my.LimeImageExplainer(
        random_state=C.fresh_rs(a.random_seed))
    ctx['_last_explainer'] = ex   # 供 data 一致性校验
    ex.explain_instance_data_label(ctx['img_arr'], ctx['predict_fn'],
                                   top_labels=1, hide_color=0,
                                   num_samples=_NS[variant],
                                   num_features=a.num_features,
                                   random_seed=a.random_seed)
    exp = ex.explain_instance(ctx['cam'])
    return (C.attribution_from_local_exp(exp, ctx['pred'], a.num_features),
            exp.segments)


def main():
    args = C.base_parser('A3: 邻域样本数消融').parse_args()
    print(f'[A3] 默认构型 N={args.num_samples}（应与 N1000 一致）')
    C.run_ablation(args, 'A3_nsamples', VARIANTS, explain_fn,
                   reference=REFERENCE, cam_fn=cam_fn)


if __name__ == '__main__':
    main()
