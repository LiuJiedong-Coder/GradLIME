# -*- coding: utf-8 -*-
"""
A5 —— 掩码填充策略消融（评估侧协议敏感性）
==========================================
变体（填充被删除超像素的内容）
    black       归一化空间真黑 RGB(0,0,0)，与 LIME 生成解释时用的
                hide_color=0 一致（对照组）
    mean        归一化后各通道 0（等价于图像均值灰）
    uniform     [0,1] 均匀随机噪声
    blur        该图各通道均值（近似模糊填充）

为什么要做
    此前的实验表明填充策略对忠实性分数影响很大，甚至超过方法本身的差异。
    本消融在**同一批解释**上只改变填充内容，从而把"协议效应"与
    "方法效应"分离。这是审稿人常问的问题。

设计要点
    同一张图的四个变体共用完全相同的解释（attr 相同），
    只有删除时的填充不同 —— 因此任何差异都纯粹来自填充策略。

用法
    python A5_masking.py --num-classes 20 --num-images-per-class 10 \
        --require-correct --device cuda:0
"""
from __future__ import annotations

import numpy as np

import abl_common as C

VARIANTS = ['black', 'mean', 'uniform', 'blur']
REFERENCE = 'black'
_FILL = {'black': 'black', 'mean': 'mean', 'uniform': 'uniform',
         'blur': 'blur'}


def cam_fn(ctx, variant):
    small, _ = C.compute_cam(ctx['model'], ctx['x_tensor'], ctx['pred'],
                             relu=True, normalise=True, device=ctx['device'])
    return C.cam_to_224(small, 224)


def explain_fn(ctx, variant):
    """四个变体返回同一份解释（内部缓存）。"""
    if '_exp' not in ctx:
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
        ctx['_exp'] = exp
        ctx['_attr'] = C.attribution_from_local_exp(exp, ctx['pred'],
                                                    a.num_features)
    return ctx['_attr'], ctx['_exp'].segments


def fill_fn(variant):
    f = _FILL[variant]
    return {'corr': f, 'aopc_mean': f, 'aopc_uniform': f}


def main():
    p = C.base_parser('A5: 掩码填充消融')
    p.set_defaults(metrics='corr,aopc_mean,aopc_uniform')
    args = p.parse_args()
    C.run_ablation(args, 'A5_masking', VARIANTS, explain_fn,
                   reference=REFERENCE, cam_fn=cam_fn, fill_fn=fill_fn)


if __name__ == '__main__':
    main()
