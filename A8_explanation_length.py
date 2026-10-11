# -*- coding: utf-8 -*-
"""
A8 —— 解释长度 K / 删除预算消融（K 依赖性的正面回应）
======================================================
变体
    K = 5 / 10 / 20 / 30 / 40
    K = 20 为默认构型（对照组）

为什么要做
    此前的多批实验显示：K=20 时 GradLIME 在相关性类指标上略优，
    K=10 时变为不显著甚至略劣。也就是说结论依赖删除预算。
    与其隐藏这一点，不如把它做成正式消融并如实报告 ——
    这正好 preempt 审稿人"为什么选这个 K"的追问。

设计要点
    K 同时影响：(a) LIME/GradLIME 保留的超像素数（解释长度）；
               (b) 删除序列的步数（评估预算）。
    两者同步变化，与论文中 K 的语义一致。
    同时记录删除比例 K/M，便于说明协议饱和度。

建议同时报告随机对照基线
    可用 A4/A8 的相同协议，把归因替换为随机排序得到基线值；
    K 较小时基线应明显下降（协议不再饱和）。

用法
    python A8_explanation_length.py --num-classes 20 \
        --num-images-per-class 10 --require-correct --device cuda:0
"""
from __future__ import annotations

import numpy as np

import abl_common as C

VARIANTS = ['K5', 'K10', 'K20', 'K30', 'K40']
REFERENCE = 'K20'
_KS = {'K5': 5, 'K10': 10, 'K20': 20, 'K30': 30, 'K40': 40}


def cam_fn(ctx, variant):
    small, _ = C.compute_cam(ctx['model'], ctx['x_tensor'], ctx['pred'],
                             relu=True, normalise=True, device=ctx['device'])
    return C.cam_to_224(small, 224)


def explain_fn(ctx, variant):
    from lime import lime_image_my
    a = ctx['args']
    K = _KS[variant]
    ex = lime_image_my.LimeImageExplainer(
        random_state=C.fresh_rs(a.random_seed))
    ctx['_last_explainer'] = ex   # 供 data 一致性校验
    ex.explain_instance_data_label(ctx['img_arr'], ctx['predict_fn'],
                                   top_labels=1, hide_color=0,
                                   num_samples=a.num_samples,
                                   num_features=K,
                                   random_seed=a.random_seed)
    exp = ex.explain_instance(ctx['cam'])
    M = int(exp.segments.max()) + 1
    C.KM.setdefault(variant, []).append(K / max(M, 1))
    return C.attribution_from_local_exp(exp, ctx['pred'], K), exp.segments


def main():
    args = C.base_parser('A8: 解释长度消融').parse_args()
    C.KM = {}
    C.run_ablation(args, 'A8_explanation_length', VARIANTS, explain_fn,
                   reference=REFERENCE, cam_fn=cam_fn,
                   k_fn=lambda v: _KS[v])
    if C.KM:
        print('\n[删除比例 K/M]（比例越大，协议越接近饱和）')
        for v in VARIANTS:
            if v in C.KM:
                arr = np.array(C.KM[v])
                print(f'  {v:<6} mean={arr.mean():.2f} '
                      f'min={arr.min():.2f} max={arr.max():.2f}')


if __name__ == '__main__':
    main()
