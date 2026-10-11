# -*- coding: utf-8 -*-
"""
A7 —— ReLU 与归一化消融（回应审稿意见 8：哪些组件真正起作用）
==============================================================
变体
    default        ReLU 开启 + 归一化开启（论文 Eq.7-8，对照组）
    no_relu        ReLU 关闭：保留负激活（a_m 可为负）
    no_normalise   归一化关闭：不做 a / max(a)
    abs_cam        用 |CAM| 缩放（保留负证据的幅度，但方向交给拟合系数）

为什么要做
    此前 400 张分析发现 GradLIME 有约 47% 的负系数（LIME 约 11%），
    而论文写的是"a_m ≥ 0 故符号保持"。本消融直接检验 ReLU 与归一化
    各自对结果的影响，判断它们是"真正起作用的组件"还是"惰性组件"。

已知预期（请在报告中如实呈现）
    * no_normalise 在数学上可能不改变结果：a/max(a) 是图内正单调缩放，
      单图内的 Top-K 排序与 Ridge 的相对结构不变。
      若实测两者几乎相同，应报告"归一化在此构型下是惰性的"，
      而不是硬说它重要。
    * no_relu 若改变了结果，说明负激活参与了拟合，
      论文中"符号保持"的表述需要相应修改。

用法
    python A7_relu_norm.py --num-classes 20 --num-images-per-class 10 \
        --require-correct --device cuda:0
"""
from __future__ import annotations

import numpy as np

import abl_common as C

VARIANTS = ['default', 'no_relu', 'no_normalise', 'abs_cam']
REFERENCE = 'default'


def cam_fn(ctx, variant):
    m = ctx['model']
    if variant == 'abs_cam':
        small, _ = C.compute_cam(m, ctx['x_tensor'], ctx['pred'],
                                 relu=False, normalise=False,
                                 device=ctx['device'])
        cam = np.abs(small)
        mx = float(cam.max())
        return C.cam_to_224(cam / mx if mx > 1e-12 else cam, 224)
    relu = (variant != 'no_relu')
    normalise = (variant != 'no_normalise')
    small, _ = C.compute_cam(m, ctx['x_tensor'], ctx['pred'], relu=relu,
                             normalise=normalise, device=ctx['device'])
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
    attr = C.attribution_from_local_exp(exp, ctx['pred'], a.num_features)
    # 记录负系数比例，用于检验"符号保持"
    items = exp.local_exp[ctx['pred']][:a.num_features]
    neg = sum(1 for _, w in items if w < 0)
    C.NEG_RATIO.setdefault(variant, []).append(neg / max(len(items), 1))
    return attr, exp.segments


def main():
    args = C.base_parser('A7: ReLU 与归一化消融').parse_args()
    C.NEG_RATIO = {}
    C.run_ablation(args, 'A7_relu_norm', VARIANTS, explain_fn,
                   reference=REFERENCE, cam_fn=cam_fn)
    if C.NEG_RATIO:
        print('\n[负系数比例]（用于检验"符号保持"）')
        for v in VARIANTS:
            if v in C.NEG_RATIO:
                arr = np.array(C.NEG_RATIO[v])
                print(f'  {v:<14} mean={arr.mean()*100:.1f}% '
                      f'std={arr.std(ddof=1)*100:.1f}%')


if __name__ == '__main__':
    main()
