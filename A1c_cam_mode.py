# -*- coding: utf-8 -*-
"""
A1c —— CAM 注入方式消融
=======================
变体（三个，对应论文消融表的三行）
--------------------------------
    native_lime       原生 LIME。完全不经过 CAM，是"无 CAM"的基线。

    scaled_input      论文当前构型 / 对照组。
                      逐行复刻 lime_base_my.py：
                          interpolation_cam = bilinear_interpolation(
                              cam_map, neighborhood_data.shape)
                          data = round(data_old * interpolation_cam, 4)
                      在缩放码上选择与拟合；返回的系数 w ≈ b / a_m。

    importance_only   CAM 只参与**特征选择的评分**：
                          score_m = |b_m| * a_m
                      拟合仍在原始二值矩阵 Z' 上进行，
                      因此返回的系数 b 与原生 LIME 同义。

为什么用 lime_base_abl.py 而不是透传
------------------------------------
lime_base_my.py 的 explain_instance_with_data **没有 cam_mode 参数**——
它只有 scaled_input 一条路径。此前"把 cam_mode 透传进去"的做法（A1/A1b）
无法真正生效：底层不接受该参数，importance_only 会退化成 scaled_input 的
复制品（实测两变体权重完全相同）。

本脚本改用 lime_base_abl.py 在运行时替换底层实现，三种构型是三条真实的
不同代码路径。

关键：随机状态
--------------
每个变体用 C.fresh_rs(seed)（= 独立的 np.random.RandomState(seed)）。
若共用 sklearn check_random_state(None) 返回的 numpy 全局单例，
各变体会依次推进同一个状态，邻域样本 data 完全不同
（实测差异约 50% 元素），对照彻底失效。

用法
----
    python A1c_cam_mode.py --num-classes 10 --num-images-per-class 10 \\
        --require-correct --device cuda:0
"""
from __future__ import annotations

import abl_common as C
import lime.lime_base_my_abl as LA

VARIANTS = ['native_lime', 'scaled_input', 'importance_only']
REFERENCE = 'scaled_input'


def cam_fn(ctx, variant):
    small, _ = C.compute_cam(ctx['model'], ctx['x_tensor'], ctx['pred'],
                             relu=True, normalise=True, device=ctx['device'])
    return C.cam_to_224(small, 224)


def explain_fn(ctx, variant):
    a = ctx['args']
    K = a.num_features
    rs = C.fresh_rs(a.random_seed)      # 独立但同种子：保证变体间 data 一致

    if variant == 'native_lime':
        from lime import lime_image
        ex = lime_image.LimeImageExplainer(random_state=rs)
        ctx['_last_explainer'] = ex
        exp = ex.explain_instance(ctx['img_arr'], ctx['predict_fn'],
                                  top_labels=1, hide_color=0,
                                  num_samples=a.num_samples,
                                  num_features=K,
                                  random_seed=a.random_seed)
    else:
        from lime import lime_image_my
        ex = lime_image_my.LimeImageExplainer(random_state=rs)
        LA.set_mode(ex, variant)
        ctx['_last_explainer'] = ex
        ex.explain_instance_data_label(
            ctx['img_arr'], ctx['predict_fn'], top_labels=1, hide_color=0,
            num_samples=a.num_samples, num_features=K,
            random_seed=a.random_seed)
        exp = ex.explain_instance(ctx['cam'])

    return C.attribution_from_local_exp(exp, ctx['pred'], K), exp.segments


def main():
    p = C.base_parser('A1c: CAM 注入方式消融')
    p.add_argument('--a-min-clip', type=float, default=None,
                   help='importance_only 下 a_m 的下限截断。默认不截断。')
    args = p.parse_args()

    C.run_ablation(args, 'A1c_cam_mode', VARIANTS, explain_fn,
                   reference=REFERENCE, cam_fn=cam_fn)


if __name__ == '__main__':
    main()
