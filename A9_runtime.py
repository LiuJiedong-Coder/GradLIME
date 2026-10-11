# -*- coding: utf-8 -*-
"""
A9 —— 运行时间消融（GradLIME vs LIME）
=====================================
审稿意见要求：补充两种方法的**运行时间对比**，并说明额外开销的来源与量级。

本脚本在 val100 的 100 张图像上，对同一张图、同一 N、同一 random_seed
分别计时 LIME 与 GradLIME，做配对比较（配对 Wilcoxon + Cohen's d），
并把 GradLIME 的全部额外开销**分解到具体环节**。

计时分解
--------
两种方法共享的部分（理论上应当等量）：
    t_seg       quickshift 超像素分割
    t_data      生成 N 个掩码样本 + N 次黑盒前向（LIME 与 GradLIME 同构）

GradLIME 独有的部分：
    t_cam       Grad-CAM：一次前向 + 一次反向 + 通道加权 + resize 到 224
    t_interp    bilinear_interpolation（scipy griddata）把 CAM 插值到
                (N, M) 的邻域样本矩阵上。单独计时仅用于分解，
                它在真实流程中发生在 t_fit_gl 内部，不重复计入总时长。
    t_fit_gl    第二阶段 explain_instance：距离/核/插值/Ridge 选择/Ridge 拟合

因此：
    t_lime_total = t_seg + t_data + t_fit_lime
    t_gl_total   = t_data + t_cam + t_fit_gl      (t_interp ⊂ t_fit_gl)
    overhead     = t_gl_total - t_lime_total  ≈  t_cam + t_interp

脚本会做一条**自检**：比较实测 overhead 与 (t_cam + t_interp)，
若二者接近，说明"额外开销确实只来自 CAM 与插值，扰动阶段未被改动"。

关键结论（论文主张）
--------------------
t_cam 与 t_interp 都与 N **无关**（t_cam 只做一次前向+反向；
t_interp 的 Delaunay 三角化只依赖 CAM 尺寸与 M，不依赖 N）。
而 t_data ∝ N。故 N 越大，相对开销越小。
脚本用 --scale-n 在一部分图像上验证这一点。

计时方法
--------
* 使用 time.perf_counter()，CUDA 前后各调用 torch.cuda.synchronize()，
  避免异步 kernel 未完成时读数。
* 预热：先用 --warmup 张图跑完整流程并丢弃，消除 CUDA 上下文建立、
  cuDNN 自动选择算法、首次内存分配、Qhull/scipy 首次调用等一次性开销。
* 每张图重复 --repeat 次取**中位数**，抑制操作系统调度抖动。
* 两种方法用同一个 random_seed（C.fresh_rs），保证分割与邻域样本一致，
  计时比较的是同一份工作量。

用法
----
    python A9_runtime.py --base-dir F:\\Databases\\ImageNet2012\\val100 \\
        --num-images 100 --device cuda:0

    # 只跑 5 张看通路（含 N 缩放子实验默认 10 张，可 --scale-images 0 关闭）
    python A9_runtime.py --num-images 5 --scale-images 0

注意事项
--------
1. cam_size 的选择会显著改变 t_interp：
   '224' 时 griddata 要在 50176 个点上做 Delaunay 三角化，开销可能达到
   秒级；'raw'（7x7，即附件 GradLIME.py 的写法）时三角化成本可忽略。
   主实验（abl_common 的 cam_fn）用的是 cam_to_224，故默认取 '224'。
   两个口径都跑一遍，论文里如实报告所采用的那个。
2. t_interp 中"三角化"部分与 N 无关，"查询"部分随 (N, M) 缓慢增长；
   t_cam 与 N 完全无关。故 overhead 近似常数、overhead% 随 N 下降，
   这一点由 --scale-n 子实验直接验证，不要只靠论证。
3. lime_base_my.py 里已有一行注释掉的 cv2.resize 版本（scale_cam_image），
   用它替换 griddata 可大幅降低 t_interp，但数值略有差异；
   若改用需重跑全部实验，本脚本不改动任何原文件。

输出
----
    A9_runtime_per_image.csv    逐图各阶段耗时
    A9_runtime_scaling.csv      N 缩放子实验
    A9_runtime_summary.txt      汇总表（可直接抄进论文）
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import abl_common as C

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG', '.PNG'}


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser(description='A9 运行时间消融')
    p.add_argument('--base-dir', default=r'ImageNet2012/val100')
    p.add_argument('--imagenet-json', default='imagenet/imagenet_class_index.json')
    p.add_argument('--num-images', type=int, default=100)
    p.add_argument('--num-samples', type=int, default=1000,
                   help='LIME/GradLIME 邻域样本数 N（主实验口径）')
    p.add_argument('--num-features', type=int, default=20)
    p.add_argument('--random-seed', type=int, default=500)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out-dir', default='results_ablation')

    p.add_argument('--repeat', type=int, default=1,
                   help='每张图每种方法重复计时次数，取中位数')
    p.add_argument('--warmup', type=int, default=1,
                   help='预热图像数（计时结果丢弃）')
    p.add_argument('--require-correct', action='store_true',
                   help='仅保留 top-1 预测正确的图像（需要 imagenet json）')

    p.add_argument('--cam-size', choices=['224', 'raw'], default='224',
                   help="'224': 与消融脚本一致，把 CAM resize 到 224x224 后"
                        "再插值（griddata 在 50176 个点上做 Delaunay，较慢）；"
                        "'raw': 直接用 7x7 的 CAM（ GradLIME.py 的写法）")

    p.add_argument('--scale-images', type=int, default=10,
                   help='N 缩放子实验的图像数；0 = 不跑')
    p.add_argument('--scale-n', default='250,500,1000,2000',
                   help='N 缩放子实验的 N 取值，逗号分隔')
    p.add_argument('--log-every', type=int, default=10)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 图像收集：兼容 val100（平铺）与 val（按类别分子目录）两种结构
# --------------------------------------------------------------------------- #
def collect_images(base, limit):
    files = []
    for root, dirs, fs in os.walk(base):
        dirs.sort()
        for f in sorted(fs):
            if os.path.splitext(f)[1] in IMG_EXT:
                files.append(os.path.join(root, f))
    files.sort()
    return files[:limit] if limit and limit > 0 else files


def load_label_map(json_path):
    if not os.path.exists(json_path):
        return {}
    import json
    with open(json_path, 'r') as f:
        ci = json.load(f)
    return {ci[str(k)][0]: k for k in range(len(ci))}


# --------------------------------------------------------------------------- #
# 计时工具
# --------------------------------------------------------------------------- #
def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def timeit(fn, device, repeat=1, quiet=True):
    """返回 (median_seconds, last_output)。前后各同步一次 CUDA。

    quiet=True 时屏蔽被调用函数的 print。
    lime_base_my 每次都会 print("used_features.shape: ...")，
    而原生 lime 不打印；不屏蔽会让 GradLIME 额外承担打印开销，
    计时不公平。故默认对两种方法一视同仁地屏蔽。
    """
    ts, out = [], None
    for _ in range(max(1, repeat)):
        _sync(device)
        t0 = time.perf_counter()
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                out = fn()
        else:
            out = fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def _mean(v):
    return float(np.mean(v))


def _std(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


# --------------------------------------------------------------------------- #
# 单次测量：对一张图测 LIME 与 GradLIME 的各阶段耗时
# --------------------------------------------------------------------------- #
def measure_one(pil, model, predict_fn, device, a, N):
    """返回 dict，单位秒。失败返回 None。"""
    from lime import lime_image            # 原生 LIME
    from lime import lime_image_my         # 个人修改版（GradLIME 走这条）
    from lime.lime_base_my import bilinear_interpolation
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    img_arr = np.array(C.trans_C(pil))
    x_tensor = C.trans_A(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = int(F.softmax(model(x_tensor), dim=1).argmax(dim=1).item())

    res = {'img': os.path.basename(getattr(pil, 'filename', '')),
           'N': N, 'pred': pred}

    seed = a.random_seed

    # ---- 共享阶段 1：分割 ------------------------------------------------ #
    def _seg():
        return SegmentationAlgorithm('quickshift', kernel_size=4,
                                     max_dist=200, ratio=0.2,
                                     random_seed=seed)(img_arr)
    t_seg, segments = timeit(_seg, device, a.repeat)
    res['t_seg'] = t_seg
    res['M'] = int(np.unique(segments).shape[0])

    # ---- 共享阶段 2：N 个掩码样本 + N 次前向 ----------------------------- #
    # 用 GradLIME 的 data_label 接口测（LIME 内部是同构的 data_labels）
    ex_gl = lime_image_my.LimeImageExplainer(random_state=C.fresh_rs(seed))
    t_data, _ = timeit(
        lambda: ex_gl.explain_instance_data_label(
            img_arr, predict_fn, top_labels=1, hide_color=0,
            num_samples=N, num_features=a.num_features,
            random_seed=seed, progress_bar=False),
        device, a.repeat)
    res['t_data'] = t_data

    M = int(ex_gl.data.shape[1])
    res['M'] = M

    # ---- GradLIME 独有阶段 1：CAM ---------------------------------------- #
    def _cam():
        small, _ = C.compute_cam(model, x_tensor, pred, relu=True,
                                 normalise=True, device=device)
        if a.cam_size == '224':
            return C.cam_to_224(small, 224)
        return C._as_hw(small)
    t_cam, cam_map = timeit(_cam, device, a.repeat)
    res['t_cam'] = t_cam
    res['cam_shape'] = f'{np.asarray(cam_map).shape}'

    # ---- GradLIME 独有阶段 2：双线性插值 --------------------------------- #
    t_interp, _ = timeit(
        lambda: bilinear_interpolation(np.asarray(cam_map), (N, M)),
        device, a.repeat)
    res['t_interp'] = t_interp

    # ---- GradLIME 独有阶段 3：距离/核/选择/拟合 -------------------------- #
    t_fit_gl, exp_gl = timeit(lambda: ex_gl.explain_instance(cam_map),
                              device, a.repeat)
    res['t_fit_gl'] = t_fit_gl

    # ---- LIME：整体 ------------------------------------------------------- #
    ex_l = lime_image.LimeImageExplainer(random_state=C.fresh_rs(seed))
    t_lime, exp_l = timeit(
        lambda: ex_l.explain_instance(
            img_arr, predict_fn, top_labels=1, hide_color=0,
            num_samples=N, num_features=a.num_features,
            random_seed=seed, progress_bar=False),
        device, a.repeat)
    res['t_lime_total'] = t_lime

    # ---- 汇总 ------------------------------------------------------------- #
    # 注意：t_interp 已经包含在 t_fit_gl 内部
    # （explain_instance -> explain_instance_with_data 里调用了
    #  bilinear_interpolation），不能重复相加。
    res['t_gl_total'] = t_data + t_cam + t_fit_gl
    res['overhead'] = res['t_gl_total'] - t_lime
    res['overhead_pct'] = (100.0 * res['overhead'] / t_lime
                           if t_lime > 1e-9 else float('nan'))
    res['overhead_parts'] = t_cam + t_interp        # 预测值
    res['residual'] = res['overhead'] - res['overhead_parts']

    # ---- 公平性自检：两种方法是否在同一份分割 / 同一批邻域样本上工作 ----- #
    # 两者使用同一个 random_seed、同一个 quickshift 配置，分割应完全一致；
    # data 的行数都等于 N、列数都等于该图的超像素数 M。
    try:
        s_li = getattr(exp_l, 'segments', None)
        s_gl = getattr(exp_gl, 'segments', None)
        res['M_seg'] = int(np.unique(segments).shape[0])
        res['M_gl'] = int(np.asarray(ex_gl.data).shape[1])
        res['N_gl'] = int(np.asarray(ex_gl.data).shape[0])
        if s_li is None or s_gl is None:
            res['same_segments'] = 'n/a'
        else:
            res['same_segments'] = bool(
                np.array_equal(np.asarray(s_li), np.asarray(s_gl)))
    except Exception:                                                # noqa: BLE001
        res['same_segments'] = 'n/a'
    return res


def measure_scaling(pil, model, predict_fn, device, a, ns):
    """对一张图在多个 N 下测总耗时；返回 {N: (t_lime, t_gl)}。"""
    from lime import lime_image
    from lime import lime_image_my

    img_arr = np.array(C.trans_C(pil))
    x_tensor = C.trans_A(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = int(F.softmax(model(x_tensor), dim=1).argmax(dim=1).item())
    small, _ = C.compute_cam(model, x_tensor, pred, relu=True,
                             normalise=True, device=device)
    cam_map = (C.cam_to_224(small, 224) if a.cam_size == '224'
               else C._as_hw(small))

    out = {}
    for N in ns:
        ex_gl = lime_image_my.LimeImageExplainer(
            random_state=C.fresh_rs(a.random_seed))
        t_data, _ = timeit(
            lambda: ex_gl.explain_instance_data_label(
                img_arr, predict_fn, top_labels=1, hide_color=0,
                num_samples=N, num_features=a.num_features,
                random_seed=a.random_seed, progress_bar=False),
            device, 1)
        t_gl2, _ = timeit(lambda: ex_gl.explain_instance(cam_map), device, 1)
        ex_l = lime_image.LimeImageExplainer(
            random_state=C.fresh_rs(a.random_seed))
        t_l, _ = timeit(
            lambda: ex_l.explain_instance(
                img_arr, predict_fn, top_labels=1, hide_color=0,
                num_samples=N, num_features=a.num_features,
                random_seed=a.random_seed, progress_bar=False),
            device, 1)
        out[N] = (t_l, t_data + t_gl2)
    return out


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def paired(a, b):
    """b - a 的配对检验（与 abl_common.paired_test 口径一致）。"""
    return C.paired_test(a, b, min_n=5)


def fmt(v, nd=3):
    return 'nan' if not np.isfinite(v) else f'{v:.{nd}f}'


# --------------------------------------------------------------------------- #
def main():
    a = get_args()
    device = a.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = C.load_model(device)
    predict_fn = C.make_batch_predict(model, device)

    files = collect_images(a.base_dir, a.num_images)
    if not files:
        raise SystemExit(f'[error] 未找到图像: {a.base_dir}')
    print(f'[A9_runtime] device={device}  图像={len(files)}  N={a.num_samples} '
          f'K={a.num_features}  repeat={a.repeat}  cam_size={a.cam_size}')

    label_map = load_label_map(a.imagenet_json) if a.require_correct else {}
    if a.require_correct and not label_map:
        print('  [warn] 未找到 imagenet json，忽略 --require-correct')

    # ---------------- 预热 ---------------- #
    n_warm = min(a.warmup, len(files))
    if n_warm:
        print(f'[A9_runtime] 预热 {n_warm} 张（计时丢弃）...')
        for fp in files[:n_warm]:
            try:
                measure_one(Image.open(fp).convert('RGB'), model, predict_fn,
                            device, a, min(100, a.num_samples))
            except Exception as exc:                                  # noqa: BLE001
                print(f'  [warn] 预热失败 {fp}: {type(exc).__name__}: {exc}')

    # ---------------- 主计时 ---------------- #
    rows, t0 = [], time.time()
    n = 0
    for i, fp in enumerate(files):
        if n_warm and i < n_warm:
            continue
        try:
            pil = Image.open(fp).convert('RGB')
            pil.filename = fp
        except Exception:                                             # noqa: BLE001
            continue
        if a.require_correct and label_map:
            wnid = os.path.basename(os.path.dirname(fp))
            lab = label_map.get(wnid)
            if lab is None:
                continue
            x = C.trans_A(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                p = int(F.softmax(model(x), dim=1).argmax(dim=1).item())
            if p != lab:
                continue
        try:
            r = measure_one(pil, model, predict_fn, device, a, a.num_samples)
        except Exception as exc:                                      # noqa: BLE001
            print(f'  [warn] {os.path.basename(fp)} 失败: '
                  f'{type(exc).__name__}: {exc}')
            continue
        if r is None:
            continue
        r['img'] = os.path.basename(fp)
        rows.append(r)
        n += 1
        if n % a.log_every == 0 or n == 1:
            print(f'  [{n}/{len(files)-n_warm}] {r["img"]}  '
                  f'LIME={r["t_lime_total"]:.2f}s  '
                  f'GradLIME={r["t_gl_total"]:.2f}s  '
                  f'overhead={r["overhead"]:.3f}s')
    print(f'[A9_runtime] 完成 {n} 张，用时 {time.time()-t0:.1f}s')

    if not rows:
        raise SystemExit('[error] 没有成功计时的图像')

    # ---------------- 汇总 ---------------- #
    def col(k):
        return np.array([float(r[k]) for r in rows], dtype=float)

    os.makedirs(a.out_dir, exist_ok=True)
    p1 = os.path.join(a.out_dir, 'A9_runtime_per_image.csv')
    with open(p1, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = []
    lines.append('===== A9_runtime =====')
    lines.append(f'device={device}  n={n}  N={a.num_samples}  '
                 f'K={a.num_features}  repeat={a.repeat}  cam_size={a.cam_size}')
    lines.append('')
    lines.append('--- 各阶段耗时（秒；均值±标准差 / 中位数）---')
    lines.append(f'{"stage":<16}{"mean±std":>22}{"median":>12}')
    for k in ['t_seg', 't_data', 't_cam', 't_interp', 't_fit_gl',
              't_lime_total', 't_gl_total', 'overhead', 'overhead_parts',
              'residual']:
        v = col(k)
        lines.append(f'{k:<16}{_mean(v):>13.3f}±{_std(v):>7.3f}'
                     f'{np.median(v):>12.3f}')
    lines.append('')

    tl, tg = col('t_lime_total'), col('t_gl_total')
    st = paired(tl, tg)
    lines.append('--- 配对检验（GradLIME vs LIME，总耗时）---')
    lines.append(f'{"metric":<14}{"n":>5}{"diff":>12}{"ratio":>10}'
                 f'{"d":>10}{"p_holm":>12}')
    st['p_holm'] = C.holm_bonferroni([st['p_raw']])[0]
    lines.append(f'{"total_time":<14}{st["n"]:>5}'
                 f'{st["mean_diff"]:>12.3f}'
                 f'{_mean(tg)/max(1e-12,_mean(tl)):>10.3f}'
                 f'{st["cohen_d"]:>10.3f}{st["p_holm"]:>12.3g}')
    lines.append('')
    lines.append(f'平均相对开销 = {_mean(col("overhead_pct")):.2f}%  '
                 f'(中位 {np.median(col("overhead_pct")):.2f}%)')
    lines.append('')
    lines.append('--- 开销分解自检 ---')
    mo, mp, mr = (_mean(col('overhead')), _mean(col('overhead_parts')),
                  _mean(col('residual')))
    lines.append(f'实测 overhead    = {mo:.3f} s')
    lines.append(f't_cam+t_interp   = {mp:.3f} s')
    lines.append(f'残差 residual    = {mr:.3f} s '
                 f'(占 overhead {100*abs(mr)/max(1e-9, abs(mo)):.1f}%)')
    lines.append('残差小 => 额外开销确实只来自 CAM 与插值，扰动阶段未被改动。')
    lines.append('')
    same = [r.get('same_segments', 'n/a') for r in rows]
    n_true = sum(1 for s in same if s is True)
    n_na = sum(1 for s in same if s == 'n/a')
    n_bad = len(same) - n_true - n_na
    if n_na == 0 and n_bad == 0:
        lines.append('公平性自检：两种方法在完全相同的超像素分割上工作 ✓')
    elif n_bad == 0:
        lines.append(f'公平性自检：分割一致 {n_true}/{len(same)}，其余 {n_na} 张未能取到 segments 属性（记为 n/a）')
    else:
        lines.append(f'[warn] 分割不一致的图: {n_bad}/{len(same)}；请检查两种方法的 random_seed 与分割配置是否相同')
    try:
        m1 = np.array([float(r.get('M_seg', np.nan)) for r in rows])
        m2 = np.array([float(r.get('M_gl', np.nan)) for r in rows])
        ng = np.array([float(r.get('N_gl', np.nan)) for r in rows])
        lines.append(f'超像素数 M: 自身分割 {np.nanmean(m1):.1f} vs GradLIME data 列数 {np.nanmean(m2):.1f}；'                     f'data 行数 N: {np.nanmean(ng):.0f}')
    except Exception:                                              # noqa: BLE001
        pass
    lines.append('')

    # ---------------- N 缩放子实验 ---------------- #
    if a.scale_images > 0:
        ns = [int(x) for x in a.scale_n.split(',') if x.strip()]
        sub = files[n_warm:n_warm + a.scale_images]
        print(f'[A9_runtime] N 缩放子实验: {len(sub)} 张 × N={ns}')
        sc = []
        for fp in sub:
            try:
                pil = Image.open(fp).convert('RGB')
                r = measure_scaling(pil, model, predict_fn, device, a, ns)
            except Exception as exc:                                  # noqa: BLE001
                print(f'  [warn] scaling 失败 {fp}: {type(exc).__name__}: {exc}')
                continue
            row = {'img': os.path.basename(fp)}
            for N in ns:
                row[f't_lime_{N}'] = r[N][0]
                row[f't_gl_{N}'] = r[N][1]
            sc.append(row)
        if sc:
            p2 = os.path.join(a.out_dir, 'A9_runtime_scaling.csv')
            with open(p2, 'w', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=list(sc[0].keys()))
                w.writeheader()
                w.writerows(sc)
            lines.append('--- N 缩放子实验（秒）---')
            lines.append(f'{"N":>8}{"LIME":>12}{"GradLIME":>12}'
                         f'{"overhead":>12}{"overhead%":>12}')
            for N in ns:
                a1 = np.array([r[f't_lime_{N}'] for r in sc])
                b1 = np.array([r[f't_gl_{N}'] for r in sc])
                ov = b1 - a1
                lines.append(f'{N:>8}{_mean(a1):>12.3f}{_mean(b1):>12.3f}'
                             f'{_mean(ov):>12.3f}'
                             f'{100*_mean(ov)/max(1e-12,_mean(a1)):>11.2f}%')
            lines.append('')
            lines.append('overhead 应近似常数，overhead% 应随 N 增大而下降'
                         '（因为 t_data ∝ N，而 t_cam、t_interp 与 N 无关）。')

    txt = '\n'.join(lines)
    print()
    print(txt)
    with open(os.path.join(a.out_dir, 'A9_runtime_summary.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(f'\n[A9_runtime] 输出目录: {a.out_dir}')


if __name__ == '__main__':
    main()
