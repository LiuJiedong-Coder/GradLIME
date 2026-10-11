# -*- coding: utf-8 -*-
"""
lime_base_my_abl.py —— 支持三种 CAM 注入方式的 LimeBase（消融专用）
================================================================
背景
----
lime_base_my.py 是最终的 scaled_input 实现：

    interpolation_cam = bilinear_interpolation(cam_map, neighborhood_data.shape)
    neighborhood_data = np.round(neighborhood_data_old * interpolation_cam, 4)

它在**邻域矩阵** Z' 上逐元素乘激活，然后在这个缩放后的矩阵上做特征选择与拟合。
该文件不含 cam_mode 参数，因此无法在同一个解释器里切换构型。

本文件提供一个**可直接替换**的 explain_instance_with_data，支持：

    no_cam          不使用 CAM。等价于在原始二值矩阵 Z' 上拟合 -> 原生 LIME。
    scaled_input    完整复刻 lime_base_my.py 的行为（逐元素乘插值激活后拟合）。
                    返回的系数是 w（缩放码上的系数）。
    importance_only 在原始 Z' 上拟合（系数语义与原生 LIME 一致，记为 b），
                    CAM 只参与**特征选择的评分**：score_m = |b_m| * a_m，
                    其中 a_m 是超像素粒度的激活向量（长度 M）。
                    返回的系数是 b。

设计约束
--------
* 不修改 lime_base_my.py / lime_image_my.py / abl_common.py 的任何内容，
  只在**运行时**对 lime_base_my.LimeBase 的类方法打补丁。
* scaled_input 分支逐行复刻原实现，保证与主体实验数值可比。
* importance_only 需要超像素粒度的 a_m，这里用 image + segments 聚合得到，
  与 scaled_input 使用的 (N,M) 插值矩阵是两种不同的构造（原实现如此）。

用法
----
    import lime_base_my_abl

    ex = lime_image_my.LimeImageExplainer()
    lime_base_my_abl.set_mode(ex, 'importance_only')     # 或 no_cam / scaled_input
    ex.explain_instance_data_label(...)
    exp = ex.explain_instance(cam_224)

也可在运行前全局设定默认模式：

    lime_base_my_abl.install(default_mode='scaled_input')
"""
from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from sklearn.linear_model import Ridge, lars_path

CAM_MODES = ('no_cam', 'scaled_input', 'importance_only')
DEFAULT_MODE = 'scaled_input'

QUIET = True          # True 时不再打印 used_features.shape


def _accepts_kw(fn, name):
    """判断 fn 是否接受名为 name 的关键字参数。"""
    import inspect
    if fn is None:
        return False
    try:
        return name in inspect.signature(fn).parameters
    except Exception:                                            # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# 与原实现完全一致的工具函数
# --------------------------------------------------------------------------- #
def bilinear_interpolation(input_matrix, output_shape):
    """与 lime_base_my.py 中同名函数一致（griddata 线性插值）。"""
    x = np.linspace(0, 1, input_matrix.shape[1])
    y = np.linspace(0, 1, input_matrix.shape[0])

    x_new = np.linspace(0, 1, output_shape[1])
    y_new = np.linspace(0, 1, output_shape[0])

    x_grid, y_grid = np.meshgrid(x, y)
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    values = input_matrix.ravel()

    output_matrix = griddata(points, values, (x_new_grid, y_new_grid),
                             method='linear')
    return output_matrix


def scale_cam_image(cam, target_size):
    cam = cam - np.min(cam)
    cam = cam / (1e-7 + np.max(cam))
    cam = np.resize(cam, target_size)
    return np.float32(cam)


# --------------------------------------------------------------------------- #
# 超像素粒度的激活向量（importance_only 使用）
# --------------------------------------------------------------------------- #
def superpixel_activation(cam_map, segments, image=None, normalise=True,
                          min_clip=None):
    """CAM -> 每个超像素一个激活值 a_m（长度 M）。

    步骤：CAM 上采样到图像空间尺寸 -> 每个超像素内取均值 -> 除以最大值。

    返回 (a, info)：
        a     长度 M 的数组，a[i] 对应 data 的第 i 列
        info  诊断字典（是否连续编号、最小/最大值等）
    """
    cam_map = np.asarray(cam_map, dtype=np.float64)
    if cam_map.ndim > 2:
        cam_map = np.squeeze(cam_map)
        while cam_map.ndim > 2:
            cam_map = cam_map[0] if cam_map.shape[0] == 1 else cam_map.mean(0)

    if image is not None:
        target = tuple(np.asarray(image).shape[:2])
    else:
        target = tuple(np.asarray(segments).shape[:2])

    up = bilinear_interpolation(cam_map, target)
    up = np.nan_to_num(up, nan=0.0)

    segments = np.asarray(segments)
    ids = np.unique(segments)
    vals = np.zeros(len(ids), dtype=np.float64)
    for i, sid in enumerate(ids):
        m = (segments == int(sid))
        if m.any():
            vals[i] = float(up[m].mean())

    if min_clip is not None:
        vals = np.maximum(vals, float(min_clip))

    mx = float(vals.max()) if len(vals) else 0.0
    if normalise and mx > 1e-12:
        vals = vals / mx

    contiguous = bool(len(ids) > 0 and
                      np.array_equal(ids, np.arange(ids.max() + 1)))
    info = dict(M=len(ids), contiguous=contiguous,
                min=float(vals.min()) if len(vals) else 0.0,
                max=mx, n_neg=int((vals < 0).sum()),
                n_zero=int((vals == 0).sum()))
    return vals, info


# --------------------------------------------------------------------------- #
# 支持三种模式的 LimeBase
# --------------------------------------------------------------------------- #
class LimeBaseAbl(object):
    """与 lime_base_my.LimeBase 接口一致，额外支持 cam_mode。

    本类的方法以**非绑定**方式被调用（把原始 LimeBase 实例作为 self 传入），
    因此只依赖 self.kernel_fn / self.random_state / self.verbose 三个属性。
    """

    def __init__(self, kernel_fn, verbose=False, random_state=None,
                 cam_mode=DEFAULT_MODE):
        from sklearn.utils import check_random_state
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.cam_mode = cam_mode

    # ---------------- 与原实现一致 ---------------- #
    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector, weighted_labels,
                                     method='lasso', verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]], labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    # ---------------- 带 cam_activation 的特征选择 ---------------- #
    def feature_selection(self, data, labels, weights, num_features, method,
                          cam_activation=None):
        """cam_activation 为 None 时与原实现逐行一致。

        cam_activation 非 None 时（importance_only），评分改为
        |coef| * a_m，但**拟合仍在原始 data 上**，系数语义保持原生。
        """
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            if cam_activation is not None:
                # forward_selection 是逐步贪心，无法直接用外部评分；
                # 这里退化为"拟合一次后按 |coef| * a 排序"（alpha=0）
                return LimeBaseAbl._rank_by_coef(self, data, labels, weights,
                                                num_features, cam_activation,
                                                alpha=0.0)
            return LimeBaseAbl.forward_selection(self, data, labels, weights,
                                                 num_features)
        elif method == 'highest_weights':
            return LimeBaseAbl._rank_by_coef(self, data, labels, weights,
                                            num_features, cam_activation,
                                            alpha=0.01)
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = LimeBaseAbl.generate_lars_path(weighted_data, weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            return nonzero
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return LimeBaseAbl.feature_selection(self, data, labels, weights,
                                                num_features, n_method,
                                                cam_activation=cam_activation)
        else:
            raise ValueError(f'unknown feature_selection: {method}')

    def _rank_by_coef(self, data, labels, weights, num_features,
                      cam_activation, alpha):
        """拟合一次 Ridge，按 |coef| (* a_m) 降序取前 num_features。"""
        clf = Ridge(alpha=alpha, fit_intercept=True,
                    random_state=self.random_state)
        clf.fit(data, labels, sample_weight=weights)
        coef = clf.coef_
        if sp.sparse.issparse(data):
            coef = sp.sparse.csr_matrix(clf.coef_)
            weighted_data = coef.multiply(data[0])
            if cam_activation is not None:
                weighted_data = weighted_data.multiply(
                    np.asarray(cam_activation).reshape(1, -1))
            sdata = len(weighted_data.data)
            argsort_data = np.abs(weighted_data.data).argsort()
            if sdata < num_features:
                nnz_indexes = argsort_data[::-1]
                indices = weighted_data.indices[nnz_indexes]
                num_to_pad = num_features - sdata
                indices = np.concatenate(
                    (indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                indices_set = set(indices)
                pad_counter = 0
                for i in range(data.shape[1]):
                    if i not in indices_set:
                        indices[pad_counter + sdata] = i
                        pad_counter += 1
                        if pad_counter >= num_to_pad:
                            break
            else:
                nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                indices = weighted_data.indices[nnz_indexes]
            return indices
        else:
            weighted_data = coef * data[0]                      # 源码
            if cam_activation is not None:
                weighted_data = weighted_data * np.asarray(cam_activation)
            feature_weights = sorted(
                zip(range(data.shape[1]), weighted_data),
                key=lambda x: np.abs(x[1]), reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])

    # ---------------- 主入口 ---------------- #
    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   image,
                                   segments,
                                   cam_map,
                                   feature_selection='auto',
                                   model_regressor=None,
                                   cam_mode=None,
                                   a_min_clip=None):
        """与原实现签名一致，额外接受 cam_mode / a_min_clip。

        cam_mode:
            'no_cam'          不缩放，直接在 Z' 上拟合与选择
            'scaled_input'    复刻 lime_base_my.py（默认）
            'importance_only' 在 Z' 上拟合，CAM 只参与选择评分
        """
        mode = cam_mode or getattr(self, 'cam_mode', DEFAULT_MODE) \
            or DEFAULT_MODE
        if mode not in CAM_MODES:
            raise ValueError(f'cam_mode 必须是 {CAM_MODES} 之一，收到 {mode!r}')

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]

        data_raw = neighborhood_data.astype(float)
        M = data_raw.shape[1]
        a_vec = None
        a_info = None

        if mode == 'scaled_input':
            # ---- 逐行复刻 lime_base_my.py ---- #
            interpolation_cam = bilinear_interpolation(cam_map,
                                                       data_raw.shape)
            data_fit = np.round(np.multiply(data_raw, interpolation_cam), 4)
        elif mode == 'no_cam':
            # ---- 不使用 CAM：等价于原生 LIME ---- #
            data_fit = data_raw
        else:                                       # importance_only
            # ---- 在原始 Z' 上拟合，CAM 只作用于选择评分 ---- #
            data_fit = data_raw
            a_vec, a_info = superpixel_activation(
                cam_map, segments, image=image, normalise=True,
                min_clip=a_min_clip)
            if len(a_vec) != M:
                # 编号不连续等情况下降级为不注入，避免静默错位
                a_vec = None
                a_info = dict(error=f'a 的长度 {len(a_vec)} != M {M}')

        # 记录到实例，便于外部诊断
        try:
            self._abl_last_mode = mode
            self._abl_last_a = a_vec
            self._abl_last_a_info = a_info
        except Exception:                                        # noqa: BLE001
            pass

        # 选择用哪个 feature_selection：
        #   * importance_only 必须注入 cam_activation，而底层的
        #     LimeBase.feature_selection 不接受该参数（这就是上一版报
        #     "unexpected keyword argument 'cam_activation'" 的原因）；
        #     此时改用本文件的实现。
        #   * no_cam / scaled_input 优先用底层原方法，保证与主体实验保真；
        #     若底层方法不存在或也不接受该参数，同样退回本文件实现。
        force_own = getattr(type(self), '_abl_force_own_fs', False) or \
            getattr(self, '_abl_force_own_fs', False)
        native_fs = None if force_own else getattr(self, 'feature_selection',
                                                   None)
        if a_vec is None and native_fs is not None and \
                _accepts_kw(native_fs, 'cam_activation'):
            used_features = native_fs(data_fit, labels_column, weights,
                                      num_features, feature_selection,
                                      cam_activation=a_vec)
        elif a_vec is None and native_fs is not None:
            # 底层不接受 cam_activation，但也用不到：直接按原签名调用
            used_features = native_fs(data_fit, labels_column, weights,
                                      num_features, feature_selection)
        else:
            used_features = LimeBaseAbl.feature_selection(
                self, data_fit, labels_column, weights, num_features,
                feature_selection, cam_activation=a_vec)

        if not QUIET:
            print('used_features.shape: ', np.asarray(used_features).shape)

        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(data_fit[:, used_features], labels_column,
                       sample_weight=weights)

        prediction_score = easy_model.score(
            data_fit[:, used_features], labels_column, sample_weight=weights)
        local_pred = easy_model.predict(
            data_fit[0, used_features].reshape(1, -1))

        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)


# --------------------------------------------------------------------------- #
# 运行时注入（不修改任何原文件）
# --------------------------------------------------------------------------- #
_PATCHED = {'cls': None}


def find_base(explainer):
    """找到解释器里承载 explain_instance_with_data 的底层对象。"""
    for _k, v in getattr(explainer, '__dict__', {}).items():
        if hasattr(v, 'explain_instance_with_data'):
            return v
    for name in ('base', 'lime_base', '_base', '_lb', 'limebase'):
        v = getattr(explainer, name, None)
        if v is not None and hasattr(v, 'explain_instance_with_data'):
            return v
    return None


def install(default_mode=DEFAULT_MODE, module=None):
    """对 LimeBase 打补丁，使其支持 cam_mode。

    补丁后：
      * 各实例可携带 base._abl_cam_mode
      * 未设置时回退到类级默认 cls._abl_default_mode
    返回 (是否成功, 说明)。

    module 为 None 时按 lime.lime_base_my -> lime_base_my -> lime.lime_base
    的顺序查找，兼容不同的项目组织方式。
    """
    import importlib

    candidates = ([module] if module else
                  ['lime.lime_base_my', 'lime_base_my', 'lime.lime_base'])
    cls, used = None, None
    errs = []
    for name in candidates:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:                                 # noqa: BLE001
            errs.append(f'{name}: {exc}')
            continue
        c = getattr(mod, 'LimeBase', None)
        if c is not None:
            cls, used = c, name
            break
        errs.append(f'{name}: 没有 LimeBase')
    if cls is None:
        return False, '未能定位 LimeBase（' + '; '.join(errs) + '）'

    if _PATCHED['cls'] is cls:
        cls._abl_default_mode = default_mode
        return True, '已打补丁（复用）'

    orig = getattr(cls, 'explain_instance_with_data', None)
    if orig is None:
        return False, 'LimeBase 没有 explain_instance_with_data'

    def patched(self, neighborhood_data, neighborhood_labels, distances,
                label, num_features, image, segments, cam_map,
                feature_selection='auto', model_regressor=None,
                cam_mode=None, a_min_clip=None):
        if cam_mode is None:
            cam_mode = getattr(self, '_abl_cam_mode', None)
        if cam_mode is None:
            cam_mode = getattr(cls, '_abl_default_mode', DEFAULT_MODE)
        return LimeBaseAbl.explain_instance_with_data(
            self, neighborhood_data, neighborhood_labels, distances, label,
            num_features, image, segments, cam_map,
            feature_selection=feature_selection,
            model_regressor=model_regressor, cam_mode=cam_mode,
            a_min_clip=a_min_clip)

    try:
        cls.explain_instance_with_data = patched
    except Exception as exc:                                     # noqa: BLE001
        return False, f'打补丁失败: {exc}'

    cls._abl_default_mode = default_mode
    _PATCHED['cls'] = cls
    return True, f'已对 {module}.LimeBase 打补丁'


def set_mode(explainer, mode, module=None, force_own_fs=False):
    """为某个解释器设定 cam_mode。返回 (是否成功, 说明)。

    force_own_fs=True 时，no_cam / scaled_input 也使用本文件的
    feature_selection（逐行复刻标准实现），而不是底层原方法。
    当自检发现 no_cam 与原生 LIME 不一致时可开启此开关。
    """
    if mode not in CAM_MODES:
        return False, f'cam_mode 必须是 {CAM_MODES} 之一，收到 {mode!r}'
    ok, how = install(default_mode=mode, module=module)
    base = find_base(explainer)
    if base is not None:
        base._abl_cam_mode = mode
        if force_own_fs:
            type(base)._abl_force_own_fs = True
        return True, f'{how}；实例级 mode={mode}'
    # 找不到实例（可能懒创建）时依赖类级默认
    return ok, f'{how}；未找到 base 实例，依赖类级默认 mode={mode}'


def last_diag(explainer):
    """取最近一次解释的模式与 a_m 诊断（若底层已打补丁）。"""
    base = find_base(explainer)
    if base is None:
        return None
    return dict(mode=getattr(base, '_abl_last_mode', None),
                a=getattr(base, '_abl_last_a', None),
                info=getattr(base, '_abl_last_a_info', None))
