"""
Contains abstract functionality for learning locally linear sparse model.

==============================================================================
GradLIME -- lime_base_my.py  (更新版 v2)
==============================================================================
本文件是后续所有实验与论文改写的唯一实现基准 (single source of truth)。

【v1 已修复的历史问题】
  v1 之前的实现把 CAM 直接点乘到邻域矩阵上:
      interpolation_cam = bilinear_interpolation(cam_map, neighborhood_data.shape)
      neighborhood_data = neighborhood_data_old * interpolation_cam
  其上采样目标是 (N, M), 与论文 Eq.(8) 不符 (Eq.(8) 要求先把 CAM 插值到
  (H, W), 再按超像素取均值, 得到长度 M 的向量 a)。v1 已由
  segment_cam_activation 修正。

【v1 的核心设计 (v2 保留)】
  CAM 不进入回归输入 (不缩放特征), 只作用于「特征重要性」:
      1. 仍在原生二值邻域矩阵 Z' 上拟合岭回归, 系数语义保持 coef = b;
      2. 特征选择以 |coef| * a 评分, 使高梯度激活区域优先入选;
      3. 返回列表按 |coef| * a 降序, 系数保留原始符号 (绿/红)。
  原因: 若按论文 Eq.(9) 令 S = a (.) Z' 再拟合, 则 coef_j ~ b_j / a_j;
  当背景超像素 a_j -> 0 时系数爆炸, 主体特征被挤出 top-K。

【v2 的更新点】
  1. 新增 cam_mode 开关, 三种模式可在同一套代码下直接对比 (服务于消融 A.1):
       'feature'    : 【论文采用/推荐】CAM 缩放回归输入 S = a (.) Z' (Eq.9)。
                     300 张消融中忠实性最高 (FaithCorr 0.7222)。
       'importance': CAM 作用于重要性评分 s = |w|*a (Eq.14)，
                     拟合仍在二值码上；消融 A.1 对照。
       'feature'           : 论文 Eq.(9) 行为, S = a (.) Z' 进入回归
       'none'              : 忽略 CAM, 退化为原生 LIME (对照组)
  2. segment_cam_activation 默认改用 cv2.resize 上采样 (更快且无 NaN),
     OpenCV 缺失时自动回退到 griddata 双线性插值。
  3. 移除未使用的 torch / 重复 numpy 导入。
  4. 'auto' 分支在提供 CAM 时统一走 CAM 感知的 'highest_weights',
     避免 num_features <= 6 时静默退化成无 CAM 的 forward_selection。
  5. 特征选择与最终拟合的 Ridge alpha 显式暴露 (selection_alpha / fit_alpha)。
  6. 稀疏分支同样支持 CAM 加权。
==============================================================================
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

try:
    import cv2
    _HAS_CV2 = True
except ImportError:                                     # pragma: no cover
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
# CAM 上采样
# --------------------------------------------------------------------------- #
def bilinear_interpolation(input_matrix, output_shape):
    """基于 scipy.griddata 的双线性插值 (回退方案)。

    output_shape: (H, W)
    注意: griddata 在凸包外会产生 nan, 调用方需自行 nan_to_num。
    """
    input_matrix = np.asarray(input_matrix, dtype=float)
    x = np.linspace(0, 1, input_matrix.shape[1])
    y = np.linspace(0, 1, input_matrix.shape[0])
    x_new = np.linspace(0, 1, output_shape[1])
    y_new = np.linspace(0, 1, output_shape[0])
    x_grid, y_grid = np.meshgrid(x, y)
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    values = input_matrix.ravel()
    output_matrix = sp.interpolate.griddata(points, values, (x_new_grid, y_new_grid),
                             method='linear')
    return output_matrix


def scale_cam_image(cam, target_size):
    """将 CAM 上采样到原图尺寸并做 min-max 归一化。

    target_size: (H, W)
    """
    from scipy.interpolate import griddata as _griddata

    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - np.min(cam)
    cam = cam / (1e-7 + np.max(cam))

    if _HAS_CV2:
        # cv2.resize 的 dsize 是 (W, H)
        cam = cv2.resize(cam, (target_size[1], target_size[0]))
    else:                                               # pragma: no cover
        cam = _griddata(
            np.column_stack((np.linspace(0, 1, cam.shape[1]).repeat(cam.shape[0]),
                             np.tile(np.linspace(0, 1, cam.shape[0]), cam.shape[1]))),
            cam.ravel(),
            np.column_stack((np.meshgrid(np.linspace(0, 1, target_size[1]),
                                         np.linspace(0, 1, target_size[0]))[0].ravel(),
                             np.meshgrid(np.linspace(0, 1, target_size[1]),
                                         np.linspace(0, 1, target_size[0]))[1].ravel())),
            method='linear')
        cam = np.nan_to_num(
            cam,
            nan=float(np.nanmean(cam)) if np.any(~np.isnan(cam)) else 0.0,
            posinf=0.0, neginf=0.0)
    return np.float32(cam)


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""

    def __init__(self, kernel_fn, verbose=False, random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                          generate random numbers.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data."""
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector, weighted_labels,
                                     method='lasso', verbose=False)
        return alphas, coefs

    # ------------------------------------------------------------------ #
    def forward_selection(self, data, labels, weights, num_features,
                          selection_alpha=0.01):
        """Iteratively adds features to the model.

        注意: 该分支不使用 CAM (与原生 LIME 一致)。若需要 CAM 生效,
        请使用 cam_mode='feature' 触发的 'highest_weights' 分支。
        """
        clf = Ridge(alpha=selection_alpha, fit_intercept=True,
                    random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels, sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    # ------------------------------------------------------------------ #
    def feature_selection(self, data, labels, weights, num_features, method,
                          cam_activation=None, selection_alpha=0.01):
        """Selects features for the model.

        cam_activation: 可选, 形状 (M,) 的每个超像素 CAM 激活强度 (已归一化)。
                        仅用于 'highest_weights' 分支的评分加权,
                        使高梯度激活区域优先被选入解释。
        selection_alpha: 特征选择阶段 Ridge 的正则化强度。
        """
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features,
                                          selection_alpha=selection_alpha)
        elif method == 'highest_weights':
            # 用岭回归训练模型, 选择权重最大的 K 个特征
            clf = Ridge(alpha=selection_alpha, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # 个人修改: 用 |coef| * a 作为特征评分 (CAM 只作用在重要性上)
                if cam_activation is not None and np.any(cam_activation):
                    weighted_data = weighted_data.multiply(
                        sp.sparse.csr_matrix(cam_activation))
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
                weighted_data = coef * data[0]               # 源码
                # 个人修改: 用 |coef| * a 作为特征评分, 使 CAM 高激活区域优先入选
                if cam_activation is not None and np.any(cam_activation):
                    weighted_data = weighted_data * cam_activation
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method,
                                          cam_activation=cam_activation,
                                          selection_alpha=selection_alpha)
        else:
            raise ValueError('Unsupported feature_selection method: %s' % method)

    # ------------------------------------------------------------------ #
    @staticmethod
    def segment_cam_activation(cam_map, image, segments, n_features,
                               normalise=True):
        """将 (H, W) 的 CAM 激活图上采样到原图尺寸, 再按超像素取均值。

        Args:
            cam_map: (H, W) 的 CAM 激活图; 为 None 时退化为全 1 (等同原生 LIME)
            image:   原图 ndarray (H, W, 3), 仅用于取目标尺寸
            segments: (H, W) 超像素标签图
            n_features: 超像素个数 M
            normalise: 是否把 a 除以最大值映射到 [0, 1] (消融 A.1 用)

        Returns:
            a: (M,) 每个超像素的 CAM 激活强度
        """
        if cam_map is None:
            return np.ones(n_features, dtype=float)

        cam_map = np.asarray(cam_map, dtype=float)
        if cam_map.ndim != 2 or image is None or segments is None:
            return np.ones(n_features, dtype=float)

        target_size = (image.shape[0], image.shape[1])          # (H, W)
        try:
            interp_cam = scale_cam_image(cam_map, target_size)
        except Exception:                                       # 回退
            interp_cam = bilinear_interpolation(cam_map, target_size)
            interp_cam = np.nan_to_num(
                interp_cam,
                nan=float(np.nanmean(interp_cam)) if np.any(~np.isnan(interp_cam)) else 0.0,
                posinf=0.0, neginf=0.0)

        interp_cam = np.nan_to_num(interp_cam, nan=0.0, posinf=0.0, neginf=0.0)

        a = np.zeros(n_features, dtype=float)
        for sid in np.unique(segments):                          # 不假设 id 连续
            sid = int(sid)
            if sid < 0 or sid >= n_features:
                continue
            mask = (segments == sid)
            if mask.any():
                a[sid] = float(np.mean(interp_cam[mask]))

        amax = a.max()
        if amax <= 1e-12:
            # CAM 全为 0 (退化情况), 退化为不加权, 避免重要性被抹成 0
            return np.ones(n_features, dtype=float)
        if normalise:
            a = a / (amax + 1e-12)
        return a

    # ------------------------------------------------------------------ #
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
                                   cam_mode='feature',
                                   selection_alpha=0.01,
                                   fit_alpha=1.0,
                                   normalise_cam=True):
        """Takes perturbed data, labels and distances, returns explanation.

        cam_mode 决定 CAM 如何进入解释流程 (v2 新增):
            'importance' (默认): 在二值邻域矩阵 Z' 上拟合, 系数语义保持原生
                                 LIME 的 coef = b; CAM 只参与特征选择评分与
                                 最终排序, 即 |coef| * a。
            'feature'          : 论文 Eq.(9) 的行为, 令 S = a (.) Z', 在 S 上
                                 拟合。注意此时 coef_j ~ b_j / a_j, 背景
                                 超像素 a_j -> 0 会导致系数爆炸。
            'none'             : 完全忽略 cam_map, 行为等同原生 LIME。

        Args:
            neighborhood_data: perturbed data, 2d array (N, M)。
            neighborhood_labels: corresponding perturbed labels (N, L)。
            distances: distances to original data point。
            label: label for which we want an explanation。
            num_features: maximum number of features in explanation。
            feature_selection: 'forward_selection' / 'highest_weights' /
                               'lasso_path' / 'none' / 'auto'。
            model_regressor: sklearn regressor, 默认 Ridge(fit_alpha)。
            cam_map: (H, W) 特征梯度激活图 (Grad-CAM 输出)。
            cam_mode: 'importance' | 'feature' | 'none'。
            selection_alpha: 特征选择阶段 Ridge 的 alpha。
            fit_alpha: 最终拟合阶段 Ridge 的 alpha。
            normalise_cam: 是否把激活向量 a 归一化到 [0,1] (消融 A.1 用)。
                           为 False 时保留原始激活量级, 评分 |w_m| * a_m 会
                           受激活图整体幅值影响。

        Returns:
            (intercept, exp, score, local_pred)
            exp 为 (feature_id, weight) 列表, 按 |coef| * cam_activation 降序,
            weight 保留原始符号以区分正向(绿)/负向(红)作用。
        """
        weights = self.kernel_fn(distances)          # 邻域样本权重
        labels_column = neighborhood_labels[:, label]

        X = neighborhood_data.astype(float)          # 原生二值邻域矩阵 (N, M)
        n_features = X.shape[1]

        # 每个超像素的 CAM 激活强度 a, shape (M,)
        cam_act = self.segment_cam_activation(cam_map, image, segments,
                                              n_features,
                                              normalise=normalise_cam)
        if cam_mode == 'none':
            cam_act = np.ones(n_features, dtype=float)

        # ---- 构造回归输入 ---------------------------------------------
        if cam_mode == 'feature':
            # 论文 Eq.(9): S = a (.) Z'
            fit_data = X * cam_act[None, :]
            sel_cam = None            # 特征选择阶段不再二次加权
        else:
            fit_data = X
            sel_cam = cam_act if cam_mode == 'importance' else None

        # ---- 特征选择 --------------------------------------------------
        fs_method = feature_selection
        if cam_mode != 'none' and fs_method == 'auto':
            # 避免 num_features <= 6 时静默退化为无 CAM 的 forward_selection
            fs_method = 'highest_weights'

        used_features = self.feature_selection(
            fit_data, labels_column, weights, num_features, fs_method,
            cam_activation=sel_cam, selection_alpha=selection_alpha)
        used_features = np.asarray(used_features, dtype=int)

        # ---- 最终拟合 --------------------------------------------------
        if model_regressor is None:
            model_regressor = Ridge(alpha=fit_alpha, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(fit_data[:, used_features], labels_column,
                       sample_weight=weights)

        prediction_score = easy_model.score(
            fit_data[:, used_features], labels_column, sample_weight=weights)
        local_pred = easy_model.predict(
            fit_data[0, used_features].reshape(1, -1))

        # ---- 重要性排序 ------------------------------------------------
        coef = np.asarray(easy_model.coef_).ravel()
        a_used = cam_act[used_features]

        if cam_mode == 'feature':
            # 系数定义在缩放特征上: 贡献 = coef_j * a_j * z_j,
            # 故超像素重要性取 |coef_j * a_j| (数量级上回归到 |b_j|)。
            importance = np.abs(coef) * a_used
        elif cam_mode == 'importance':
            # CAM 作用在分子上: 重要性 = |coef| * a
            importance = np.abs(coef) * a_used
        else:
            importance = np.abs(coef)

        order = np.argsort(-importance)
        exp = [(int(used_features[i]), float(coef[i])) for i in order]

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred)
            print('Right:', neighborhood_labels[0, label])
            print('cam_mode:', cam_mode)
            print('CAM activation (per segment):', cam_act)
        return (easy_model.intercept_, exp, prediction_score, local_pred)
