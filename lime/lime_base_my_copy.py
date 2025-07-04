"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
import cv2
import torch
import torch.nn as nn


import numpy as np
from scipy.interpolate import griddata
def bilinear_interpolation(input_matrix, output_shape):
    # 创建网格坐标
    x = np.linspace(0, 1, input_matrix.shape[1])
    y = np.linspace(0, 1, input_matrix.shape[0])

    # 创建目标网格坐标
    x_new = np.linspace(0, 1, output_shape[1])
    y_new = np.linspace(0, 1, output_shape[0])

    # 生成网格点坐标
    x_grid, y_grid = np.meshgrid(x, y)
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

    # 将输入坐标和值转化为适合griddata函数的形式
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    values = input_matrix.ravel()

    # 使用双线性插值
    output_matrix = griddata(points, values, (x_new_grid, y_new_grid), method='linear')
    return output_matrix

def scale_cam_image(cam, target_size):
    cam = cam - np.min(cam)
    cam = cam / (1e-7 + np.max(cam))
    cam = cv2.resize(cam, target_size)
    result = np.float32(cam)
    return result


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
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
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':         #用岭回归训练模型，选择权重最大的K个特征
            clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_   # 特征权重
            if sp.sparse.issparse(data):   #检查data是否稀疏
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
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
                weighted_data = coef * data[0]   #源码
                #weighted_data = coef * data.numpy()[0]    #个人改,当data是tensor时
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)   #计算每个特征的权重，并按权重的绝对值从大到小进行排序
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
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
                                          num_features, n_method)
    # 生成解释器
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
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)         #通过邻域样本距离计算邻域样本权重
        labels_column = neighborhood_labels[:, label]      #labels_colum.shape(8000,)

        #print(image.shape)
        #### 个人方法，将CAM激活图通过线性插值法，得到与 org_image 的宽高相等的矩阵，然后对应上segement，取每个segement所对应cam激活块的最大值
        interpolation_cam = scale_cam_image(cam_map, target_size=(image.shape[1], image.shape[0]))
        #print("interpolation_cam: ", interpolation_cam.shape)
        cam_list = []
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            #cam_list.append(np.max(interpolation_cam[mask])) #取每一块中的最大值
            cam_list.append(np.mean(interpolation_cam[mask]))   #取每一块中的均值
        print("cam_list: ", cam_list)
        print(len(cam_list))
        cam_array = np.array(cam_list).reshape(1, -1)   # 转化为array
        repeat_cam_array = scale_cam_image(cam_array, target_size=(neighborhood_data.shape[1], neighborhood_data.shape[0]))   #插值
        # repeat_cam_array = cam_array.repeat(neighborhood_data.shape[0], 0)  #复制邻域样本数量行
        # print("复制行后的repeat_cam_array", repeat_cam_array.shape)
        neighborhood_data_old = neighborhood_data.astype(float)
        neighborhood_data = np.round(np.multiply(neighborhood_data_old, repeat_cam_array), 4)
        ################

        #一直使用,效果好
        #### 个人方法，将CAM激活图通过线性插值法，得到与 neighborhood_data 大小相等的矩阵，然后与之点乘
        # interpolation_cam = bilinear_interpolation(cam_map, neighborhood_data.shape)   #自写双线性插值法,效果更好一些
        # #interpolation_cam = scale_cam_image(cam_map, target_size=(neighborhood_data.shape[1], neighborhood_data.shape[0])) #CAM方法中双线性插值法的代码,与上句等效,效果略差
        # # print("interpolation_cam: ", interpolation_cam.shape)
        # neighborhood_data_old = neighborhood_data.astype(float)
        # neighborhood_data = np.round(np.multiply(neighborhood_data_old, interpolation_cam), 4)  #CAM激活图与neighborhood_data 点乘
        ##########################################

        ###个人方法，将CAM激活图通过线性插值法转化为 1*used_features 的一维向量，再将该一维张量复制成  邻域样本数*used_features 大小的矩阵，最后与neighborhood_data点乘
        # interpolation_cam = scale_cam_image(cam_map, target_size=(neighborhood_data.shape[1], 1))  # 将cam矩阵通过双线性插值方法拉成 1*num_features 大小的一维向量
        # #print("interpolation_cam: ", interpolation_cam.shape)
        # repeat_cam = interpolation_cam.repeat(neighborhood_data.shape[0], 0)
        # #print("复制行后的repeat_cam", repeat_cam.shape)
        # neighborhood_data_old = neighborhood_data.astype(float)
        # neighborhood_data = np.round(np.multiply(neighborhood_data_old, repeat_cam), 4)
        ##########

        #得到特征重要性排序列表
        used_features = self.feature_selection(neighborhood_data, labels_column, weights, num_features, feature_selection)
        print("used_features.shape: ", used_features.shape)

        ##第一版添加代码处
        ##

        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)

        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)
