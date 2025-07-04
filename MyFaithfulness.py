import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import argparse
import torch
import PIL
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import quantus
from pytorch_grad_cam import GradCAM
from my_metrics import MyMetrics
import time
import warnings
warnings.filterwarnings("ignore")




# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('imagenet/imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}


trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

trans_A = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    trans_norm
    ])

trans_B = transforms.Compose([
        transforms.ToTensor(),
        trans_norm
    ])

trans_C = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
])


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='gradcam',
                        help='CAM method')

    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component of cam_weights*activations')

    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    
    parser.add_argument('--NumFile', type=int, default=1,
                        help='Number of files to process')
    
    parser.add_argument('--FileSample', type=int, default=50,
                        help='The number of samples in each folder')
    args = parser.parse_args()
    return args

def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

methods = {
    "gradcam": GradCAM
}

def load_images_from_folder(folder_path):
    images = []
    paths = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            try:
                img = PIL.Image.open(img_path)
                images.append(img)
                paths.append(img_path)
            except:
                print("Unable to load image:", img_path)
    return paths, images

def computer_score(model, input, black_pred_id, attr_arry, metric_arry, device):
    exp_method = ['Lime', 'our', 'IG', 'SA', 'GS', 'IXG', 'DeCon', 'KH', 'LRP', 'Gradient', 'CVSF', 'CVRU']
    score_total = {}

    for metric in metric_arry:
        score_dist = {}
        i = 0
        for attr in attr_arry:
            score = MyMetrics.compute_Faithfulness(model, input, black_pred_id, attr, metric, device)
            score_dist[exp_method[i]] = score
            i += 1

        score_total[metric] = score_dist

    return score_total

def computer_avg_score(score_total):
    # 初始化均值字典
    average_dict = {}

    # 遍历每个字典的键
    for key in score_total[0]:
        # 遍历每个子键
        sub_keys = score_total[0][key].keys()

        # 初始化均值列表
        average_values = []

        # 遍历每个子键，计算均值
        for sub_key in sub_keys:
            values = [d[key][sub_key][0] for d in score_total]
            average_value = np.sum(values) / len(values)
            average_values.append(average_value)

        # 封装到均值字典
        average_dict[key] = dict(zip(sub_keys, average_values))

    return average_dict

def write_to_file(cor_black, sample_num, sample_score, class_name):
    with open('results/faithfulness-GradCAM.txt', 'a') as file:
        file.write(f'-------------File_Name: {class_name}  label: {cls2idx.get(class_name)}----------------\n')
        file.write(f'sample_score: \n')
        for i in range(len(sample_score)):
                file.write(f'{sample_score[i]}\n')

        file.write(f'sample_num: {sample_num}\n')
        file.write(f'cor_black: {(cor_black / sample_num)*100:.2f}\n')
        file.write(f'-------------------------------------------' + os.linesep)

def end_write_to_file(cor_black, sample_num, avg_score, time):
    # 打开文件以写入模式，如果文件不存在则会创建
    with open('results/faithfulness-GradCAM.txt', 'a') as file:
        file.write(f'--------------------Calculate End---------------------' + os.linesep)
        file.write(f'total_sample_num: {sample_num}\n')
        file.write(f'total_cor_black: {(cor_black / sample_num)*100:.2f}\n')
        file.write(f'avg_score: {avg_score}\n')
        file.write(f'total_time: {time:.2f}  sample_avg_explain_time: {time/sample_num:.2f}\n')
        file.write(f'-------------------------------------------' + os.linesep)


if __name__ == '__main__':
    args = get_args()
    #img_path = 'test_img/cat_dog.jpg'
    #img_path = 'examples/both.png'
    #img_path = 'test_img/bird2.png'

    base_dir = 'F:\data\ImageNet2012/val'
    classes = np.random.permutation(classes).tolist()   #随机打乱目录列表
    #classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]  #顺序

    #model = models.inception_v3(pretrained=True).eval().to(device)
    model = models.resnet18(pretrained=True).eval().to(device)
    n_files = 0
    total_score = []
    total_cor_pred = 0
    total_sample_num = 0
    total_cor_black = 0
    Time = time.time()
    for class_name in classes:
        label = cls2idx.get(class_name)
        print(f'-----------------------该file_name为: {class_name} 对应label为: {label}------------------------')
        class_dir = os.path.join(base_dir, class_name)
        paths, loaded_images = load_images_from_folder(class_dir)
        sample_num = 0
        cor_black = 0
        sample_score = []
        i = 0
        for img_path in range(len(loaded_images)):
            info = {}
            img_pil = loaded_images[i]
            input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
            pred_logits = model(input_tensor)
            pred_softmax = F.softmax(pred_logits, dim=1)
            top_n = pred_softmax.topk(1)
            #print(f'Predicted:{top_n}')
            pred_id = top_n[1].detach().cpu().numpy().squeeze().item()
            print(f'黑盒网络预测分类：{pred_id}')
            black_pred_id = np.expand_dims(np.array(pred_id), 0)  #进行该处理是为了满足评价准指标的计算输入维度， ndarray(1,)

            if pred_id == label:
                cor_black += 1

            targets = [ClassifierOutputTarget(pred_id)]
            cam_algorithm = methods[args.method]
            target_layers = [model.layer4]

            with cam_algorithm(model=model, target_layers=target_layers, use_cuda=True) as cam:
                # print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
                # activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
                # gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图

                grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)  # 激活图(1,7,7)
                grayscale_cam = grayscale_cam[0, :]  # cam激活图，ndarray(7,7)
                # grayscale_cam = grayscale_cam[0]  # cam激活图，ndarray(7,7)
                from torchcam.utils import overlay_mask
                fig_cam = overlay_mask(img_pil, Image.fromarray(grayscale_cam), alpha=0.4)  # alpha越小，原图越淡

            random_seed = 500
            num_features = 20
            from skimage.segmentation import mark_boundaries
            ####################原始LIME######################################

            from lime import lime_image
            explainer_org = lime_image.LimeImageExplainer()
            explanation_org = explainer_org.explain_instance(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)

            temp, mask = explanation_org.get_image_and_mask(explanation_org.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
            img_boundry_org = mark_boundaries(temp/255.0, mask)  #ndarray(224, 224, 3), lime结果图，可以通过plt显示

            attributions_org = img_boundry_org.sum(axis=2)  # 计算每个像素的通道值之和，作为归因贡献图 ndarray(224, 224)
            attr_img_org = attributions_org.reshape(1, 1, 224, 224)   #满足评价准指标的计算输入维度
            mask_org = mask.reshape(1, 1, 224, 224)

            ####################OUrsLIME#####################################
            from lime import lime_image_my

            explainer = lime_image_my.LimeImageExplainer()  # 个人修改
            # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
            data, labels = explainer.explain_instance_data_label(np.array(trans_C(img_pil)), batch_predict, top_labels=1,hide_color=0, num_samples=8000, random_seed=random_seed)
            explanation = explainer.explain_instance(grayscale_cam)
            print(f'解释器预测分类{explanation.top_labels}')

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=num_features, hide_rest=False)
            img_boundry_our = mark_boundaries(temp / 255.0, mask)

            attributions_our = img_boundry_our.sum(axis=2)  # 计算每个像素的通道值之和，作为归因贡献图 ndarray(224, 224)
            attr_img_our = attributions_our.reshape(1, 1, 224, 224)   #满足评价准指标的计算输入维度
            mask_our = mask.reshape(1, 1, 224, 224)

            # ['GradientShap', 'IntegratedGradients', 'DeepLift', 'DeepLiftShap', 'InputXGradient', 'Saliency', 'FeatureAblation',
            #  'Deconvolution', 'FeaturePermutation', 'Lime', 'KernelShap', 'LRP', 'Gradient', 'Occlusion', 'LayerGradCam',
            #  'GuidedGradCam', 'LayerConductance', 'LayerActivation', 'InternalInfluence', 'LayerGradientXActivation',
            #  'Control Var. Sobel Filter', 'Control Var. Constant', 'Control Var. Random Uniform']
            # ['VanillaGradients', 'IntegratedGradients', 'GradientsInput', 'OcclusionSensitivity', 'GradCAM', 'SmoothGrad']

            attr_IG = quantus.explain(model, input_tensor, black_pred_id, method="IntegratedGradients")  #2017
            attr_SA = quantus.explain(model, input_tensor, black_pred_id, method="Saliency")    #2014
            attr_GS = quantus.explain(model, input_tensor, black_pred_id, method="GradientShap")  #2017
            attr_IXG = quantus.explain(model, input_tensor, black_pred_id, method="InputXGradient")  #2016
            attr_DeCon = quantus.explain(model, input_tensor, black_pred_id, method="Deconvolution")   #2014
            attr_KH = quantus.explain(model, input_tensor, black_pred_id, method="KernelShap")  #2017
            attr_LRP = quantus.explain(model, input_tensor, black_pred_id, method="LRP")   #2017
            attr_G = quantus.explain(model, input_tensor, black_pred_id, method="Gradient")
            attr_CVSF = quantus.explain(model, input_tensor, black_pred_id, method="Control Var. Sobel Filter")
            attr_CVRU = quantus.explain(model, input_tensor, black_pred_id, method="Control Var. Random Uniform")

            metrics = ['Correlation', 'Estimate', 'MonotonicityCorrelation', 'IROF', 'Sufficiency', 'RegionPerturbation']
            attrs = [attr_img_org, attr_img_our, attr_IG, attr_SA, attr_GS, attr_IXG, attr_DeCon, attr_KH, attr_LRP, attr_G, attr_CVSF, attr_CVRU]

            #print(f'#####Faithfulness Metrics#####')
            score = computer_score(model, input_tensor.cpu().numpy(), black_pred_id, attrs, metrics, device)

            info[paths[i]] = label
            info['black_pred_label'] = pred_id
            info['scores'] = score
            sample_score.append(info)
            total_score.append(score)
            print(f'Faithfulness Score: {score}')
            print(f'sample_id: {sample_num}')

            print(f'---------------------------------')
            sample_num += 1
            i += 1
            if i == args.FileSample:
                break

        write_to_file(cor_black, sample_num, sample_score, class_name)
        total_sample_num += sample_num
        total_cor_black += cor_black
        n_files += 1
        if n_files == args.NumFile:
            break


    endTime = time.time() - Time
    avg_score = computer_avg_score(total_score)
    end_write_to_file(total_cor_black, total_sample_num, avg_score, endTime)
    print(f'Time: {endTime}')


