import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam.base_cam import BaseCAM

from lime import lime_image
import torch.nn.functional as F
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./test_img/both.png',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',

                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layers, use_cuda, reshape_transform)

    def get_cam_weights(self, input_tensor, target_layer, target_category, activations, grads):
        return np.mean(grads, axis=(2, 3))

# 归一化图像，并转成tensor
def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


trans_A = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

trans_C = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
])

trans_B = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()





if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM
    }

    model = models.resnet18(pretrained=True)
    #print(model)

    target_layers = [model.layer4]
    #print(target_layers)

    rgb_img_org = cv2.imread(args.image_path, 1)[:, :, ::-1]   #是对颜色通道把BGR转换成RGB,CV库处理后为BGR格式，转为常规RGB格式
    rgb_img = np.float32(rgb_img_org) / 255   # 得到0-1之间的数
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #归一化，并转tensor, (1,3, 224, 224)
    #print(input_tensor)

    model_pred = model(input_tensor)
    pred_softmax = F.softmax(model_pred, dim=1)
    top_n = pred_softmax.topk(5)
    print(f'Model predicted labels:{top_n}')

    # img_pil = Image.open('examples/both.png')
    # input_tensor1 = trans_A(img_pil).unsqueeze(0)
    # pred_logits = model(input_tensor1)
    # pred_softmax = F.softmax(pred_logits, dim=1)
    # top_n = pred_softmax.topk(5)
    # print(f'Model predicted labels:{top_n}')

    targets = None
    cam_algorithm = methods[args.method]

    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
        #print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
        activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
        gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)  # 激活图(1,224,224)
        grayscale_cam = grayscale_cam[0, :]  #激活图，ndarray(224,224)



    #from lime import lime_image
    from lime import lime_image_my
    #explainer = lime_image.LimeImageExplainer()  #源码
    explainer = lime_image_my.LimeImageExplainer()  #个人修改

    #explanation = explainer.explain_instance(rgb_img_org, batch_predict, top_labels=5, hide_color=0, num_samples=8000)  # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
    data, labels = explainer.explain_instance_data_label(rgb_img_org, batch_predict, top_labels=1, hide_color=0, num_samples=2, batch_size=2)  # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
    # interpolation_cam = bilinear_interpolation(grayscale_cam, data.shape)
    # print(interpolation_cam.shape)

    explanation = explainer.explain_instance(grayscale_cam)
    print(f'解释器预测分类{explanation.top_labels}')






    from skimage.segmentation import mark_boundaries
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)
    plt.imshow(img_boundry)
    plt.show()
    #
    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[4], positive_only=False, num_features=20, hide_rest=False)
    # img_boundry = mark_boundaries(temp / 255.0, mask)
    # plt.imshow(img_boundry)
    # plt.show()