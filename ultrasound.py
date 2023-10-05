import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models import vit_b_16, ViT_B_16_Weights, vgg16_bn, VGG16_BN_Weights, resnet18, ResNet18_Weights
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torch import nn
from torch.nn import functional as F
from torch import optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt
# !pip install grad-cam
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from torchvision.transforms import Compose, Normalize, ToTensor

class VGG_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_vgg = vgg16_bn(weights=VGG16_BN_Weights)
        self.pretrained_vgg.classifier = nn.Sequential(*[self.pretrained_vgg.classifier[i] for i in range(4)])
        for params in self.pretrained_vgg.parameters():
            params.requires_grad = True

        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 10)
        self.fc6 = nn.Linear(10, 4)
        self.relu = nn.ReLU()
    
    def forward(self, img):
        features = self.pretrained_vgg(img)
        features = self.relu(features)
        out = self.fc1(features)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        return out


def load_classifier(device):
    classifier = VGG_classifier().to(device)
    checkpoint = torch.load('/Users/bishwakarki/Downloads/ultrasound/ultrasound_vgg_kfold.pth', map_location=device)
    classifier.load_state_dict(checkpoint)
    for param in classifier.parameters():
        param.requires_grad = True
    return classifier


def reshape_transform(tensor, height=224, width=224):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, 3)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def load_cam(classifier, device):
    target_layers = [classifier.pretrained_vgg.features[-1]]
    if device == 'cuda':
        cam = GradCAM(model=classifier, target_layers=target_layers, use_cuda=True)
    else:
        cam = GradCAM(model=classifier, target_layers=target_layers, use_cuda=False)
    return cam

def show_focused_image(grayscale_cam, img):
    grayscale_ = grayscale_cam[0, :]
    grayscale_[grayscale_ < 0.9] =0
    img_ = img/255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam[0, :]), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    ca = (1-0.8) * heatmap + 0.9 * img_
    ca = ca / np.max(ca)
    ca = np.uint8(255*ca)
    return ca


def preprocess_image(
    img, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img)


def load_img(path_):
    rgb_img = cv2.imread(path_, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    rgb_img = preprocess_image(rgb_img)
    rgb_img = torch.unsqueeze(rgb_img, 0)
    return rgb_img


def predict_(img_path, device):
    classifier = load_classifier(device)
    cam = load_cam(classifier, device)
    rgb_img = load_img(img_path)

    grayscale_cam = cam(input_tensor=rgb_img.to(device), targets=None, eigen_smooth=True)

    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (224, 224))
    predicted_img = show_focused_image(grayscale_cam, img)
    return img, predicted_img

def save_img(img_array, path):
    '''function to save image'''
    predicted_img = Image.fromarray(img_array.astype('uint8') )
    predicted_img.save(path)
    return True

def merge_img(img_1, img_2):
    '''merge two image into single'''
    pass


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = '/Users/bishwakarki/Downloads/ultrasound/Normal_3.png'
    original_img, predicted_img = predict_(img_path, device)
    predicted_img = Image.fromarray(predicted_img.astype('uint8') )
    predicted_img.save('predicted_2.png')
    # merged_img = merge_img(original_img, predicted_img)
    # save_img(merged_img)