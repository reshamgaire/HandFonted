import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import os

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # Max pooling -> 1x1 convolution branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResInceptionNet(nn.Module):
    def __init__(self, num_classes=52, dropout_rate=0.4):
        super(ResInceptionNet, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # First ResNet block (with downsample)
        self.res_block1 = self._make_res_layer(32, 64, stride=2)

        # First Inception block
        self.inception1 = InceptionBlock(64, 16, 24, 32, 4, 8, 8)  # Output: 64 channels

        # Second ResNet block (with downsample)
        # Inception1 outputs 16+32+8+8 = 64 channels
        self.res_block2 = self._make_res_layer(64, 128, stride=2)

        # Second Inception block
        self.inception2 = InceptionBlock(128, 32, 48, 64, 8, 16, 16)  # Output: 128 channels

        # Third ResNet block (with downsample)
        # Inception2 outputs 32+64+16+16 = 128 channels
        self.res_block3 = self._make_res_layer(128, 256, stride=2)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def _make_res_layer(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet block 1
        x = self.res_block1(x)

        # Inception block 1
        x = self.inception1(x)

        # ResNet block 2
        x = self.res_block2(x)

        # Inception block 2
        x = self.inception2(x)

        # ResNet block 3
        x = self.res_block3(x)

        # Global average pooling
        x = self.gap(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)

        return x

def pad_image_to_square(image, extra_padding_amount=0):
    """
    Pads an image based on custom rules:
    1. If the image is square, add 'extra_padding_amount' uniformly to all sides.
    2. If not square, add 'extra_padding_amount' uniformly to the longer dimension,
       then pad the shorter dimension to make the image square.

    Args:
        image (np.ndarray): The input image.
        extra_padding_amount (int): The amount of extra uniform padding to apply.
                                   Should be non-negative.

    Returns:
        np.ndarray: The padded square image.
    """

    h, w = image.shape[:2]

    target_side_length = 0

    if h == w:
        target_side_length = h + 2 * extra_padding_amount
    else:
        if h > w:
            target_side_length = h + 2 * extra_padding_amount
        else:
            target_side_length = w + 2 * extra_padding_amount

    delta_h = target_side_length - h
    top = delta_h // 2
    bottom = delta_h - top

    delta_w = target_side_length - w
    left = delta_w // 2
    right = delta_w - left

    padded_square_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=0)

    return padded_square_image

def assign_classes(probabilities):
    num_samples = probabilities.shape[0]
    num_classes = probabilities.shape[1]

    cost_matrix = -probabilities

    if num_samples > num_classes:
        row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
        assigned_class_indices = row_ind
        assigned_sample_indices = col_ind
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_sample_indices = row_ind
        assigned_class_indices = col_ind

    assigned_probabilities = probabilities[assigned_sample_indices, assigned_class_indices]

    return assigned_sample_indices, assigned_probabilities, assigned_class_indices

def classify_characters(list_of_char_images, model_path="", OUTPUT_PATH=""):
    """
    Predict the class of each character image using a pre-trained model.
    Args:
        list_of_char_images (list): List of character images to predict.
        OUTPUT_PATH(str): Path to save the predicted images.
    Returns:
        list: A list of tuples, where each tuple is (character_string, numpy.ndarray)    
    """
    orignal_images = []
    image_tensors = []
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,))
    ])
    for img_arr in list_of_char_images:

        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
        )
        image = pad_image_to_square(thresh,extra_padding_amount=1)
        image_tensor = transform(Image.fromarray(image))
        image_tensors.append(image_tensor)

        orignal_images.append(thresh)

    classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    if image_tensors:
        batch_tensor = torch.stack(image_tensors)

        model = ResInceptionNet(num_classes=52)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        model.eval()
        with torch.inference_mode():
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        orignal_indices, confidence, predicted_classes = assign_classes(probabilities)

        images = [orignal_images[i] for i in orignal_indices.tolist()]
        predicted_labels = [classes[i] for i in predicted_classes.tolist()]

        if OUTPUT_PATH:
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            for img_index, confidence, prediction in zip(orignal_indices, confidence, predicted_classes):
                save_path = os.path.join(OUTPUT_PATH, f"char_{classes[prediction.item()]}_{confidence.item():3f}.png")
                img = orignal_images[img_index]
                cv2.imwrite(save_path, img)

        return list(zip(predicted_labels, images))
    else:
        return [tuple()]

if __name__=="__main__":
    from character_segmentation import segment_characters

    list_of_char_images = segment_characters("resources/good_example.jpg")
    char_images = classify_characters(list_of_char_images, model_path="resources/best_ResInceptionNet_model0.8811.pth", OUTPUT_PATH="predicted characters")
