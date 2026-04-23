import os

# Set environment variables to prevent OpenBLAS memory allocation errors
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import cv2

torch.backends.mkldnn.enabled = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

NORM_MEAN = [0.7630392, 0.5456477, 0.57004845]
NORM_STD = [0.1409286, 0.15261266, 0.16997074]
IMAGE_SIZE = 224

with CLASS_NAMES_PATH.open("r", encoding="utf-8") as handle:
    CLASS_NAMES = json.load(handle)


def build_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = build_model()

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)


def prepare_image(image: Image.Image) -> torch.Tensor:
    tensor = PREPROCESS(image.convert("RGB"))
    return tensor.unsqueeze(0)


def predict(image: Image.Image):
    tensor = prepare_image(image)
    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
    
    top_index = int(torch.argmax(probabilities))
    confidence = float(probabilities[top_index])
    return probabilities.cpu().numpy(), top_index, confidence


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        if np.max(cam) > 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam


# Initialize GradCAM with the last convolutional layer of MobileNetV2
GRAD_CAM = GradCAM(MODEL, MODEL.features[-1])


def get_heatmap_overlay(image: Image.Image, class_idx: int):
    tensor = prepare_image(image)
    # Enable gradients for Grad-CAM
    tensor.requires_grad = True
    MODEL.train() # Set to train mode to allow gradient computation
    heatmap = GRAD_CAM.generate_heatmap(tensor, class_idx)
    MODEL.eval() # Set back to eval mode
    
    # Convert heatmap to RGB overlay
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Resize original image to match heatmap
    img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    
    # Clean up gradients to avoid memory leaks
    tensor.grad = None
    
    return Image.fromarray(img_array), Image.fromarray(heatmap_color)
