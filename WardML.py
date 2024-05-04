import torch
from torchvision import transforms
from PIL import Image
from models.resnet_finetune import ResNet50FineTune as Net
import json

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model_path = './checkpoints/final_model.pth'  # Path to the preloaded model file

    # Load the model
    num_classes = 63
    model = Net(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(local_model_path, map_location=device))
    model.eval()
    return model

def process_image(image_stream):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_stream).convert('RGB')
    return transform(image).unsqueeze(0)

def load_class_index_mapping(class_to_idx_path):
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    return idx_to_class

def classify_image(image_stream, model, device, idx_to_class):
    image = process_image(image_stream).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class_index = predicted.item()
    predicted_class_name = idx_to_class.get(str(predicted_class_index), "Unknown Class")
    return predicted_class_name
