import torch
import json  
from torchvision import transforms
from PIL import Image
from models.resnet_finetune import ResNet50FineTune as Net

def load_model(model_path, device):
    # Update the number of classes to match the trained model
    num_classes = 63  # This should match the original model's configuration
    model = Net(num_classes=num_classes).to(device)  # Pass the correct number of classes to the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_image(image_path):
    transform = transforms.Compose([
        # transforms.Grayscale(),  # Remove this line as ResNet expects 3-channel RGB images
        transforms.Resize((224, 224)),  # Adjust to match ResNet input dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_class_index_mapping(class_to_idx_path):
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    return idx_to_class

def classify_image(image_path, model, device, idx_to_class):
    image = process_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class_index = predicted.item()
    predicted_class_name = idx_to_class.get(str(predicted_class_index), "Unknown Class")
    return predicted_class_name

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    model_path = './checkpoints/final_model.pth'
    class_to_idx_path = './checkpoints/class_to_idx.json'  # Path to the class_to_idx json file
    image_path = input("Enter the path of the image: ")

    # Load the trained model
    model = load_model(model_path, device)

    # Load the class index mapping
    idx_to_class = load_class_index_mapping(class_to_idx_path)

    # Classify the image
    predicted_class = classify_image(image_path, model, device, idx_to_class)
    print(f'The image is classified as: {predicted_class}')

if __name__ == '__main__':
    main()
