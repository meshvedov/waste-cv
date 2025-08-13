import torch
from torchvision import models
import torchvision.transforms.v2 as tfs
from PIL import Image
from pathlib import Path
import pandas as pd
import argparse

# Классы в том порядке, как в ImageFolder
class_names = ["O", "R"]

transform = tfs.Compose([
    tfs.Resize(255),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

def load_model(weights_path="waste_resnet18_best.pth", device="cpu"):
    # Загружаем ResNet18 с 2 выходами
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def predict_image(img_path, model, device="cpu"):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        class_name = class_names[pred.item()]
    return class_name

def predict_folder(folder_path, model, device="cpu", output_csv="predictions.csv"):
    folder = Path(folder_path)
    img_paths = list(folder.rglob("*.jpg"))

    results = []
    for img_path in img_paths:
        pred_class = predict_image(img_path, model, device)
        results.append({"image_path": str(img_path), "prediction": pred_class})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Предсказания сохранены в {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python Lightning script")
    parser.add_argument('--weights_path', type=str, default='./models/waste_resnet18_best.pth', help='Weights path')
    parser.add_argument('--data_dir', type=str, default='data/DATASET/TEST', help='Data path')
    parser.add_argument('--predictions_path', type=str, default='./data/predictions/results.csv', help='Predictions path')
    arg = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(arg.weights_path, device=device)
    
    predict_folder(arg.data_dir, model, device=device, output_csv=arg.predictions_path)
# python src/modeling/predict.py 
#     --weights_path ./models/checkpoint1.pth 
#     --data_dir .data/raw/DATA/TEST 
#     --predictions_path ./data/predictions/results.csv