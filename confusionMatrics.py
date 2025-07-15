import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
MODEL_PATH = 'accident_detection_model.pth'
DATASET_PATH = 'E:\\AI\\crash detection\\data\\test'
IMG_SIZE = (180, 180)
class_names = ['Accident', 'Non Accident']
class_indices = {name: idx for idx, name in enumerate(class_names)}

# 1. Recreate the same model architecture (ResNet-50 with 2 outputs)
model = models.resnet50(weights=None)  # pretrained not needed since you're loading weights
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# 2. Load saved weights (state_dict)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# 3. Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# 4. Prediction loop
y_true = []
y_pred = []

for label in class_names:
    folder_path = os.path.join(DATASET_PATH, label)
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)  # add batch dimension

            with torch.no_grad():
                output = model(img_tensor)
                predicted_label = torch.argmax(output, dim=1).item()

            y_true.append(class_indices[label])
            y_pred.append(predicted_label)

# 5. Confusion matrix and report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

# 6. Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Classification Report:\n")
print(report)
