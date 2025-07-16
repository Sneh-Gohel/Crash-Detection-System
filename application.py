import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

img_size = (180, 180)
class_names = ['Crash', 'No Crash']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'accident_detection_model.pth'

model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Select a video file",
    filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
)

if not video_path:
    print("No video file was selected.")
    exit()

video_name = os.path.splitext(os.path.basename(video_path))[0]
frame_dir = os.path.join("frames", video_name)
os.makedirs(frame_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
crash_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_path = os.path.join(frame_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, frame)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    if label == 'Crash':
        print(f"Crash detected at frame {frame_count}.")
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Frame {frame_count} — Crash Detected", color='red')
        plt.show() 
        crash_detected = True
        break
    else:
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Frame {frame_count} — No Crash", color='green')
        plt.pause(0.001) 

    frame_count += 1

cap.release()

if not crash_detected:
    print("No accident was detected in the video.")
    plt.close()  
