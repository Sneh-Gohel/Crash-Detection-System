import os
import ssl
import cv2
import torch
import tempfile
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image, ImageTk
from fpdf import FPDF
import yt_dlp
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

# CONFIG
img_size = (180, 180)
class_names = ['Crash', 'No Crash']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'accident_detection_model.pth'
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
ssl._create_default_https_context = ssl._create_unverified_context

# VIDEO DOWNLOAD (YOUTUBE)
def download_video_with_ytdlp(url):
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "yt_video.%(ext)s")
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'mp4[height<=360]',
        'quiet': True,
        'progress_hooks': [progress_hook]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    for file in os.listdir(temp_dir):
        if file.endswith(".mp4"):
            return os.path.join(temp_dir, file)
    raise Exception("MP4 not found.")

# LOAD MODEL
def load_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# CRASH DETECTION CORE
def detect_crash(video_path):
    model = load_model()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join("frames", video_name)
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    crash_detected = False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_step = max(total_frames // 100, 1)

    status_label.config(text="Analyzing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 3 != 0:
            frame_count += 1
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]

        if frame_count % progress_step == 0:
            progress.set(min(100, (frame_count * 100) // total_frames))
            root.update_idletasks()

        if label == 'Crash':
            crash_detected = True
            crash_frame_path = os.path.join(frame_dir, f"crash_frame_{frame_count}.jpg")
            cv2.imwrite(crash_frame_path, frame)
            export_crash_report(image, frame_count, crash_frame_path)
            show_crash_image(image, frame_count)
            break

        frame_count += 1

    cap.release()

    if not crash_detected:
        messagebox.showinfo("Result", "✅ No accident detected in the video.")
        status_label.config(text="Done")
    progress.set(0)

# EXPORT CRASH REPORT PDF
def export_crash_report(pil_img, frame_number, crash_frame_path):
    report_path = os.path.join("frames", f"crash_report_frame_{frame_number}.pdf")
    pil_img.save("temp_crash.jpg")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"Crash Detected at Frame: {frame_number}", ln=True)
    pdf.image("temp_crash.jpg", x=10, y=20, w=180)
    pdf.output(report_path)
    os.remove("temp_crash.jpg")
    messagebox.showinfo("Saved", f"📄 Crash report exported to:\n{report_path}")

# GUI IMAGE VIEW
def show_crash_image(pil_image, frame_id):
    crash_window = tk.Toplevel(root)
    crash_window.title(f"Crash Detected - Frame {frame_id}")
    imgtk = ImageTk.PhotoImage(pil_image.resize((400, 300)))
    label = tk.Label(crash_window, image=imgtk)
    label.image = imgtk
    label.pack()
    tk.Label(crash_window, text=f"🚨 Crash at Frame {frame_id}", fg="red", font=("Arial", 14)).pack(pady=5)

# YOUTUBE BUTTON
def handle_youtube():
    url = url_entry.get().strip()
    if not url.startswith("http"):
        messagebox.showerror("Error", "Please enter a valid YouTube URL.")
        return
    status_label.config(text="Downloading video...")
    try:
        video_path = download_video_with_ytdlp(url)
        status_label.config(text="Video downloaded. Analyzing...")
        detect_crash(video_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# UPLOAD VIDEO BUTTON
def handle_upload():
    path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if path:
        status_label.config(text="Processing uploaded video...")
        detect_crash(path)

# YT-DLP Progress Hook
def progress_hook(d):
    if d['status'] == 'downloading':
        downloaded = d.get('downloaded_bytes', 0)
        total = d.get('total_bytes', 1)
        percent = int((downloaded / total) * 100)
        progress.set(percent)
        root.update_idletasks()

# ---------------- GUI ---------------- #
root = tk.Tk()
root.title("🚗 Crash Detection App")
root.geometry("550x350")
root.configure(bg="#f0f0f0")

tk.Label(root, text="🚨 Crash Detection from YouTube / Local Video", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)

url_entry = tk.Entry(root, width=50)
url_entry.insert(0, "Paste YouTube URL here...")
url_entry.pack(pady=5)

tk.Button(root, text="▶ Detect from YouTube", command=handle_youtube, bg="#007acc", fg="white").pack(pady=5)
tk.Button(root, text="📁 Upload Local Video", command=handle_upload, bg="#5cb85c", fg="white").pack(pady=5)

progress = tk.IntVar()
ttk.Progressbar(root, variable=progress, length=300, maximum=100).pack(pady=10)

status_label = tk.Label(root, text="", bg="#f0f0f0", fg="green", font=("Arial", 10))
status_label.pack(pady=5)

root.mainloop()
