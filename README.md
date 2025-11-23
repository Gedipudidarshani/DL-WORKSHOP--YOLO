## DL-WORKSHOP--YOLO
## Project Title
Computer Vision Powered Image Search using YOLOv11
## Abstract / Introduction
This project implements a Computer Vision–based Image Search Application that processes a folder of images, extracts object detections using YOLOv11, and allows users to search images by detected objects. The system uses PyTorch, Ultralytics YOLO, and Streamlit for deployment.

The goal is to build a fast, scalable multimedia retrieval system using deep learning.
## Dataset & YOLO Model Details (COCO)

The model is trained on the COCO dataset (2017 version)

COCO contains:
118K training images
5K validation images
80 object classes

YOLOv11 model used:
yolo11m.pt
Pre-trained on COCO
Supports bounding box detection + class prediction
Sample classes: person, car, chair, dog, bottle, backpack, airplane, etc.
## Environment Setup
### Create Conda environment
```
conda create -n yolo_image_search_gpu python=3.11 -y
conda activate yolo_image_search_gpu
```
### Install dependencies
#### CPU installation:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics streamlit pillow numpy pandas tqdm
```
### GPU installation (NVIDIA CUDA 11.8):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics streamlit pillow numpy pandas tqdm
```
### GPU Installation Steps or CPU Installation Steps
#### GPU (NVIDIA CUDA)

Install GPU drivers (latest NVIDIA driver)
Install CUDA-enabled PyTorch:
```
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```
#### CPU

If your system does not have an NVIDIA GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
### How to Run in VS Code using Conda
Step 1 — Open folder in VS Code
File → Open Folder → Select project folder
Step 2 — Select Conda environment
Press Ctrl + Shift + P → Python: Select Interpreter → yolo_image_search_gpu

Step 3 — Run script
```
python app.py
```
Step 4 — If using Streamlit
```
streamlit run app.py
```
### How to Deploy using Streamlit

Inside your project folder:
```
streamlit run app.py
```
UI features:
Upload image directory
View detection previews
Search by detected objects
Load existing metadata
### Output Screenshots (UI + detection + VS Code terminal)

Include the following screenshots:

✅ Home Page of Streamlit App
✅ Image directory input & model selection page
✅ "Start Inference" output
✅ Object detection preview images
✅ Search results panel
✅ VS Code terminal screenshot showing no errors
✅ CUDA enabled (if GPU used)
```
CUDA available: True
CUDA version: 11.8
GPU Name: NVIDIA GeForce RTX …
```
### Enhancements / Innovations Added

You may include features such as:

🚀 GPU acceleration with CUDA

🔍 Fast reverse image search

🏷️ Metadata caching using JSON

🖼️ Thumbnail preview for faster UI rendering

⚡ Batch processing optimizations

⏱️ Progress bar (tqdm) integration

📁 Directory auto-scanning

💡 Custom YOLO model support
### 10. Results & Conclusion

This project successfully demonstrates:

Efficient object detection using YOLOv11

Real-time image search based on detected classes

Fast inference using GPU acceleration (if available)

User-friendly UI built using Streamlit

Scalable metadata storage for thousands of images

#### Conclusion:
The system is highly useful for multimedia retrieval, smart photo albums, CCTV object-based search, and intelligent image management applications.
