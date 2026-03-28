# 🎭 RAVDESS Emotion Recognition - Complete Setup Guide

This guide will help you set up the project in **VS Code** with **Claude Code** for continued development.

---

## 📋 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Project Setup](#-project-setup)
3. [VS Code Configuration](#-vs-code-configuration)
4. [Claude Code Setup](#-claude-code-setup)
5. [Running the Application](#-running-the-application)
6. [Project Structure](#-project-structure)
7. [How It Works](#-how-it-works)
8. [Development Guide](#-development-guide)
9. [Troubleshooting](#-troubleshooting)

---

## 🔧 Prerequisites

### 1. Install Python 3.9+

```bash
# Check Python version
python3 --version

# Should be 3.9 or higher
```

### 2. Install FFmpeg (Required for audio extraction)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg -y

# Windows (using Chocolatey)
choco install ffmpeg

# Verify installation
ffmpeg -version
```

### 3. Install VS Code

Download from: https://code.visualstudio.com/

### 4. Install Node.js (Required for Claude Code)

```bash
# macOS
brew install node

# Ubuntu/Debian
sudo apt install nodejs npm -y

# Verify
node --version  # Should be 18+
```

---

## 📁 Project Setup

### Step 1: Create Project Directory

```bash
# Navigate to your preferred location
cd ~/Documents  # or wherever you want

# Create project folder
mkdir emotion-recognition
cd emotion-recognition
```

### Step 2: Extract Project Files

If you downloaded the ZIP file:
```bash
# Unzip into current directory
unzip ~/Downloads/emotion-recognition-vscode.zip -d .
mv emotion-recognition-vscode/* .
rm -rf emotion-recognition-vscode
```

### Step 3: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate

# Verify activation (should show venv path)
which python
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (for Mac with Apple Silicon)
pip install torch torchvision torchaudio

# Install all other dependencies
pip install -r requirements.txt
```

### Step 5: Download Pre-trained Models

The models will download automatically on first run, but you can also download manually:

```bash
# Install gdown if not already
pip install gdown

# Download models
python -c "
import gdown
import os

models = {
    'video_model.pth': '1SMZL7V-nnaOItT-CvI2rm_Oyeg5zkS1i',
    'audio_model.pth': '18grYKARGAt0qRz1qNui3tnWyzGCYdiu_',
    'fusion_model.pth': '1RSPfbQToHUzRDOrlQse7kNXQ5TAGucAm'
}

for name, id in models.items():
    if not os.path.exists(name):
        print(f'Downloading {name}...')
        gdown.download(f'https://drive.google.com/uc?id={id}', name)
        print(f'Downloaded {name}')
"
```

---

## 💻 VS Code Configuration

### Step 1: Open Project in VS Code

```bash
# From project directory
code .
```

### Step 2: Install Recommended Extensions

Open VS Code and install these extensions (Cmd/Ctrl + Shift + X):

| Extension | ID | Purpose |
|-----------|-----|---------|
| Python | `ms-python.python` | Python language support |
| Pylance | `ms-python.vscode-pylance` | Python IntelliSense |
| Python Debugger | `ms-python.debugpy` | Debugging support |
| Claude Code | `anthropic.claude-code` | AI coding assistant |

Or install via command line:
```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
```

### Step 3: Select Python Interpreter

1. Press `Cmd/Ctrl + Shift + P`
2. Type "Python: Select Interpreter"
3. Choose the venv interpreter: `./venv/bin/python`

### Step 4: VS Code Settings

The project includes `.vscode/settings.json` with recommended settings.

---

## 🤖 Claude Code Setup

### Step 1: Install Claude Code CLI

```bash
# Install globally via npm
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

### Step 2: Authenticate

```bash
# Login to Claude
claude login

# This will open a browser for authentication
```

### Step 3: Initialize in Project

```bash
# Navigate to project
cd ~/Documents/emotion-recognition

# Initialize Claude Code
claude init
```

### Step 4: Using Claude Code

```bash
# Start interactive session
claude

# Or ask a specific question
claude "explain how the cross-attention fusion works in this project"

# Get help with code
claude "add a new emotion class 'surprised' to the model"

# Debug errors
claude "why am I getting a dimension mismatch error?"
```

### Claude Code Tips

- Use `claude` in the terminal for quick questions
- Use `/help` inside Claude Code for commands
- Use `/add <file>` to add files to context
- Use `/clear` to reset conversation

---

## 🚀 Running the Application

### Option 1: Streamlit Web App

```bash
# Make sure venv is activated
source venv/bin/activate

# Run the app
python -m streamlit run app.py

# Or simply
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Option 2: VS Code Debug

1. Press `F5` or go to Run > Start Debugging
2. Select "Streamlit App" configuration
3. Browser will open automatically

### Option 3: Test Individual Components

```bash
# Test video model
python -c "
from models.model import VideoTransformer
import torch
model = VideoTransformer()
x = torch.randn(1, 3, 224, 224)
out = model(x)
print(f'Video output shape: {out.shape}')  # Should be [1, 1024]
"

# Test audio model
python -c "
from models.audio_model import AudioEncoder
import torch
model = AudioEncoder()
x = torch.randn(1, 40)
out = model(x)
print(f'Audio output shape: {out.shape}')  # Should be [1, 256]
"
```

---

## 📂 Project Structure

```
emotion-recognition/
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 requirements.txt          # Python dependencies
├── 📄 SETUP.md                  # This setup guide
├── 📄 README.md                 # Project overview
├── 📄 CLAUDE.md                 # Instructions for Claude Code
│
├── 📁 models/
│   ├── __init__.py              # Package exports
│   ├── model.py                 # VideoTransformer (MobileNetV3)
│   ├── audio_model.py           # AudioEncoder (MFCC → features)
│   └── fusion_model.py          # CrossAttentionModel (fusion)
│
├── 📁 utils/
│   ├── __init__.py              # Package exports
│   ├── video_utils.py           # Frame extraction
│   └── audio_utils.py           # Audio/MFCC extraction
│
├── 📁 .vscode/
│   ├── settings.json            # VS Code settings
│   └── launch.json              # Debug configurations
│
├── 📁 venv/                     # Virtual environment (created by you)
│
└── 📄 *.pth                     # Model weights (downloaded)
    ├── video_model.pth
    ├── audio_model.pth
    └── fusion_model.pth
```

---

## 🧠 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT VIDEO                             │
└─────────────────────────────────────────────────────────────┘
                 │                              │
                 ▼                              ▼
      ┌─────────────────┐            ┌─────────────────┐
      │  Extract Frames │            │  Extract Audio  │
      │   (5 samples)   │            │   (FFmpeg)      │
      └─────────────────┘            └─────────────────┘
                 │                              │
                 ▼                              ▼
      ┌─────────────────┐            ┌─────────────────┐
      │  MobileNetV3    │            │  MFCC Features  │
      │  + conv_head    │            │  (40 coeffs)    │
      │  → 1024 dims    │            │  → 40 dims      │
      └─────────────────┘            └─────────────────┘
                 │                              │
                 │                              ▼
                 │                   ┌─────────────────┐
                 │                   │  AudioEncoder   │
                 │                   │  40→128→256     │
                 │                   └─────────────────┘
                 │                              │
                 └──────────┬───────────────────┘
                            ▼
              ┌───────────────────────────┐
              │   Cross-Attention Fusion  │
              │   Video queries Audio     │
              │   256 → 128 → 4 classes   │
              └───────────────────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │  😊 Happy    😢 Sad       │
              │  😠 Angry    😐 Neutral   │
              └───────────────────────────┘
```

### Key Components

1. **VideoTransformer** (`models/model.py`)
   - MobileNetV3 Small backbone
   - conv_head projects to 1024 features
   - Processes 5 frames, averages features

2. **AudioEncoder** (`models/audio_model.py`)
   - Takes 40 MFCC coefficients
   - Two linear layers: 40→128→256
   - Captures speech emotion patterns

3. **CrossAttentionModel** (`models/fusion_model.py`)
   - Projects video (1024) and audio (256) to 256
   - Multi-head attention (4 heads)
   - Final classifier: 256→128→4

---

## 🛠 Development Guide

### Adding a New Emotion Class

1. Modify `fusion_model.py`:
```python
# Change num_classes from 4 to 5
self.fc = nn.Sequential(
    nn.Linear(hidden_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 5)  # 5 classes now
)
```

2. Update `app.py`:
```python
EMOTIONS = ["😊 Happy", "😢 Sad", "😠 Angry", "😐 Neutral", "😨 Fearful"]
```

3. Retrain the model with new data.

### Improving Accuracy

1. **More frames**: Change `max_frames=5` to `max_frames=10` in `app.py`
2. **Data augmentation**: Add transforms in training
3. **Larger model**: Use MobileNetV3 Large instead of Small

### Adding Real-time Webcam

```python
# Add to app.py
import cv2

if st.button("📹 Use Webcam"):
    cap = cv2.VideoCapture(0)
    # Process frames...
```

---

## ❓ Troubleshooting

### "No module named 'torch'"

```bash
pip install torch torchvision torchaudio
```

### "No module named 'cv2'"

```bash
pip install opencv-python
```

### "FFmpeg not found"

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

### "CUDA out of memory"

The app runs on CPU by default. If using GPU:
```python
# In app.py, change:
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### "Model dimension mismatch"

Delete cached models and re-download:
```bash
rm -f video_model.pth audio_model.pth fusion_model.pth
python -m streamlit run app.py
```

### VS Code doesn't recognize imports

1. Make sure venv is selected as interpreter
2. Reload VS Code window (Cmd/Ctrl + Shift + P → "Reload Window")

---

## 📚 Resources

- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)

---

## 📝 License

This project is for educational purposes (Capstone Project).

---

**Happy Coding! 🚀**
