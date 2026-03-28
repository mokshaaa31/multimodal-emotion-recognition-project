# 🎭 Multimodal Emotion Recognition

Detect emotions from video using **facial expressions** and **speech patterns**.

Trained on the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

## 🚀 Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Test
python test_models.py

# 3. Run
python -m streamlit run app.py
```

## 📊 Emotions Detected

| Emotion | Emoji |
|---------|-------|
| Happy | 😊 |
| Sad | 😢 |
| Angry | 😠 |
| Neutral | 😐 |

## 🏗️ Architecture

```
Video → MobileNetV3 → 1024 features ─┐
                                     ├→ Cross-Attention → Classifier → Emotion
Audio → MFCC → Encoder → 256 features┘
```

## 📁 Project Structure

```
├── app.py              # Streamlit web app
├── models/
│   ├── model.py        # VideoTransformer
│   ├── audio_model.py  # AudioEncoder
│   └── fusion_model.py # CrossAttentionModel
├── utils/
│   ├── video_utils.py  # Frame extraction
│   └── audio_utils.py  # Audio processing
├── test_models.py      # Verification script
├── SETUP.md            # Detailed setup guide
└── CLAUDE.md           # Claude Code context
```

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Complete setup guide for VS Code & Claude Code
- **[CLAUDE.md](CLAUDE.md)** - Project context for Claude Code AI assistant

## 🛠️ Requirements

- Python 3.9+
- FFmpeg
- ~2GB disk space for models

## 📝 License

Educational purposes (Capstone Project)
