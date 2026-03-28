# CLAUDE.md - Project Context for Claude Code

## Project Overview

This is a **Multimodal Emotion Recognition** system trained on the RAVDESS dataset. It detects emotions from video using both facial expressions (video) and speech patterns (audio).

## Tech Stack

- **Python 3.9+**
- **PyTorch** - Deep learning framework
- **timm** - Pre-trained vision models (MobileNetV3)
- **Streamlit** - Web application framework
- **librosa** - Audio feature extraction (MFCC)
- **OpenCV** - Video frame extraction
- **FFmpeg** - Audio extraction from video

## Architecture

### Models (in `models/` directory)

1. **VideoTransformer** (`model.py`)
   - Uses `timm.create_model("mobilenetv3_small_100", num_classes=0)`
   - Output: 1024 features (via conv_head)
   - Input: RGB frames (3, 224, 224)

2. **AudioEncoder** (`audio_model.py`)
   - IMPORTANT: Uses `self.net` (not `self.encoder`) to match checkpoint
   - Architecture: Linear(40→128) → ReLU → Linear(128→256) → ReLU
   - Input: 40 MFCC coefficients
   - Output: 256 features

3. **CrossAttentionModel** (`fusion_model.py`)
   - video_proj: Linear(1024→256)
   - audio_proj: Linear(256→256)
   - attn: MultiheadAttention(256, 4 heads)
   - fc: Linear(256→128) → ReLU → Linear(128→4)
   - IMPORTANT: fc uses indices 0, 2 (no Dropout) to match checkpoint

### Utilities (in `utils/` directory)

- **video_utils.py**: Frame extraction with OpenCV
- **audio_utils.py**: FFmpeg audio extraction + librosa MFCC

## Pre-trained Checkpoints

Hosted on Google Drive:
- `video_model.pth` - ID: 1SMZL7V-nnaOItT-CvI2rm_Oyeg5zkS1i
- `audio_model.pth` - ID: 18grYKARGAt0qRz1qNui3tnWyzGCYdiu_
- `fusion_model.pth` - ID: 1RSPfbQToHUzRDOrlQse7kNXQ5TAGucAm

## Emotion Classes

| Index | Emotion | Emoji |
|-------|---------|-------|
| 0 | Happy | 😊 |
| 1 | Sad | 😢 |
| 2 | Angry | 😠 |
| 3 | Neutral | 😐 |

## Common Tasks

### Run the app
```bash
source venv/bin/activate
python -m streamlit run app.py
```

### Test models
```bash
python -c "from models import VideoTransformer; print('OK')"
```

### Inspect checkpoint
```bash
python -c "import torch; print(torch.load('video_model.pth').keys())"
```

## Critical Implementation Notes

1. **VideoTransformer**: Must use `num_classes=0` in timm to keep conv_head layer
2. **AudioEncoder**: Layer must be named `self.net` (checkpoint uses `net.0`, `net.2`)
3. **CrossAttentionModel**: fc Sequential must NOT have Dropout (checkpoint has fc.0, fc.2 only)
4. **MFCC**: Use 40 coefficients, averaged across time dimension

## File Locations

- Main app: `app.py`
- Models: `models/`
- Utilities: `utils/`
- Dependencies: `requirements.txt`
- Setup guide: `SETUP.md`

## Debugging Tips

- Dimension errors → Check model output shapes match expected inputs
- Import errors → Ensure venv is activated and packages installed
- FFmpeg errors → Install FFmpeg system-wide
- Model loading errors → Delete .pth files and re-download
