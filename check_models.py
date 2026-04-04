import torch

print("🔍 Checking model files...\n")

# Check video model
print("=" * 50)
print("VIDEO MODEL (video_model.pth)")
print("=" * 50)
video_state = torch.load("video_model.pth", map_location="cpu")
for key, value in video_state.items():
    print(f"  {key}: {value.shape}")

# Check audio model
print("\n" + "=" * 50)
print("AUDIO MODEL (audio_model.pth)")
print("=" * 50)
audio_state = torch.load("audio_model.pth", map_location="cpu")
for key, value in audio_state.items():
    print(f"  {key}: {value.shape}")

# Check fusion model
print("\n" + "=" * 50)
print("FUSION MODEL (fusion_model.pth)")
print("=" * 50)
fusion_state = torch.load("fusion_model.pth", map_location="cpu")
for key, value in fusion_state.items():
    print(f"  {key}: {value.shape}")

print("\n✅ Done!")