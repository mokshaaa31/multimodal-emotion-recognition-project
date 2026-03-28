"""
Test script to verify all models load and work correctly.
Run this to check your setup before running the main app.

Usage:
    python test_models.py
"""

import torch
import sys

def test_video_model():
    """Test VideoTransformer model."""
    print("\n🎥 Testing VideoTransformer...")
    
    from models.model import VideoTransformer
    
    model = VideoTransformer()
    print(f"   ✓ Model created")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    
    expected_shape = (1, 1024)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"   ✓ Output shape: {out.shape} (correct!)")
    
    return True


def test_audio_model():
    """Test AudioEncoder model."""
    print("\n🎧 Testing AudioEncoder...")
    
    from models.audio_model import AudioEncoder
    
    model = AudioEncoder()
    print(f"   ✓ Model created")
    
    # Test forward pass
    x = torch.randn(1, 40)
    with torch.no_grad():
        out = model(x)
    
    expected_shape = (1, 256)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"   ✓ Output shape: {out.shape} (correct!)")
    
    return True


def test_fusion_model():
    """Test CrossAttentionModel."""
    print("\n🔗 Testing CrossAttentionModel...")
    
    from models.fusion_model import CrossAttentionModel
    
    model = CrossAttentionModel()
    print(f"   ✓ Model created")
    
    # Test forward pass
    video_feat = torch.randn(1, 1024)
    audio_feat = torch.randn(1, 256)
    
    with torch.no_grad():
        out = model(video_feat, audio_feat)
    
    expected_shape = (1, 4)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"   ✓ Output shape: {out.shape} (correct!)")
    
    return True


def test_checkpoint_loading():
    """Test loading pre-trained checkpoints."""
    print("\n📦 Testing checkpoint loading...")
    
    import os
    from models.model import VideoTransformer
    from models.audio_model import AudioEncoder
    from models.fusion_model import CrossAttentionModel
    
    checkpoints = {
        "video_model.pth": VideoTransformer,
        "audio_model.pth": AudioEncoder,
        "fusion_model.pth": CrossAttentionModel
    }
    
    all_loaded = True
    
    for filename, model_class in checkpoints.items():
        if os.path.exists(filename):
            try:
                model = model_class()
                model.load_state_dict(torch.load(filename, map_location="cpu"))
                print(f"   ✓ {filename} loaded successfully")
            except Exception as e:
                print(f"   ✗ {filename} failed: {e}")
                all_loaded = False
        else:
            print(f"   ⚠ {filename} not found (will download on first app run)")
    
    return all_loaded


def test_utils():
    """Test utility functions."""
    print("\n🔧 Testing utilities...")
    
    # Test video utils import
    from utils.video_utils import get_frames, preprocess_frame
    print("   ✓ video_utils imported")
    
    # Test audio utils import
    from utils.audio_utils import extract_audio, extract_audio_features
    print("   ✓ audio_utils imported")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 RAVDESS Emotion Recognition - Model Tests")
    print("=" * 60)
    
    tests = [
        ("Video Model", test_video_model),
        ("Audio Model", test_audio_model),
        ("Fusion Model", test_fusion_model),
        ("Utilities", test_utils),
        ("Checkpoints", test_checkpoint_loading),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {name}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! You're ready to run the app.")
        print("\nRun: python -m streamlit run app.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
