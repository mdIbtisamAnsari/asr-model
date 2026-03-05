# System Requirements

## Training Requirements

### Minimum (Slow Training)

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GTX 1060 (6GB VRAM) |
| **RAM** | 16 GB |
| **Storage** | 50 GB SSD |
| **CPU** | Quad-core (Intel i5 / AMD Ryzen 5) |
| **OS** | Linux, Windows 10+, macOS |
| **Python** | 3.8+ |

### Recommended (Faster Training)

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA RTX 3080+ (10GB+ VRAM) |
| **RAM** | 32 GB |
| **Storage** | 100 GB NVMe SSD |
| **CPU** | 8+ cores (Intel i7 / AMD Ryzen 7) |
| **CUDA** | 11.7+ |

### Training Time Estimates

| Hardware | Time (100 epochs, Common Voice) |
|----------|--------------------------------|
| RTX 4090 | ~6-10 hours |
| RTX 3080 | ~12-24 hours |
| RTX 3060 | ~24-48 hours |
| GTX 1080 | ~48-72 hours |
| CPU only | ~1-2 weeks (not recommended) |

### GPU Memory Usage

| Batch Size | VRAM Required |
|------------|---------------|
| 32 | ~10-12 GB |
| 16 | ~6-8 GB |
| 8 | ~4-5 GB |
| 4 | ~2-3 GB |

---

## Inference Requirements

### Standard Inference (PyTorch)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 2 GB | 4 GB |
| **GPU** | Not required | Any NVIDIA GPU |
| **CPU** | Dual-core | Quad-core |
| **Storage** | 500 MB | 1 GB |

### Lightweight Inference (ONNX)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 512 MB | 1 GB |
| **GPU** | Not required | Not required |
| **CPU** | Single-core | Dual-core |
| **Storage** | 200 MB | 500 MB |

---

## Memory Comparison

| Runtime | RAM Usage | Best For |
|---------|-----------|----------|
| PyTorch (`inference.py`) | ~600 MB | Desktop/Laptop |
| ONNX (`inference_lite.py`) | ~150-200 MB | Embedded/Low-RAM |

---

## Inference Latency

### Per Utterance (3 seconds of audio)

| Hardware | PyTorch | ONNX |
|----------|---------|------|
| NVIDIA RTX 3080 | ~30-50 ms | ~20-40 ms |
| NVIDIA GTX 1060 | ~50-100 ms | ~40-80 ms |
| Intel i7 (CPU) | ~200-400 ms | ~150-300 ms |
| Intel i5 (CPU) | ~400-600 ms | ~300-500 ms |
| Raspberry Pi 4 | ~2-3 seconds | ~1-2 seconds |

---

## Supported Platforms

### Training

| Platform | Supported | Notes |
|----------|-----------|-------|
| Ubuntu 20.04+ | ✅ Yes | Recommended |
| Windows 10/11 | ✅ Yes | CUDA support |
| macOS (Apple Silicon) | ⚠️ Limited | MPS backend, slower |
| macOS (Intel) | ⚠️ Limited | CPU only |
| Google Colab | ✅ Yes | Free GPU |
| AWS/GCP/Azure | ✅ Yes | GPU instances |

### Inference

| Platform | PyTorch | ONNX |
|----------|---------|------|
| Ubuntu/Debian | ✅ | ✅ |
| Windows 10/11 | ✅ | ✅ |
| macOS | ✅ | ✅ |
| Raspberry Pi 4 | ⚠️ Slow | ✅ |
| Raspberry Pi 3 | ❌ | ⚠️ Very Slow |
| Android (via ONNX) | ❌ | ✅ |
| iOS (via ONNX) | ❌ | ✅ |

---

## Software Dependencies

### Training

```
Python >= 3.8
PyTorch >= 2.0.0
torchaudio >= 2.0.0
CUDA >= 11.7 (for GPU)
cuDNN >= 8.5 (for GPU)
```

### Inference (Full)

```
Python >= 3.8
PyTorch >= 2.0.0
torchaudio >= 2.0.0
sounddevice >= 0.4.6
```

### Inference (Lightweight)

```
Python >= 3.8
onnxruntime >= 1.15.0
librosa >= 0.10.0
sounddevice >= 0.4.6
numpy >= 1.24.0
```

---

## Low-Resource Configuration

For systems with limited resources, modify `config.py`:

```python
# Reduced model size (smaller memory footprint)
ENCODER_DIM = 128      # Default: 256
ENCODER_LAYERS = 2     # Default: 4
NUM_HEADS = 4          # Default: 8

# Reduced batch size
BATCH_SIZE = 4         # Default: 16
```

This reduces:
- Model parameters: ~75%
- GPU memory: ~60%
- Training time per epoch: ~50%
- Accuracy: ~5-10% lower

---

## Cloud Training Options

### Google Colab (Free)

```python
# In Colab notebook
!git clone <your-repo>
%cd arabic_stt
!pip install -r requirements.txt
!python train.py --use-common-voice --batch-size 8 --epochs 50
```

- GPU: NVIDIA T4 (16GB)
- RAM: 12-25 GB
- Time limit: ~12 hours/session

### AWS EC2

| Instance | GPU | Cost/Hour | Recommended |
|----------|-----|-----------|-------------|
| g4dn.xlarge | T4 16GB | ~$0.50 | Budget |
| g5.xlarge | A10G 24GB | ~$1.00 | Standard |
| p3.2xlarge | V100 16GB | ~$3.00 | Fast |

### RunPod / Lambda Labs

More cost-effective for GPU training:
- RTX 4090: ~$0.40/hour
- A100 40GB: ~$1.50/hour

---

## Disk Space Requirements

| Item | Size |
|------|------|
| Common Voice Arabic (download) | ~10 GB |
| Common Voice Arabic (extracted) | ~15 GB |
| Model checkpoints (per save) | ~30-50 MB |
| TensorBoard logs (full training) | ~100-500 MB |
| ONNX exported model | ~20-30 MB |
| **Total (training)** | **~30-50 GB** |
| **Total (inference only)** | **~100 MB** |

---

## Microphone Requirements

For real-time inference:

| Requirement | Specification |
|-------------|---------------|
| Sample Rate | 16000 Hz (resampled automatically) |
| Channels | Mono (stereo converted automatically) |
| Bit Depth | 16-bit or higher |
| Type | Any USB or built-in microphone |

### Recommended Microphones

- **Budget**: Any USB headset microphone
- **Better**: Blue Snowball, Fifine K669
- **Best**: Blue Yeti, Audio-Technica AT2020

### Audio Environment

- Quiet room recommended
- Background noise reduces accuracy
- Voice Activity Detection helps filter silence
