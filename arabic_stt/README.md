# نظام التعرف على الكلام العربي
# Arabic Speech-to-Text Model

A custom deep learning model for Arabic speech recognition with real-time inference capabilities.

## Features

- **Conformer-based Architecture**: Uses a state-of-the-art Conformer encoder with CTC loss
- **Arabic Language Support**: Full support for Arabic characters, diacritics (tashkeel), and Eastern Arabic numerals
- **Real-time Inference**: Microphone input with Voice Activity Detection (VAD)
- **Common Voice Integration**: Easy training with Mozilla Common Voice Arabic dataset
- **TensorBoard Logging**: Track training progress with visualization
- **Lightweight Deployment**: ONNX export for low-memory devices (500 MB - 1 GB RAM)

## Project Structure

```
arabic_stt/
├── config.py              # Configuration and hyperparameters
├── model.py               # Conformer-based STT model architecture
├── dataset.py             # Data loading and preprocessing
├── train.py               # Training pipeline
├── inference.py           # Real-time inference (PyTorch)
├── inference_lite.py      # Lightweight inference (ONNX)
├── export_onnx.py         # Export model to ONNX format
├── create_manifest.py     # Create TSV manifest from audio files
├── utils.py               # Augmentation and utilities
├── requirements.txt       # Full dependencies (training)
├── requirements_lite.txt  # Minimal dependencies (inference)
├── checkpoints/           # Saved model checkpoints
├── logs/                  # TensorBoard logs
└── data/                  # Training data (if using custom dataset)
```

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyAudio (for real-time inference)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Windows:**
```bash
pip install pyaudio
```

## Training

### Using Mozilla Common Voice Dataset

The easiest way to train is using the Common Voice Arabic dataset, which is automatically downloaded:

```bash
python train.py --use-common-voice --batch-size 16 --epochs 100
```

**Note**: First run will download the dataset (~10GB), which may take some time.

### Using Custom Dataset

#### Step 1: Prepare Your Audio Files

Organize your Arabic audio recordings (WAV, MP3, FLAC supported).

#### Step 2: Create a TSV Manifest

**Option A: Create from audio directory**

If you have audio files with matching `.txt` transcript files:
```
audio/
  recording1.wav
  recording1.txt   # contains: مرحبا بالعالم
  recording2.wav
  recording2.txt   # contains: كيف حالك
```

Run:
```bash
python create_manifest.py from-dir --audio-dir /path/to/audio --output data/manifest.tsv
```

**Option B: Create from CSV**

If you have a CSV with audio paths and transcripts:
```bash
python create_manifest.py from-csv --csv data.csv --audio-column audio_path --transcript-column text
```

**Option C: Create empty template and edit manually**
```bash
python create_manifest.py template --output data/manifest.tsv
```

Then edit `data/manifest.tsv`:
```
/path/to/audio1.wav	السلام عليكم
/path/to/audio2.wav	مرحبا كيف حالك
```

#### Step 3: Split into Train/Validation Sets

```bash
python create_manifest.py split --manifest data/manifest.tsv --train-ratio 0.9
```

This creates `data/train.tsv` (90%) and `data/val.tsv` (10%).

#### Step 4: Train

```bash
python train.py \
    --train-manifest data/train.tsv \
    --val-manifest data/val.tsv \
    --batch-size 16 \
    --epochs 100
```

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--use-common-voice` | Use Common Voice dataset | True |
| `--train-manifest` | Path to training manifest | None |
| `--val-manifest` | Path to validation manifest | None |
| `--batch-size` | Batch size | 16 |
| `--epochs` | Number of epochs | 100 |
| `--lr` | Learning rate | 3e-4 |
| `--resume` | Resume from checkpoint | None |
| `--device` | Device (cuda/cpu) | auto |
| `--num-workers` | DataLoader workers | 4 |

### Monitor Training

```bash
tensorboard --logdir logs/
```

## Inference

### Real-time Microphone Input

```bash
python inference.py --model checkpoints/best_model.pt
```

This will:
1. Load the trained model
2. Listen to your microphone
3. Transcribe Arabic speech in real-time

### Transcribe Audio File

```bash
python inference.py --model checkpoints/best_model.pt --audio-file path/to/audio.wav
```

### List Audio Devices

```bash
python inference.py --model checkpoints/best_model.pt --list-devices
```

### Select Specific Microphone

```bash
python inference.py --model checkpoints/best_model.pt --input-device 2
```

## Model Architecture

The model uses a Conformer-based encoder, which combines:

1. **Convolutional Subsampling**: Reduces sequence length by 4x
2. **Conformer Blocks**: Each block contains:
   - Feed-forward module (half-step)
   - Multi-head self-attention
   - Convolution module
   - Feed-forward module (half-step)
3. **CTC Decoder**: Connectionist Temporal Classification for end-to-end training

### Default Configuration

| Parameter | Value |
|-----------|-------|
| Encoder Dimension | 256 |
| Encoder Layers | 4 |
| Attention Heads | 8 |
| Mel Filterbanks | 80 |
| Sample Rate | 16000 Hz |

## Dataset Requirements

### Audio Files
- Format: WAV, MP3, FLAC (any format supported by torchaudio)
- Sample Rate: Any (will be resampled to 16kHz)
- Channels: Mono or Stereo (will be converted to mono)

### Transcripts
- UTF-8 encoded Arabic text
- Can include diacritics (tashkeel)
- Punctuation supported

## Supported Characters

The model supports:
- All Arabic letters (ء - ي)
- Common diacritics (ً ٌ ٍ َ ُ ِ ّ ْ)
- Arabic punctuation (، ؟ !)
- Eastern Arabic numerals (٠-٩)
- Basic Latin punctuation

## Evaluation Metrics

The training pipeline reports:
- **CTC Loss**: Training objective
- **WER (Word Error Rate)**: Word-level accuracy
- **CER (Character Error Rate)**: Character-level accuracy

## Tips for Better Results

1. **More Data**: Use larger datasets for better generalization
2. **Data Augmentation**: Add noise, speed perturbation, and spec augment
3. **Longer Training**: Train for more epochs (100+ recommended)
4. **Learning Rate**: Use warmup and cosine annealing
5. **Batch Size**: Larger batches often help (if GPU memory allows)

## Lightweight Deployment (Low RAM Devices)

For devices with limited RAM (500 MB - 1 GB), export to ONNX format:

### Export Model to ONNX

```bash
python export_onnx.py --checkpoint checkpoints/best_model.pt --output arabic_stt.onnx
```

### Run Lightweight Inference

```bash
# Install lightweight dependencies only
pip install -r requirements_lite.txt

# Real-time inference
python inference_lite.py --model arabic_stt.onnx

# Transcribe file
python inference_lite.py --model arabic_stt.onnx --audio-file audio.wav
```

### Memory Comparison

| Runtime | RAM Usage | Dependencies |
|---------|-----------|--------------|
| PyTorch (`inference.py`) | ~600 MB | torch, torchaudio |
| ONNX (`inference_lite.py`) | ~150-200 MB | onnxruntime, librosa |

### Supported Devices

With ONNX deployment:
- Raspberry Pi 3/4 (1 GB RAM)
- Older laptops with 2 GB RAM
- Embedded systems
- Edge devices

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 8`
- Use gradient accumulation (modify train.py)

### Audio Device Not Found
- List devices: `python inference.py --list-devices`
- Install PortAudio (see Installation)

### Slow Training
- Reduce `--num-workers` if CPU-bound
- Use SSD for dataset storage
- Enable mixed precision training (modify train.py)

## License

This project is for educational purposes. Please respect the licenses of any datasets you use.

## References

- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [CTC: Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
