# Quranic Arabic ASR for Microcontrollers

Lightweight Automatic Speech Recognition for Quranic Arabic, optimized for embedded devices like ESP32-S3, STM32H7, and similar MCUs.

## Overview

This project provides a tiny ASR model (~500KB INT8) that can run on microcontrollers with limited resources. It uses CTC decoding for simple, efficient inference without autoregressive loops.

### Why Not Whisper?

| Model | Parameters | Size (INT8) | MCU Compatible |
|-------|------------|-------------|----------------|
| Whisper-tiny | 39M | ~40MB | ❌ Too large |
| Whisper-small | 244M | ~250MB | ❌ Too large |
| **TinyQuranASR** | **~200K** | **~500KB** | ✅ Yes |

## Supported Platforms

| Platform | RAM | Flash | Status |
|----------|-----|-------|--------|
| ESP32-S3 | 512KB | 8MB | ✅ Supported |
| STM32H7 | 1MB | 2MB | ✅ Supported |
| nRF5340 | 512KB | 1MB | ✅ Supported |
| Raspberry Pi Pico | 264KB | 2MB | ⚠️ Edge case |

### Memory Requirements

| Component | Size |
|-----------|------|
| Model (Flash) | ~500KB |
| Tensor Arena (RAM) | ~150KB |
| Audio Buffer (RAM) | ~160KB |
| **Total RAM** | **~320KB minimum** |

---

## Procedure

### Step 1: Train the Model (Google Colab)

1. **Upload notebook to Colab**
   - Open [Google Colab](https://colab.research.google.com/)
   - Upload `Quranic_ASR_MCU.ipynb`
   - Select Runtime → Change runtime type → **GPU (T4)**

2. **Run all cells**
   - The notebook will:
     - Install dependencies
     - Load the `rabah2026/Quran-Ayah-Corpus` dataset (streaming mode)
     - Train the TinyQuranASR model with CTC loss
     - Export to TensorFlow Lite INT8
     - Generate MCU deployment files

3. **Download generated files**
   - `quran_asr_mcu_int8.tflite` - TFLite INT8 model
   - `quran_asr_model.h` - Model as C array
   - `quran_asr_mcu.cpp` - Inference code
   - `quran_asr_config.h` - Configuration

### Step 2: Set Up MCU Development Environment

#### For ESP32-S3 (ESP-IDF)

```bash
# Install ESP-IDF
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh && source export.sh

# Create project
idf.py create-project quran_asr
cd quran_asr

# Add TFLite Micro dependency
idf.py add-dependency "espressif/esp-tflite-micro"
```

#### For STM32 (STM32CubeIDE)

1. Create new STM32 project in STM32CubeIDE
2. Enable X-CUBE-AI or add TFLite Micro manually
3. Enable CMSIS-DSP for mel spectrogram computation

#### For Arduino

```cpp
// Install library via Arduino IDE
// Sketch → Include Library → Manage Libraries
// Search: "Arduino_TensorFlowLite"
```

### Step 3: Add Generated Files to Project

```
your_project/
├── main/
│   ├── quran_asr_model.h      # Model weights (C array)
│   ├── quran_asr_mcu.cpp      # Inference code
│   ├── quran_asr_config.h     # Configuration
│   └── main.cpp               # Your application
├── CMakeLists.txt
└── ...
```

### Step 4: Implement Mel Spectrogram

The model expects mel spectrogram input. Implement for your platform:

#### ESP32 (using ESP-DSP)

```cpp
#include "esp_dsp.h"

void compute_mel_spectrogram(const int16_t* audio, int len, int8_t* mel_out) {
    // Use dsps_fft2r_fc32 for FFT
    // Apply mel filterbank
    // Quantize to INT8
}
```

#### STM32 (using CMSIS-DSP)

```cpp
#include "arm_math.h"

void compute_mel_spectrogram(const int16_t* audio, int len, int8_t* mel_out) {
    // Use arm_rfft_q15 for FFT
    // Apply mel filterbank
    // Quantize to INT8
}
```

### Step 5: Integrate ASR in Your Application

```cpp
#include "quran_asr_mcu.cpp"
#include "quran_asr_config.h"

QuranASR asr;
char transcription[512];

void setup() {
    Serial.begin(115200);
    
    if (!asr.Init()) {
        Serial.println("ASR init failed!");
        while(1);
    }
    Serial.println("Quranic ASR ready!");
}

void loop() {
    // 1. Record audio (16kHz, mono, max 5 seconds)
    int16_t audio_buffer[QURAN_ASR_MAX_SAMPLES];
    int audio_len = record_audio(audio_buffer);
    
    // 2. Compute mel spectrogram
    int8_t mel_buffer[QURAN_ASR_MAX_FRAMES * QURAN_ASR_N_MELS];
    compute_mel_spectrogram(audio_buffer, audio_len, mel_buffer);
    
    // 3. Run ASR
    if (asr.Transcribe(mel_buffer, transcription, sizeof(transcription))) {
        Serial.print("Transcription: ");
        Serial.println(transcription);
    }
    
    delay(1000);
}
```

### Step 6: Build and Flash

#### ESP32-S3

```bash
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

#### STM32

Build in STM32CubeIDE and flash via ST-Link.

#### Arduino

Upload via Arduino IDE.

---

## Audio Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample Rate | 16000 Hz | Standard for speech |
| Max Duration | 5 seconds | Adjustable in config |
| Mel Bins | 40 | Reduced for MCU |
| FFT Size | 512 | Smaller than default |
| Hop Length | 160 | 10ms frame shift |

---

## Model Architecture

```
TinyQuranASR
├── Conv2D (1→32, stride 2)
├── DepthwiseSeparableConv (32→64, stride 2)
├── DepthwiseSeparableConv (64→128, stride 2)
├── DepthwiseSeparableConv (128→128)
├── AdaptiveAvgPool
├── GRU (128 hidden, 2 layers)
└── Linear (128→vocab_size)
```

- **Input:** Mel spectrogram (500 frames × 40 mels)
- **Output:** CTC logits for greedy decoding
- **Parameters:** ~200K
- **Size (INT8):** ~500KB

---

## Character Set

The model recognizes Quranic Arabic characters including:

- Basic Arabic letters (ا-ي)
- Tashkeel (diacritics): فَتْحَة، ضَمَّة، كَسْرَة، سُكُون، شَدَّة
- Tanween: ً، ٌ، ٍ
- Quranic stop marks: ۖ، ۗ، ۘ، etc.

---

## Troubleshooting

### Model fails to initialize
- Check tensor arena size (increase if needed)
- Verify all TFLite ops are registered

### Poor transcription quality
- Ensure audio is 16kHz mono
- Check mel spectrogram normalization
- Verify quantization parameters match

### Out of memory
- Reduce `QURAN_ASR_MAX_FRAMES` for shorter audio
- Use external PSRAM if available (ESP32-S3)

---

## Files Description

| File | Description |
|------|-------------|
| `Quranic_ASR_MCU.ipynb` | Training notebook for Google Colab |
| `quran_asr_mcu_int8.tflite` | TFLite INT8 model |
| `quran_asr_model.h` | Model as C array for embedding |
| `quran_asr_mcu.cpp` | Complete inference code |
| `quran_asr_config.h` | Configuration parameters |

---

## License

This project uses the `rabah2026/Quran-Ayah-Corpus` dataset. Please check the dataset license for usage terms.

---

## Contributing

Contributions welcome! Areas for improvement:
- Mel spectrogram implementations for different platforms
- Beam search CTC decoding
- Additional language/dialect support
