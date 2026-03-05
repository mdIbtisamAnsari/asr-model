"""
Lightweight ONNX Inference for Arabic Speech-to-Text

Optimized for low-memory devices (500 MB - 1 GB RAM).
Uses ONNX Runtime instead of PyTorch for ~3x less memory.

Memory usage: ~150-200 MB (vs ~600 MB with PyTorch)
"""
import argparse
import sys
import queue
import threading
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("Please install librosa: pip install librosa")
    sys.exit(1)


# Minimal config (no PyTorch/torch dependencies)
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
BLANK_IDX = 0

# Arabic character mapping
ARABIC_CHARS = [
    '<blank>', '<sos>', '<eos>', '<pad>', '<unk>', ' ',
    'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ',
    'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق',
    'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ى',
    'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ',
    '،', '؟', '!', '.', ':', '-',
    '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩',
]

IDX_TO_CHAR = {idx: char for idx, char in enumerate(ARABIC_CHARS)}


class LightweightAudioProcessor:
    """
    Audio processor using librosa (numpy-based, no PyTorch).
    """
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.n_mels = N_MELS
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.win_length = WIN_LENGTH
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel spectrogram features.
        
        Args:
            audio: numpy array of audio samples
        Returns:
            features: [time, n_mels] numpy array
        """
        # Normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Compute mel spectrogram using librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=20,
            fmax=8000
        )
        
        # Convert to log scale
        log_mel = np.log(mel_spec + 1e-9)
        
        # Normalize
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-9)
        
        # Transpose to [time, n_mels]
        log_mel = log_mel.T
        
        return log_mel.astype(np.float32)


class SimpleVAD:
    """
    Simple Voice Activity Detection using energy threshold.
    No external dependencies (no webrtcvad).
    """
    
    def __init__(self, energy_threshold: float = 0.01, frame_size: int = 480):
        self.energy_threshold = energy_threshold
        self.frame_size = frame_size
    
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Check if frame contains speech based on energy"""
        energy = np.sqrt(np.mean(audio_frame ** 2))
        return energy > self.energy_threshold


class LightweightSTT:
    """
    Lightweight Speech-to-Text using ONNX Runtime.
    
    Memory usage: ~150-200 MB
    """
    
    def __init__(self, model_path: str, use_gpu: bool = False):
        """
        Args:
            model_path: Path to ONNX model file
            use_gpu: Whether to use GPU (if available)
        """
        # Configure ONNX Runtime for low memory
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2  # Limit threads for low-end devices
        sess_options.inter_op_num_threads = 1
        
        # Enable memory optimization
        sess_options.enable_cpu_mem_arena = False  # Saves memory
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        # Select provider
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        print(f"Loading ONNX model from {model_path}...")
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        print(f"Model loaded (provider: {self.session.get_providers()[0]})")
        
        # Audio processor
        self.audio_processor = LightweightAudioProcessor()
        
        # VAD
        self.vad = SimpleVAD()
        
        # Audio buffer
        self.audio_buffer = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_recording = False
        self.silence_threshold = 30
        self.min_speech_frames = 10
        
        # Audio queue
        self.audio_queue = queue.Queue()
        self.is_running = False
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: numpy array of audio samples
        Returns:
            Transcribed text
        """
        # Extract features
        features = self.audio_processor.extract_features(audio)
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        
        # Run inference
        log_probs = self.session.run(
            None,
            {'audio_features': features}
        )[0]
        
        # Decode (greedy)
        predictions = np.argmax(log_probs, axis=-1)[0]  # [time]
        
        # CTC decode: collapse repeated characters and remove blanks
        chars = []
        prev_idx = None
        for idx in predictions:
            if idx != BLANK_IDX and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    char = IDX_TO_CHAR[idx]
                    if char not in ['<blank>', '<sos>', '<eos>', '<pad>', '<unk>']:
                        chars.append(char)
            prev_idx = idx
        
        return ''.join(chars)
    
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Add audio chunk and return transcription if utterance is complete.
        """
        is_speech = self.vad.is_speech(audio_chunk)
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.is_recording and self.speech_frames >= 3:
                self.is_recording = True
            
            if self.is_recording:
                self.audio_buffer.append(audio_chunk.copy())
        else:
            if self.is_recording:
                self.audio_buffer.append(audio_chunk.copy())
                self.silence_frames += 1
                
                if self.silence_frames >= self.silence_threshold:
                    if self.speech_frames >= self.min_speech_frames:
                        # Transcribe accumulated audio
                        audio = np.concatenate(self.audio_buffer)
                        self._reset_buffer()
                        return self.transcribe(audio)
                    else:
                        self._reset_buffer()
        
        return None
    
    def _reset_buffer(self):
        """Reset audio buffer"""
        self.audio_buffer = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_recording = False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy().flatten())
    
    def process_audio(self):
        """Process audio from queue"""
        while self.is_running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                text = self.add_audio_chunk(audio_chunk)
                
                if text and text.strip():
                    print(f"\n[تم التعرف]: {text}")
                    print(">>> ", end="", flush=True)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error: {e}")
    
    def start_realtime(self):
        """Start real-time transcription"""
        print("\n" + "="*50)
        print("نظام التعرف على الكلام العربي (خفيف)")
        print("Arabic STT - Lightweight Mode")
        print("="*50)
        print("\nListening... (Press Ctrl+C to stop)")
        print(">>> ", end="", flush=True)
        
        self.is_running = True
        
        # Start audio stream
        chunk_size = int(SAMPLE_RATE * 0.03)  # 30ms chunks
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=chunk_size,
            callback=self.audio_callback
        )
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.start()
        
        try:
            stream.start()
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.is_running = False
            stream.stop()
            stream.close()
            process_thread.join()


def transcribe_file(model_path: str, audio_path: str) -> str:
    """
    Transcribe an audio file.
    
    Args:
        model_path: Path to ONNX model
        audio_path: Path to audio file
    Returns:
        Transcribed text
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Create STT
    stt = LightweightSTT(model_path)
    
    # Transcribe
    text = stt.transcribe(audio)
    
    return text


def main():
    parser = argparse.ArgumentParser(description='Lightweight Arabic STT (ONNX)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--audio-file', type=str,
                       help='Path to audio file (if not provided, uses microphone)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    parser.add_argument('--list-devices', action='store_true',
                       help='List audio input devices')
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("\nAvailable audio devices:")
        print("-" * 40)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']}")
        return
    
    if args.audio_file:
        # Transcribe file
        print(f"Transcribing: {args.audio_file}")
        text = transcribe_file(args.model, args.audio_file)
        print(f"\nTranscription:\n{text}")
    else:
        # Real-time
        stt = LightweightSTT(args.model, use_gpu=args.gpu)
        stt.start_realtime()


if __name__ == "__main__":
    main()
