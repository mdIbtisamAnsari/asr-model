"""
Real-time Arabic Speech-to-Text Inference

This script captures audio from your microphone and transcribes it in real-time
using the trained Arabic STT model.
"""
import argparse
import sys
import queue
import threading
import time
import numpy as np
import torch

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(1)

try:
    import webrtcvad
except ImportError:
    print("Please install webrtcvad: pip install webrtcvad")
    webrtcvad = None

from config import Config
from model import ArabicSTTModel
from dataset import AudioProcessor


class VoiceActivityDetector:
    """
    Voice Activity Detection using WebRTC VAD.
    Detects when someone is speaking vs silence.
    """
    
    def __init__(self, sample_rate: int = Config.SAMPLE_RATE, mode: int = Config.VAD_MODE):
        if webrtcvad is None:
            self.vad = None
            return
        
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        # VAD requires 10, 20, or 30 ms frames
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
    
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Check if audio frame contains speech"""
        if self.vad is None:
            return True  # Assume speech if VAD not available
        
        # Convert to 16-bit PCM
        audio_bytes = (audio_frame * 32768).astype(np.int16).tobytes()
        
        try:
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except:
            return True  # Assume speech on error


class AudioBuffer:
    """
    Manages audio buffering for real-time processing.
    Accumulates audio until enough speech is detected, then returns for processing.
    """
    
    def __init__(
        self,
        sample_rate: int = Config.SAMPLE_RATE,
        silence_threshold_frames: int = Config.SILENCE_THRESHOLD,
        min_speech_frames: int = 10
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold_frames
        self.min_speech_frames = min_speech_frames
        
        self.vad = VoiceActivityDetector(sample_rate)
        self.buffer = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_recording = False
    
    def add_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Add audio chunk to buffer.
        Returns accumulated audio when speech ends, or None if still recording.
        """
        # Check for speech
        is_speech = self.vad.is_speech(audio_chunk)
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            
            if not self.is_recording and self.speech_frames >= 3:
                # Start recording after some speech frames
                self.is_recording = True
            
            if self.is_recording:
                self.buffer.append(audio_chunk.copy())
        else:
            if self.is_recording:
                self.buffer.append(audio_chunk.copy())
                self.silence_frames += 1
                
                # Check if we have enough silence to end recording
                if self.silence_frames >= self.silence_threshold:
                    if self.speech_frames >= self.min_speech_frames:
                        # Return accumulated audio
                        audio = np.concatenate(self.buffer)
                        self.reset()
                        return audio
                    else:
                        # Not enough speech, reset
                        self.reset()
        
        return None
    
    def reset(self):
        """Reset buffer state"""
        self.buffer = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_recording = False


class RealTimeSTT:
    """
    Real-time Speech-to-Text system for Arabic.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        sample_rate: int = Config.SAMPLE_RATE
    ):
        self.device = device
        self.sample_rate = sample_rate
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = ArabicSTTModel.load_checkpoint(model_path, device)
        self.model.eval()
        print(f"Model loaded on {device}")
        
        # Audio processor
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # Audio buffer
        self.audio_buffer = AudioBuffer(sample_rate=sample_rate)
        
        # Audio queue for thread-safe communication
        self.audio_queue = queue.Queue()
        
        # Control flags
        self.is_running = False
        self.stream = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Put audio data in queue
        self.audio_queue.put(indata.copy().flatten())
    
    @torch.no_grad()
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio array to text.
        
        Args:
            audio: numpy array of audio samples
        Returns:
            Transcribed text
        """
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        # Extract features
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        features = self.audio_processor.extract_features(waveform)
        
        # Add batch dimension
        features = features.unsqueeze(0).to(self.device)
        
        # Decode
        transcription = self.model.decode_greedy(features)
        
        return transcription[0]
    
    def process_audio(self):
        """Process audio from queue"""
        while self.is_running:
            try:
                # Get audio chunk (blocking with timeout)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer and check if we have a complete utterance
                complete_audio = self.audio_buffer.add_audio(audio_chunk)
                
                if complete_audio is not None:
                    # Transcribe
                    text = self.transcribe(complete_audio)
                    if text.strip():
                        print(f"\n[تم التعرف]: {text}")
                        print(">>> ", end="", flush=True)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def start(self):
        """Start real-time transcription"""
        print("\n" + "="*50)
        print("نظام التعرف على الكلام العربي")
        print("Arabic Speech Recognition System")
        print("="*50)
        print("\nListening... (Press Ctrl+C to stop)")
        print(">>> ", end="", flush=True)
        
        self.is_running = True
        
        # Start audio stream
        chunk_size = int(self.sample_rate * 0.03)  # 30ms chunks
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=chunk_size,
            callback=self.audio_callback
        )
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.start()
        
        try:
            self.stream.start()
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.stop()
            process_thread.join()
    
    def stop(self):
        """Stop real-time transcription"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()


class FileTranscriber:
    """
    Transcribe audio files (not real-time).
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = ArabicSTTModel.load_checkpoint(model_path, device)
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor()
    
    @torch.no_grad()
    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
        Returns:
            Transcribed text
        """
        # Process audio
        features = self.audio_processor.process_audio(audio_path)
        
        # Add batch dimension
        features = features.unsqueeze(0).to(self.device)
        
        # Decode
        transcription = self.model.decode_greedy(features)
        
        return transcription[0]


def list_audio_devices():
    """List available audio input devices"""
    print("\nAvailable audio devices:")
    print("-" * 40)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{i}] {device['name']}")
            print(f"      Sample rate: {device['default_samplerate']} Hz")
            print(f"      Input channels: {device['max_input_channels']}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description='Arabic Speech-to-Text Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--audio-file', type=str,
                       help='Path to audio file to transcribe (if not provided, uses microphone)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices')
    parser.add_argument('--input-device', type=int,
                       help='Audio input device index')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    if args.input_device is not None:
        sd.default.device[0] = args.input_device
    
    if args.audio_file:
        # Transcribe file
        transcriber = FileTranscriber(args.model, args.device)
        print(f"\nTranscribing: {args.audio_file}")
        text = transcriber.transcribe_file(args.audio_file)
        print(f"\nTranscription:\n{text}")
    else:
        # Real-time transcription
        stt = RealTimeSTT(args.model, args.device)
        stt.start()


if __name__ == "__main__":
    main()
