"""
Dataset handling and audio preprocessing for Arabic STT
"""
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import librosa
from config import Config


class AudioProcessor:
    """
    Handles all audio preprocessing: loading, resampling, and feature extraction.
    """
    
    def __init__(
        self,
        sample_rate: int = Config.SAMPLE_RATE,
        n_mels: int = Config.N_MELS,
        n_fft: int = Config.N_FFT,
        hop_length: int = Config.HOP_LENGTH,
        win_length: int = Config.WIN_LENGTH
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=20,
            f_max=8000
        )
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and resample audio file.
        
        Args:
            audio_path: Path to audio file
        Returns:
            waveform: [1, time] tensor
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel spectrogram features.
        
        Args:
            waveform: [1, time] tensor
        Returns:
            features: [time, n_mels] tensor
        """
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-9)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
        
        # Transpose to [time, n_mels]
        log_mel = log_mel.squeeze(0).transpose(0, 1)
        
        return log_mel
    
    def process_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio and extract features in one step"""
        waveform = self.load_audio(audio_path)
        features = self.extract_features(waveform)
        return features
    
    def process_audio_stream(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """
        Process audio chunk from real-time stream.
        
        Args:
            audio_chunk: numpy array of audio samples
        Returns:
            features: [time, n_mels] tensor
        """
        # Convert to tensor
        waveform = torch.from_numpy(audio_chunk).float().unsqueeze(0)
        
        # Normalize audio
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-9)
        
        # Extract features
        features = self.extract_features(waveform)
        
        return features


class TextProcessor:
    """
    Handles Arabic text preprocessing and encoding.
    """
    
    def __init__(self):
        self.char_to_idx = Config.char_to_idx()
        self.idx_to_char = Config.idx_to_char()
        self.vocab_size = Config.VOCAB_SIZE
    
    def normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text:
        - Remove extra whitespace
        - Normalize some character variants
        """
        import re
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize alef variants to bare alef (optional, can be adjusted)
        # text = re.sub(r'[آأإ]', 'ا', text)
        
        # Remove tatweel (kashida)
        text = text.replace('ـ', '')
        
        return text
    
    def encode(self, text: str) -> List[int]:
        """
        Encode Arabic text to sequence of indices.
        
        Args:
            text: Arabic text string
        Returns:
            List of token indices
        """
        text = self.normalize_arabic(text)
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(Config.UNK_IDX)
        return encoded
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode sequence of indices to Arabic text.
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens
        Returns:
            Decoded text string
        """
        chars = []
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if remove_special and char in ['<blank>', '<sos>', '<eos>', '<pad>', '<unk>']:
                    continue
                chars.append(char)
        return ''.join(chars)


class ArabicSpeechDataset(Dataset):
    """
    Dataset for Arabic speech data.
    
    Expected data format:
    - audio_files: List of paths to audio files
    - transcripts: List of corresponding Arabic transcriptions
    
    Or use a manifest file (TSV/CSV) with columns: audio_path, transcript
    """
    
    def __init__(
        self,
        manifest_path: Optional[str] = None,
        audio_files: Optional[List[str]] = None,
        transcripts: Optional[List[str]] = None,
        audio_processor: Optional[AudioProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        max_audio_len: int = 3000,  # Max frames
        max_text_len: int = 500     # Max characters
    ):
        self.audio_processor = audio_processor or AudioProcessor()
        self.text_processor = text_processor or TextProcessor()
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        if manifest_path:
            self.audio_files, self.transcripts = self._load_manifest(manifest_path)
        else:
            self.audio_files = audio_files or []
            self.transcripts = transcripts or []
        
        assert len(self.audio_files) == len(self.transcripts), \
            "Number of audio files must match number of transcripts"
    
    def _load_manifest(self, manifest_path: str) -> Tuple[List[str], List[str]]:
        """Load data from manifest file"""
        audio_files = []
        transcripts = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    audio_files.append(parts[0])
                    transcripts.append(parts[1])
        
        return audio_files, transcripts
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict:
        audio_path = self.audio_files[idx]
        transcript = self.transcripts[idx]
        
        # Process audio
        features = self.audio_processor.process_audio(audio_path)
        
        # Truncate if too long
        if features.size(0) > self.max_audio_len:
            features = features[:self.max_audio_len]
        
        # Process text
        encoded_text = self.text_processor.encode(transcript)
        
        # Truncate if too long
        if len(encoded_text) > self.max_text_len:
            encoded_text = encoded_text[:self.max_text_len]
        
        return {
            'features': features,
            'feature_length': features.size(0),
            'targets': torch.tensor(encoded_text, dtype=torch.long),
            'target_length': len(encoded_text),
            'transcript': transcript
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    Pads features and targets to max length in batch.
    """
    # Sort by feature length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['feature_length'], reverse=True)
    
    # Get max lengths
    max_feature_len = max(item['feature_length'] for item in batch)
    max_target_len = max(item['target_length'] for item in batch)
    
    # Prepare batch tensors
    batch_size = len(batch)
    n_mels = batch[0]['features'].size(1)
    
    features = torch.zeros(batch_size, max_feature_len, n_mels)
    feature_lengths = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.full((batch_size, max_target_len), Config.PAD_IDX, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    transcripts = []
    
    for i, item in enumerate(batch):
        feat_len = item['feature_length']
        tgt_len = item['target_length']
        
        features[i, :feat_len] = item['features']
        feature_lengths[i] = feat_len
        targets[i, :tgt_len] = item['targets']
        target_lengths[i] = tgt_len
        transcripts.append(item['transcript'])
    
    return {
        'features': features,
        'feature_lengths': feature_lengths,
        'targets': targets,
        'target_lengths': target_lengths,
        'transcripts': transcripts
    }


class CommonVoiceArabicDataset(Dataset):
    """
    Dataset loader for Mozilla Common Voice Arabic dataset.
    
    This uses the HuggingFace datasets library to load Common Voice.
    """
    
    def __init__(
        self,
        split: str = 'train',
        audio_processor: Optional[AudioProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        max_audio_len: int = 3000,
        max_text_len: int = 500
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install 'datasets' library: pip install datasets")
        
        self.audio_processor = audio_processor or AudioProcessor()
        self.text_processor = text_processor or TextProcessor()
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        # Load Common Voice Arabic dataset
        print(f"Loading Common Voice Arabic ({split}) dataset...")
        self.dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "ar",
            split=split,
            trust_remote_code=True
        )
        print(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        
        # Get audio array and sample rate
        audio_array = item['audio']['array']
        sample_rate = item['audio']['sampling_rate']
        
        # Convert to tensor and resample if necessary
        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
        if sample_rate != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, Config.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Extract features
        features = self.audio_processor.extract_features(waveform)
        
        # Truncate if too long
        if features.size(0) > self.max_audio_len:
            features = features[:self.max_audio_len]
        
        # Get transcript
        transcript = item['sentence']
        encoded_text = self.text_processor.encode(transcript)
        
        # Truncate if too long
        if len(encoded_text) > self.max_text_len:
            encoded_text = encoded_text[:self.max_text_len]
        
        return {
            'features': features,
            'feature_length': features.size(0),
            'targets': torch.tensor(encoded_text, dtype=torch.long),
            'target_length': len(encoded_text),
            'transcript': transcript
        }


def create_dataloaders(
    train_manifest: str = None,
    val_manifest: str = None,
    use_common_voice: bool = True,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_manifest: Path to training manifest file
        val_manifest: Path to validation manifest file
        use_common_voice: If True, use Common Voice dataset instead of manifest
        batch_size: Batch size
        num_workers: Number of data loader workers
    
    Returns:
        train_loader, val_loader
    """
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    
    if use_common_voice:
        train_dataset = CommonVoiceArabicDataset(
            split='train',
            audio_processor=audio_processor,
            text_processor=text_processor
        )
        val_dataset = CommonVoiceArabicDataset(
            split='validation',
            audio_processor=audio_processor,
            text_processor=text_processor
        )
    else:
        train_dataset = ArabicSpeechDataset(
            manifest_path=train_manifest,
            audio_processor=audio_processor,
            text_processor=text_processor
        )
        val_dataset = ArabicSpeechDataset(
            manifest_path=val_manifest,
            audio_processor=audio_processor,
            text_processor=text_processor
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test audio processing
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    
    # Test text encoding/decoding
    sample_text = "مرحبا بالعالم"
    encoded = text_processor.encode(sample_text)
    decoded = text_processor.decode(encoded)
    print(f"Original: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
