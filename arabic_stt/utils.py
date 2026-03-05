"""
Utility functions for Arabic Speech-to-Text

Includes data augmentation, spectrogram augmentation, and helper functions.
"""
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random
from typing import Tuple, Optional


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    https://arxiv.org/abs/1904.08779
    
    Applies frequency and time masking to mel spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            x: [time, freq] or [batch, time, freq]
        Returns:
            Augmented spectrogram
        """
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0).transpose(1, 2)  # [1, freq, time]
            squeeze = True
        else:
            x = x.transpose(1, 2)  # [batch, freq, time]
            squeeze = False
        
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            x = self.freq_mask(x)
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            x = self.time_mask(x)
        
        # Restore shape
        x = x.transpose(1, 2)  # [batch, time, freq]
        if squeeze:
            x = x.squeeze(0)
        
        return x


class AudioAugmentation:
    """
    Audio-level augmentations applied to waveforms.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        speed_perturb_range: Tuple[float, float] = (0.9, 1.1),
        noise_snr_range: Tuple[float, float] = (10, 30),
        volume_range: Tuple[float, float] = (0.8, 1.2)
    ):
        self.sample_rate = sample_rate
        self.speed_perturb_range = speed_perturb_range
        self.noise_snr_range = noise_snr_range
        self.volume_range = volume_range
    
    def speed_perturbation(self, waveform: torch.Tensor, rate: float = None) -> torch.Tensor:
        """
        Apply speed perturbation to waveform.
        
        Args:
            waveform: [1, time] or [time]
            rate: Speed factor (0.9 = slower, 1.1 = faster)
        """
        if rate is None:
            rate = random.uniform(*self.speed_perturb_range)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample to change speed
        effects = [['speed', str(rate)], ['rate', str(self.sample_rate)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        
        return augmented
    
    def add_noise(self, waveform: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """
        Add white noise to waveform.
        
        Args:
            waveform: [1, time] or [time]
            snr_db: Signal-to-noise ratio in dB
        """
        if snr_db is None:
            snr_db = random.uniform(*self.noise_snr_range)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Generate noise
        noise = torch.randn_like(waveform)
        
        # Calculate noise level
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        
        # Add noise
        augmented = waveform + noise_scale * noise
        
        return augmented
    
    def volume_perturbation(self, waveform: torch.Tensor, gain: float = None) -> torch.Tensor:
        """
        Apply volume perturbation.
        
        Args:
            waveform: Audio tensor
            gain: Volume multiplier
        """
        if gain is None:
            gain = random.uniform(*self.volume_range)
        
        return waveform * gain
    
    def augment(
        self,
        waveform: torch.Tensor,
        apply_speed: bool = True,
        apply_noise: bool = True,
        apply_volume: bool = True,
        prob: float = 0.5
    ) -> torch.Tensor:
        """
        Apply random augmentations with given probability.
        """
        if apply_speed and random.random() < prob:
            waveform = self.speed_perturbation(waveform)
        
        if apply_noise and random.random() < prob:
            waveform = self.add_noise(waveform)
        
        if apply_volume and random.random() < prob:
            waveform = self.volume_perturbation(waveform)
        
        return waveform


def beam_search_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    blank_idx: int = 0
) -> list:
    """
    Beam search decoding for CTC output.
    
    Args:
        log_probs: [time, batch, vocab_size] log probabilities
        beam_width: Number of beams to keep
        blank_idx: Index of blank token
    
    Returns:
        List of decoded sequences (one per batch)
    """
    T, B, V = log_probs.shape
    results = []
    
    for b in range(B):
        # Beam: (log_prob, sequence, last_char)
        beams = [(0.0, [], None)]
        
        for t in range(T):
            new_beams = []
            
            for log_prob, seq, last_char in beams:
                for v in range(V):
                    new_log_prob = log_prob + log_probs[t, b, v].item()
                    
                    if v == blank_idx:
                        # Blank: keep sequence as is
                        new_beams.append((new_log_prob, seq, None))
                    elif v == last_char:
                        # Same char: keep sequence (CTC collapse)
                        new_beams.append((new_log_prob, seq, v))
                    else:
                        # New char: extend sequence
                        new_beams.append((new_log_prob, seq + [v], v))
            
            # Keep top beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            
            # Merge beams with same sequence
            seen = set()
            merged_beams = []
            for beam in new_beams:
                key = tuple(beam[1])
                if key not in seen:
                    seen.add(key)
                    merged_beams.append(beam)
                    if len(merged_beams) >= beam_width:
                        break
            
            beams = merged_beams
        
        # Get best beam
        best_beam = max(beams, key=lambda x: x[0])
        results.append(best_beam[1])
    
    return results


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class LabelSmoothingCTCLoss(nn.Module):
    """
    CTC Loss with label smoothing for regularization.
    """
    
    def __init__(self, blank: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        # Standard CTC loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Add label smoothing penalty (KL divergence from uniform)
        vocab_size = log_probs.size(-1)
        smooth_loss = -log_probs.mean()
        
        # Combine
        loss = (1 - self.smoothing) * ctc_loss + self.smoothing * smooth_loss
        
        return loss


def calculate_model_size(model: nn.Module) -> dict:
    """
    Calculate model size in parameters and MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size in MB (assuming float32)
    size_mb = total_params * 4 / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'size_mb': size_mb
    }


def print_model_summary(model: nn.Module):
    """Print a summary of model architecture and size."""
    print("\nModel Summary")
    print("=" * 50)
    
    size_info = calculate_model_size(model)
    print(f"Total parameters: {size_info['total_params']:,}")
    print(f"Trainable parameters: {size_info['trainable_params']:,}")
    print(f"Model size: {size_info['size_mb']:.2f} MB")
    
    print("\nLayer breakdown:")
    print("-" * 50)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} params")
    print("=" * 50)
