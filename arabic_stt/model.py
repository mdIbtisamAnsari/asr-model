"""
Arabic Speech-to-Text Model Architecture

Uses a Conformer-style encoder with CTC loss for end-to-end ASR.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ConvSubsampling(nn.Module):
    """
    Convolutional subsampling layer that reduces sequence length by factor of 4.
    This helps reduce computation and memory for long audio sequences.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Calculate output dimension after conv layers
        self.out_proj = nn.Linear(32 * (in_channels // 4), out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, time, features]
        Returns:
            [batch, time//4, out_channels]
        """
        x = x.unsqueeze(1)  # [batch, 1, time, features]
        x = self.conv(x)    # [batch, 32, time//4, features//4]
        batch, channels, time, features = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch, time//4, 32, features//4]
        x = x.view(batch, time, channels * features)  # [batch, time//4, 32*features//4]
        x = self.out_proj(x)
        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module with residual connection"""
    
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = F.swish(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + 0.5 * x


class MultiHeadSelfAttentionModule(nn.Module):
    """Multi-head self-attention with relative positional encoding"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.dropout(x)
        return residual + x


class ConvolutionModule(nn.Module):
    """Conformer convolution module for local feature extraction"""
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise conv -> GLU -> Depthwise conv -> BatchNorm -> Swish -> Pointwise conv
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, 
            padding=(kernel_size - 1) // 2, 
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, time, d_model]
        """
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [batch, d_model, time]
        
        # Pointwise conv with GLU activation
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # [batch, d_model, time]
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.swish(x)
        
        # Pointwise conv out
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # [batch, time, d_model]
        return residual + x


class ConformerBlock(nn.Module):
    """
    Conformer block combining self-attention and convolution.
    Structure: FFN -> MHSA -> Conv -> FFN
    """
    
    def __init__(self, d_model: int, num_heads: int, conv_kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_model * 4, dropout)
        self.attention = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, d_model * 4, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.attention(x, mask)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.layer_norm(x)
        return x


class ArabicSTTEncoder(nn.Module):
    """
    Conformer-based encoder for Arabic speech recognition.
    """
    
    def __init__(
        self,
        input_dim: int = Config.N_MELS,
        d_model: int = Config.ENCODER_DIM,
        num_layers: int = Config.ENCODER_LAYERS,
        num_heads: int = Config.NUM_HEADS,
        dropout: float = Config.DROPOUT
    ):
        super().__init__()
        
        # Subsampling layer (reduces time steps by 4x)
        self.subsampling = ConvSubsampling(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, time, features] - mel spectrogram
            mask: [batch, time] - padding mask
        Returns:
            [batch, time//4, d_model]
        """
        x = self.subsampling(x)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class ArabicSTTModel(nn.Module):
    """
    End-to-end Arabic Speech-to-Text model using CTC loss.
    """
    
    def __init__(
        self,
        input_dim: int = Config.N_MELS,
        d_model: int = Config.ENCODER_DIM,
        num_encoder_layers: int = Config.ENCODER_LAYERS,
        num_heads: int = Config.NUM_HEADS,
        vocab_size: int = Config.VOCAB_SIZE,
        dropout: float = Config.DROPOUT
    ):
        super().__init__()
        
        self.encoder = ArabicSTTEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # CTC output layer
        self.ctc_linear = nn.Linear(d_model, vocab_size)
        
        # Store config
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        """
        Args:
            x: [batch, time, features] - mel spectrogram
            x_lengths: [batch] - lengths of each sequence
        Returns:
            log_probs: [time, batch, vocab_size] - log probabilities for CTC
            output_lengths: [batch] - output lengths after subsampling
        """
        # Create mask for padding if lengths provided
        mask = None
        if x_lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) >= x_lengths.unsqueeze(1)
        
        # Encode
        encoder_out = self.encoder(x, mask)  # [batch, time//4, d_model]
        
        # CTC output
        logits = self.ctc_linear(encoder_out)  # [batch, time//4, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose for CTC loss (expects [time, batch, vocab])
        log_probs = log_probs.transpose(0, 1)
        
        # Compute output lengths (subsampled by factor of 4)
        output_lengths = None
        if x_lengths is not None:
            output_lengths = (x_lengths // 4).long()
        
        return log_probs, output_lengths
    
    def decode_greedy(self, x: torch.Tensor) -> list:
        """
        Greedy decoding for inference.
        
        Args:
            x: [batch, time, features]
        Returns:
            List of decoded strings
        """
        self.eval()
        with torch.no_grad():
            log_probs, _ = self.forward(x)
            predictions = log_probs.argmax(dim=-1)  # [time, batch]
            predictions = predictions.transpose(0, 1)  # [batch, time]
        
        idx_to_char = Config.idx_to_char()
        decoded = []
        
        for batch_idx in range(predictions.size(0)):
            sequence = predictions[batch_idx].tolist()
            # CTC decode: collapse repeated characters and remove blanks
            chars = []
            prev_char = None
            for idx in sequence:
                if idx != Config.BLANK_IDX and idx != prev_char:
                    if idx < len(idx_to_char):
                        char = idx_to_char[idx]
                        if char not in ['<blank>', '<sos>', '<eos>', '<pad>', '<unk>']:
                            chars.append(char)
                prev_char = idx
            decoded.append(''.join(chars))
        
        return decoded
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = 'cpu'):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(**checkpoint.get('model_config', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def save_checkpoint(self, checkpoint_path: str, optimizer=None, epoch=None, loss=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'd_model': self.d_model,
                'vocab_size': self.vocab_size,
            },
            'epoch': epoch,
            'loss': loss,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, checkpoint_path)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = ArabicSTTModel()
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    time_steps = 200
    features = Config.N_MELS
    
    x = torch.randn(batch_size, time_steps, features)
    x_lengths = torch.tensor([200, 150])
    
    log_probs, output_lengths = model(x, x_lengths)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {log_probs.shape}")
    print(f"Output lengths: {output_lengths}")
