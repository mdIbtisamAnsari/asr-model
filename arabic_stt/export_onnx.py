"""
Export Arabic STT Model to ONNX format for lightweight inference.

ONNX reduces RAM usage from ~600 MB (PyTorch) to ~150-200 MB.
"""
import argparse
import os
import torch
import torch.nn as nn
import numpy as np

from config import Config
from model import ArabicSTTModel


class ArabicSTTForExport(nn.Module):
    """
    Wrapper model for ONNX export.
    Removes CTC-specific operations that don't export well.
    """
    
    def __init__(self, model: ArabicSTTModel):
        super().__init__()
        self.encoder = model.encoder
        self.ctc_linear = model.ctc_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, time, features] - mel spectrogram
        Returns:
            log_probs: [batch, time//4, vocab_size]
        """
        encoder_out = self.encoder(x, mask=None)
        logits = self.ctc_linear(encoder_out)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 14,
    simplify: bool = True
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt)
        output_path: Output path for ONNX model (.onnx)
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model
    """
    print(f"Loading model from {checkpoint_path}...")
    model = ArabicSTTModel.load_checkpoint(checkpoint_path, device='cpu')
    model.eval()
    
    # Wrap for export
    export_model = ArabicSTTForExport(model)
    export_model.eval()
    
    # Create dummy input
    # Typical input: 3 seconds of audio = ~300 frames after mel + subsampling
    batch_size = 1
    time_steps = 300  # ~3 seconds
    features = Config.N_MELS
    
    dummy_input = torch.randn(batch_size, time_steps, features)
    
    # Export to ONNX
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_features'],
        output_names=['log_probs'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'time_steps'},
            'log_probs': {0: 'batch_size', 1: 'output_time'}
        }
    )
    
    print(f"Exported to {output_path}")
    
    # Verify the model
    print("Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")
    
    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print("Simplifying ONNX model...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, output_path)
                print("Model simplified successfully!")
            else:
                print("Simplification check failed, keeping original model")
        except ImportError:
            print("onnx-simplifier not installed, skipping simplification")
            print("Install with: pip install onnx-simplifier")
    
    # Print model size
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nONNX model size: {model_size:.2f} MB")
    
    # Test inference
    print("\nTesting ONNX inference...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(output_path)
    test_input = np.random.randn(1, 100, features).astype(np.float32)
    outputs = session.run(None, {'audio_features': test_input})
    print(f"Test output shape: {outputs[0].shape}")
    print("ONNX export complete!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export Arabic STT to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='arabic_stt.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Skip ONNX model simplification')
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )


if __name__ == "__main__":
    main()
