"""
Training script for Arabic Speech-to-Text Model on Quran Datasets

Supports various Quran recitation datasets from HuggingFace:
- arbml/quran_speech
- tarteel-ai datasets
- Custom Quran datasets

Usage:
    python train_quran.py --dataset arbml/quran_speech --batch-size 16 --epochs 100
"""
import os
import sys
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from model import ArabicSTTModel, count_parameters
from dataset import QuranDataset, create_quran_dataloaders, collate_fn
from train import Trainer, compute_wer, compute_cer


def main():
    parser = argparse.ArgumentParser(description='Train Arabic STT on Quran Dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='arbml/quran_speech',
                       help='HuggingFace dataset name (e.g., arbml/quran_speech)')
    parser.add_argument('--subset', type=str, default=None,
                       help='Dataset subset/config name')
    parser.add_argument('--train-split', type=str, default='train',
                       help='Training split name')
    parser.add_argument('--val-split', type=str, default='test',
                       help='Validation split name')
    parser.add_argument('--audio-column', type=str, default='audio',
                       help='Column name for audio data')
    parser.add_argument('--text-column', type=str, default='text',
                       help='Column name for transcription')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Create directories
    Config.create_dirs()
    
    # Print dataset info
    print("="*50)
    print("Quran Speech-to-Text Training")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Subset: {args.subset or 'default'}")
    print(f"Train split: {args.train_split}")
    print(f"Val split: {args.val_split}")
    print(f"Device: {args.device}")
    print("="*50)
    
    # Create dataloaders
    print("\nLoading Quran dataset...")
    train_loader, val_loader = create_quran_dataloaders(
        dataset_name=args.dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        audio_column=args.audio_column,
        text_column=args.text_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset
    )
    
    # Create model
    print("\nCreating model...")
    model = ArabicSTTModel()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
