"""
Training script for Arabic Speech-to-Text Model
"""
import os
import sys
import argparse
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from model import ArabicSTTModel, count_parameters
from dataset import create_dataloaders, AudioProcessor, TextProcessor, ArabicSpeechDataset, collate_fn
from torch.utils.data import DataLoader


def compute_wer(predictions: list, references: list) -> float:
    """
    Compute Word Error Rate (WER)
    """
    total_words = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        
        # Use dynamic programming for edit distance
        m, n = len(ref_words), len(pred_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == pred_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        total_errors += dp[m][n]
        total_words += m
    
    return total_errors / max(total_words, 1)


def compute_cer(predictions: list, references: list) -> float:
    """
    Compute Character Error Rate (CER)
    """
    total_chars = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        pred_chars = list(pred.replace(' ', ''))
        ref_chars = list(ref.replace(' ', ''))
        
        m, n = len(ref_chars), len(pred_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == pred_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        total_errors += dp[m][n]
        total_chars += m
    
    return total_errors / max(total_chars, 1)


class Trainer:
    """
    Training manager for the Arabic STT model.
    """
    
    def __init__(
        self,
        model: ArabicSTTModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY,
        gradient_clip: float = Config.GRADIENT_CLIP,
        checkpoint_dir: str = Config.CHECKPOINT_DIR,
        log_dir: str = Config.LOG_DIR
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # CTC Loss
        self.criterion = nn.CTCLoss(blank=Config.BLANK_IDX, zero_infinity=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * Config.EPOCHS
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # TensorBoard writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(log_dir, f'run_{timestamp}'))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in progress_bar:
            # Move data to device
            features = batch['features'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            log_probs, output_lengths = self.model(features, feature_lengths)
            
            # Compute CTC loss
            # log_probs: [T, N, C], targets: [N, S]
            loss = self.criterion(
                log_probs,
                targets,
                output_lengths,
                target_lengths
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Log
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # TensorBoard logging
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.global_step)
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # Move data to device
            features = batch['features'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            transcripts = batch['transcripts']
            
            # Forward pass
            log_probs, output_lengths = self.model(features, feature_lengths)
            
            # Compute loss
            loss = self.criterion(
                log_probs,
                targets,
                output_lengths,
                target_lengths
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Decode predictions
            predictions = self.model.decode_greedy(features)
            all_predictions.extend(predictions)
            all_references.extend(transcripts)
        
        # Compute metrics
        avg_loss = total_loss / max(num_batches, 1)
        wer = compute_wer(all_predictions, all_references)
        cer = compute_cer(all_predictions, all_references)
        
        return {
            'loss': avg_loss,
            'wer': wer,
            'cer': cer,
            'predictions': all_predictions[:5],  # Sample predictions
            'references': all_references[:5]
        }
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint"""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pt'
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int = Config.EPOCHS, resume_from: str = None):
        """
        Main training loop
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 50)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Val Loss: {val_metrics['loss']:.4f}, WER: {val_metrics['wer']:.2%}, CER: {val_metrics['cer']:.2%}")
            
            # Log sample predictions
            print("\nSample predictions:")
            for pred, ref in zip(val_metrics['predictions'], val_metrics['references']):
                print(f"  REF: {ref}")
                print(f"  PRD: {pred}")
                print()
            
            # TensorBoard
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], self.global_step)
            self.writer.add_scalar('Val/WER', val_metrics['wer'], self.global_step)
            self.writer.add_scalar('Val/CER', val_metrics['cer'], self.global_step)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(is_best=is_best)
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            print("-" * 50)
        
        self.writer.close()
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train Arabic STT Model')
    parser.add_argument('--train-manifest', type=str, help='Path to training manifest')
    parser.add_argument('--val-manifest', type=str, help='Path to validation manifest')
    parser.add_argument('--use-common-voice', action='store_true', default=True,
                       help='Use Mozilla Common Voice Arabic dataset')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Create directories
    Config.create_dirs()
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        use_common_voice=args.use_common_voice,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
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
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
