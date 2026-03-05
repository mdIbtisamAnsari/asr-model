"""
Script to create TSV manifest file from audio files and transcripts.

Usage:
    python create_manifest.py --audio-dir /path/to/audio --output data/train.tsv
"""
import os
import argparse
from pathlib import Path


def create_manifest_from_directory(
    audio_dir: str,
    transcript_dir: str = None,
    output_path: str = "data/manifest.tsv"
):
    """
    Create manifest from directory structure.
    
    Expected structure (Option 1 - text files with same name):
        audio_dir/
            file1.wav
            file1.txt  (contains Arabic transcript)
            file2.wav
            file2.txt
    
    Or (Option 2 - separate transcript directory):
        audio_dir/
            file1.wav
            file2.wav
        transcript_dir/
            file1.txt
            file2.txt
    """
    audio_dir = Path(audio_dir)
    transcript_dir = Path(transcript_dir) if transcript_dir else audio_dir
    
    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f'*{ext}'))
        audio_files.extend(audio_dir.glob(f'**/*{ext}'))  # Recursive
    
    audio_files = sorted(set(audio_files))
    print(f"Found {len(audio_files)} audio files")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Write manifest
    entries = []
    missing_transcripts = []
    
    for audio_path in audio_files:
        # Look for corresponding transcript
        transcript_path = transcript_dir / (audio_path.stem + '.txt')
        
        if transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            if transcript:
                entries.append((str(audio_path.absolute()), transcript))
        else:
            missing_transcripts.append(str(audio_path))
    
    # Write TSV
    with open(output_path, 'w', encoding='utf-8') as f:
        for audio_path, transcript in entries:
            f.write(f"{audio_path}\t{transcript}\n")
    
    print(f"Created manifest: {output_path}")
    print(f"Total entries: {len(entries)}")
    
    if missing_transcripts:
        print(f"\nWarning: {len(missing_transcripts)} files missing transcripts:")
        for path in missing_transcripts[:5]:
            print(f"  - {path}")
        if len(missing_transcripts) > 5:
            print(f"  ... and {len(missing_transcripts) - 5} more")
    
    return entries


def create_manifest_from_csv(
    csv_path: str,
    audio_column: str = "audio",
    transcript_column: str = "transcript",
    output_path: str = "data/manifest.tsv"
):
    """
    Create manifest from existing CSV/Excel file.
    """
    import pandas as pd
    
    # Read CSV
    if csv_path.endswith('.xlsx'):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Write TSV
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            audio_path = str(row[audio_column])
            transcript = str(row[transcript_column]).strip()
            if audio_path and transcript:
                f.write(f"{audio_path}\t{transcript}\n")
    
    print(f"Created manifest: {output_path}")


def create_empty_manifest_template(output_path: str = "data/manifest_template.tsv"):
    """
    Create an empty template with example entries.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    template = """# Arabic Speech-to-Text Manifest File
# Format: audio_path<TAB>transcript
# Lines starting with # are ignored
# 
# Example entries (replace with your data):
/path/to/audio1.wav	مرحبا بالعالم
/path/to/audio2.wav	كيف حالك اليوم
/path/to/audio3.wav	هذا نص تجريبي للتدريب
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"Created template: {output_path}")
    print("Edit this file and add your audio paths and transcripts.")


def split_manifest(
    manifest_path: str,
    train_ratio: float = 0.9,
    output_dir: str = "data"
):
    """
    Split manifest into train and validation sets.
    """
    import random
    
    # Read all entries
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                entries.append(line)
    
    # Shuffle
    random.shuffle(entries)
    
    # Split
    split_idx = int(len(entries) * train_ratio)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]
    
    # Write
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.tsv')
    val_path = os.path.join(output_dir, 'val.tsv')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_entries))
    
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_entries))
    
    print(f"Train set: {len(train_entries)} entries -> {train_path}")
    print(f"Val set: {len(val_entries)} entries -> {val_path}")


def main():
    parser = argparse.ArgumentParser(description='Create TSV manifest for Arabic STT')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # From directory
    dir_parser = subparsers.add_parser('from-dir', help='Create from audio directory')
    dir_parser.add_argument('--audio-dir', type=str, required=True,
                           help='Directory containing audio files')
    dir_parser.add_argument('--transcript-dir', type=str,
                           help='Directory containing transcript .txt files (optional)')
    dir_parser.add_argument('--output', type=str, default='data/manifest.tsv',
                           help='Output TSV path')
    
    # From CSV
    csv_parser = subparsers.add_parser('from-csv', help='Create from CSV file')
    csv_parser.add_argument('--csv', type=str, required=True,
                           help='Path to CSV file')
    csv_parser.add_argument('--audio-column', type=str, default='audio',
                           help='Column name for audio paths')
    csv_parser.add_argument('--transcript-column', type=str, default='transcript',
                           help='Column name for transcripts')
    csv_parser.add_argument('--output', type=str, default='data/manifest.tsv',
                           help='Output TSV path')
    
    # Template
    template_parser = subparsers.add_parser('template', help='Create empty template')
    template_parser.add_argument('--output', type=str, default='data/manifest_template.tsv',
                                help='Output path')
    
    # Split
    split_parser = subparsers.add_parser('split', help='Split manifest into train/val')
    split_parser.add_argument('--manifest', type=str, required=True,
                             help='Path to manifest file')
    split_parser.add_argument('--train-ratio', type=float, default=0.9,
                             help='Ratio of training data (default: 0.9)')
    split_parser.add_argument('--output-dir', type=str, default='data',
                             help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'from-dir':
        create_manifest_from_directory(
            args.audio_dir,
            args.transcript_dir,
            args.output
        )
    elif args.command == 'from-csv':
        create_manifest_from_csv(
            args.csv,
            args.audio_column,
            args.transcript_column,
            args.output
        )
    elif args.command == 'template':
        create_empty_manifest_template(args.output)
    elif args.command == 'split':
        split_manifest(args.manifest, args.train_ratio, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
