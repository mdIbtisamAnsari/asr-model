"""
Configuration for Arabic Speech-to-Text Model
"""
import os

class Config:
    # Project paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    
    # Audio settings
    SAMPLE_RATE = 16000  # 16kHz is standard for speech
    N_MELS = 80  # Number of mel filterbanks
    N_FFT = 400  # FFT window size
    HOP_LENGTH = 160  # Hop length for STFT
    WIN_LENGTH = 400  # Window length
    
    # Model architecture
    ENCODER_DIM = 256
    DECODER_DIM = 512
    ATTENTION_DIM = 256
    ENCODER_LAYERS = 4
    DECODER_LAYERS = 2
    NUM_HEADS = 8
    DROPOUT = 0.1
    
    # Training settings
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 100
    GRADIENT_CLIP = 5.0
    WARMUP_STEPS = 1000
    
    # CTC settings
    BLANK_TOKEN = 0
    
    # Arabic character set (including common diacritics)
    ARABIC_CHARS = [
        '<blank>',  # CTC blank
        '<sos>',    # Start of sequence
        '<eos>',    # End of sequence
        '<pad>',    # Padding
        '<unk>',    # Unknown
        ' ',        # Space
        # Arabic letters
        'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ',
        'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق',
        'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ى',
        # Diacritics (tashkeel)
        'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ',
        # Common punctuation
        '،', '؟', '!', '.', ':', '-',
        # Numbers (Eastern Arabic numerals)
        '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩',
    ]
    
    VOCAB_SIZE = len(ARABIC_CHARS)
    
    # Special token indices
    BLANK_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    PAD_IDX = 3
    UNK_IDX = 4
    
    # Real-time inference settings
    CHUNK_SIZE = 1024  # Audio chunk size for real-time
    VAD_MODE = 3  # Voice Activity Detection aggressiveness (0-3)
    SILENCE_THRESHOLD = 30  # Frames of silence before processing
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.CHECKPOINT_DIR, cls.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def char_to_idx(cls):
        """Create character to index mapping"""
        return {char: idx for idx, char in enumerate(cls.ARABIC_CHARS)}
    
    @classmethod
    def idx_to_char(cls):
        """Create index to character mapping"""
        return {idx: char for idx, char in enumerate(cls.ARABIC_CHARS)}
