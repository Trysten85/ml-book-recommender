"""
Fine-tune sentence transformer on book similarity pairs

This script takes the 21,665 training pairs we generated and fine-tunes
a sentence transformer model to create a custom "ThemeMatch" model that
understands literary themes and book similarity better than the base model.

Training approach:
- Base model: all-MiniLM-L6-v2 (77MB, 384 dimensions)
- Loss function: CosineSimilarityLoss (learns to match our similarity labels)
- Epochs: 4-6 (typical for fine-tuning)
- Batch size: 16 (fits in 8GB RAM)
- Training time: ~10-15 mins CPU, ~2-3 mins GPU

Output: Custom model saved to models/thematch-v1/
"""

import json
import sys
import io
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import random
import multiprocessing

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set random seed for reproducibility
random.seed(42)

# CPU Optimization: Use all available cores
# AMD 9600X has 6 cores / 12 threads
# Windows multiprocessing has pickling issues, so use num_workers=0
NUM_WORKERS = 0 if sys.platform == 'win32' else min(6, multiprocessing.cpu_count() // 2)
torch.set_num_threads(multiprocessing.cpu_count())  # Use all threads for compute

print(f"CPU Optimization Enabled:")
print(f"  - Torch threads: {torch.get_num_threads()}")
print(f"  - DataLoader workers: {NUM_WORKERS} (Windows: single-process mode)")
print()


class BookThemeTrainer:
    """Fine-tune sentence transformer on book similarity pairs"""

    def __init__(self, base_model='all-MiniLM-L6-v2'):
        """
        Initialize trainer

        Args:
            base_model: Pre-trained model to fine-tune from
        """
        print("="*80)
        print("FINE-TUNING SENTENCE TRANSFORMER FOR BOOK THEMES")
        print("="*80)
        print()

        print(f"Loading base model: {base_model}")
        self.model = SentenceTransformer(base_model)
        print(f"✓ Loaded model: {self.model.get_sentence_embedding_dimension()} dimensions")
        print()

    def load_training_data(self, train_dir='data/training',
                          val_split=0.1, max_pairs=None):
        """
        Load training pairs from JSONL files

        Args:
            train_dir: Directory with training JSONL files
            val_split: Fraction of data to use for validation
            max_pairs: Maximum pairs to load (None = all)

        Returns:
            train_examples, val_examples
        """
        print("Loading training data...")
        train_path = Path(train_dir)

        all_pairs = []

        # Load all three types of pairs
        for file in ['series_pairs.jsonl', 'genre_pairs.jsonl', 'negative_pairs.jsonl']:
            filepath = train_path / file
            print(f"  Loading {file}...")

            with open(filepath, 'r', encoding='utf-8') as f:
                pairs = [json.loads(line) for line in f]
                all_pairs.extend(pairs)
                print(f"    Loaded {len(pairs):,} pairs")

        print(f"✓ Total pairs loaded: {len(all_pairs):,}")

        # Limit if requested
        if max_pairs and max_pairs < len(all_pairs):
            all_pairs = random.sample(all_pairs, max_pairs)
            print(f"  Sampled {max_pairs:,} pairs for training")

        # Convert to InputExample format
        print("\nConverting to training format...")
        examples = []
        for pair in all_pairs:
            examples.append(
                InputExample(
                    texts=[pair['text1'], pair['text2']],
                    label=float(pair['label'])
                )
            )

        # Shuffle
        random.shuffle(examples)

        # Split into train/val
        val_size = int(len(examples) * val_split)
        val_examples = examples[:val_size]
        train_examples = examples[val_size:]

        print(f"✓ Training examples: {len(train_examples):,}")
        print(f"✓ Validation examples: {len(val_examples):,}")
        print()

        return train_examples, val_examples

    def train(self, train_examples, val_examples,
              output_path='models/thematch-v1',
              epochs=4, batch_size=16, warmup_steps=100):
        """
        Fine-tune the model

        Args:
            train_examples: List of InputExample for training
            val_examples: List of InputExample for validation
            output_path: Where to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            warmup_steps: Number of warmup steps
        """
        print("="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Training examples: {len(train_examples):,}")
        print(f"Steps per epoch: {len(train_examples) // batch_size}")
        print(f"Total steps: {(len(train_examples) // batch_size) * epochs}")
        print()

        # Create DataLoader with CPU optimization
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,  # Multi-process data loading
            pin_memory=False,  # CPU only, no GPU
            persistent_workers=True if NUM_WORKERS > 0 else False  # Keep workers alive between epochs
        )

        # Define loss function
        # CosineSimilarityLoss learns to make cosine similarity match our labels
        train_loss = losses.CosineSimilarityLoss(self.model)

        # Create evaluator for validation
        sentences1 = [ex.texts[0] for ex in val_examples[:1000]]  # Limit for speed
        sentences2 = [ex.texts[1] for ex in val_examples[:1000]]
        scores = [ex.label for ex in val_examples[:1000]]

        evaluator = EmbeddingSimilarityEvaluator(
            sentences1, sentences2, scores,
            name='book-similarity-validation'
        )

        print("="*80)
        print("STARTING FINE-TUNING")
        print("="*80)
        print()

        # Train!
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            evaluation_steps=500,  # Evaluate every 500 steps
            save_best_model=True,
            show_progress_bar=True
        )

        print()
        print("="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Model saved to: {output_path}")
        print()
        print("You can now use this model in your recommender:")
        print(f"  model = SentenceTransformer('{output_path}')")
        print()

        return output_path


def main():
    """Main training pipeline"""

    # Configuration - OPTIMIZED FOR CPU
    BASE_MODEL = 'all-MiniLM-L6-v2'
    OUTPUT_PATH = 'models/thematch-v2'  # v2 for CPU-optimized model
    EPOCHS = 4
    BATCH_SIZE = 32  # Increased from 16 for better throughput
    WARMUP_STEPS = 100

    # Initialize trainer
    trainer = BookThemeTrainer(base_model=BASE_MODEL)

    # Load data
    train_examples, val_examples = trainer.load_training_data(
        train_dir='data/training',
        val_split=0.1,  # 10% for validation
        max_pairs=None  # Use all pairs (set to e.g. 5000 for testing)
    )

    # Train
    model_path = trainer.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=OUTPUT_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS
    )

    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Evaluate the model:")
    print("   python src/evaluate_models.py")
    print()
    print("2. Use in recommender:")
    print("   recommender = BookRecommender(model_name='thematch-v1')")
    print()
    print("3. Compare with baseline:")
    print("   Test with 'Red Rising' and see the improvement!")


if __name__ == '__main__':
    main()
