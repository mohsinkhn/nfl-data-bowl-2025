"""Test the NFLTrajectoryDataset implementation."""

import sys
from pathlib import Path
import time
import psutil
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import NFLTrajectoryDataset, collate_fn
from torch.utils.data import DataLoader


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def test_dataset():
    """Test dataset loading and basic functionality."""
    print("\n" + "=" * 60)
    print("Testing NFLTrajectoryDataset")
    print("=" * 60)

    # Measure initial memory
    mem_before = get_memory_usage()
    print(f"\nMemory before loading: {mem_before:.2f} MB")

    # Create dataset
    dataset = NFLTrajectoryDataset(
        input_parquet="data/processed/fold1/train_input.parquet",
        output_parquet="data/processed/fold1/train_output.parquet",
        max_encoder_len=40,
        max_decoder_len=32,
        normalize=True,
        rotation_normalize=True,
    )

    # Measure memory after loading
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before
    mem_per_sample = mem_used / len(dataset) * 1024  # Convert to KB

    print(f"\n✓ Dataset loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Memory after loading: {mem_after:.2f} MB")
    print(f"  Memory used by dataset: {mem_used:.2f} MB")
    print(f"  Memory per sample: {mem_per_sample:.2f} KB")

    # Test access speed
    print(f"\nTesting access speed...")
    start = time.time()
    for i in range(100):
        _ = dataset[i]
    elapsed = time.time() - start
    print(
        f"  Accessed 100 samples in {elapsed:.3f}s ({elapsed/100*1000:.2f}ms per sample)"
    )

    # Test single sample
    print(f"\nTesting single sample...")
    sample = dataset[0]

    print(f"  Sample keys: {sample.keys()}")
    print(f"  encoder_input shape: {sample['encoder_input'].shape}")
    print(f"  decoder_input shape: {sample['decoder_input'].shape}")
    print(f"  decoder_target shape: {sample['decoder_target'].shape}")
    print(f"  encoder_mask shape: {sample['encoder_mask'].shape}")
    print(f"  decoder_mask shape: {sample['decoder_mask'].shape}")
    print(f"  encoder_mask sum: {sample['encoder_mask'].sum()}")
    print(f"  decoder_mask sum: {sample['decoder_mask'].sum()}")
    print(f"  metadata: {sample['metadata']}")

    # Test batch
    print(f"\nTesting DataLoader with batch_size=4...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    print(f"  Batch keys: {batch.keys()}")
    print(f"  encoder_input shape: {batch['encoder_input'].shape}")
    print(f"  decoder_input shape: {batch['decoder_input'].shape}")
    print(f"  decoder_target shape: {batch['decoder_target'].shape}")
    print(f"  encoder_mask shape: {batch['encoder_mask'].shape}")
    print(f"  decoder_mask shape: {batch['decoder_mask'].shape}")
    print(f"  metadata length: {len(batch['metadata'])}")

    # Check statistics
    print(f"\nStatistics from first batch:")
    print(f"  encoder_input mean: {batch['encoder_input'].mean():.4f}")
    print(f"  encoder_input std: {batch['encoder_input'].std():.4f}")
    print(f"  decoder_target mean: {batch['decoder_target'].mean():.4f}")
    print(f"  decoder_target std: {batch['decoder_target'].std():.4f}")

    # Check per-feature statistics
    print(f"\nPer-feature statistics (encoder_input):")
    feature_names = ["x", "y", "s", "a", "dir", "o"]
    for i, name in enumerate(feature_names):
        feat = batch["encoder_input"][:, :, i]
        # Only compute stats on non-zero values (ignore padding)
        valid = feat[batch["encoder_mask"]]
        print(
            f"  {name:3s}: min={valid.min():.4f}, max={valid.max():.4f}, "
            f"mean={valid.mean():.4f}, std={valid.std():.4f}"
        )

    print(f"\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    # Test validation dataset memory
    print(f"\n" + "=" * 60)
    print("Testing Validation Dataset Memory")
    print("=" * 60)

    mem_before_val = get_memory_usage()
    val_dataset = NFLTrajectoryDataset(
        input_parquet="data/processed/fold1/val_input.parquet",
        output_parquet="data/processed/fold1/val_output.parquet",
        max_encoder_len=40,
        max_decoder_len=32,
        normalize=True,
        rotation_normalize=True,
    )
    mem_after_val = get_memory_usage()
    mem_used_val = mem_after_val - mem_before_val

    print(f"\n✓ Validation dataset loaded")
    print(f"  Total samples: {len(val_dataset)}")
    print(f"  Memory used: {mem_used_val:.2f} MB")
    print(f"  Memory per sample: {mem_used_val / len(val_dataset) * 1024:.2f} KB")

    print(f"\n" + "=" * 60)
    print("Memory Summary")
    print("=" * 60)
    print(f"  Train dataset: {mem_used:.2f} MB ({len(dataset):,} samples)")
    print(f"  Val dataset: {mem_used_val:.2f} MB ({len(val_dataset):,} samples)")
    print(f"  Total memory: {mem_used + mem_used_val:.2f} MB")
    print(f"  Current total: {get_memory_usage():.2f} MB")


if __name__ == "__main__":
    test_dataset()
