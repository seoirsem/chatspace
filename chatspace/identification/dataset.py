"""Tulu dataset loading utilities."""

import json
import os
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset

FILTER_CACHE_DIR = Path("/workspace-vast/seoirsem/chunky/tulu/tulu_dataset_indices")

TULU_SET_MAP = {
    "0": "ai2-adapt-dev/numinamath_tir_math_decontaminated",
    "1": "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
    "2": "ai2-adapt-dev/oasst1_converted",
    "3": "ai2-adapt-dev/tulu_v3.9_table_gpt_5k",
    "4": "allenai/tulu-3-sft-personas-math-grade",
    "5": "ai2-adapt-dev/flan_v2_converted",
    "6": "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
    "7": "ai2-adapt-dev/no_robots_converted",
    "8": "ai2-adapt-dev/personahub_code_v2_34999",
    "9": "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
    "10": "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
    "11": "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    "12": "ai2-adapt-dev/coconot_converted",
    "13": "ai2-adapt-dev/tulu_hard_coded_repeated_10",
    "14": "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    "15": "ai2-adapt-dev/tulu_v3.9_aya_100k",
    "16": "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "17": "ai2-adapt-dev/personahub_math_v5_regen_149960",
    "18": "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
}


def save_source_indices(dataset: Dataset, indices_dir: str | Path) -> dict:
    """
    Create an index mapping source dataset names to row indices.

    Args:
        dataset: The Tulu dataset with a 'source' field
        indices_dir: Directory to save the index files

    Returns:
        Dictionary mapping source names to lists of indices
    """
    os.makedirs(indices_dir, exist_ok=True)

    unique_sources = set(dataset["source"])
    source_indices = {}

    for source_class in unique_sources:
        indices = [
            i for i, example in enumerate(dataset) if example["source"] == source_class
        ]
        source_indices[source_class] = indices

        safe_filename = "".join(
            c for c in source_class if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        np.save(
            os.path.join(indices_dir, f"{safe_filename}_indices.npy"), np.array(indices)
        )

    with open(os.path.join(indices_dir, "sources.json"), "w") as f:
        json.dump(list(unique_sources), f)

    return source_indices


def load_source_datasets(dataset: Dataset, indices_dir: str | Path) -> dict[str, Dataset]:
    """
    Load pre-computed source indices and create filtered datasets.

    Args:
        dataset: The full Tulu dataset
        indices_dir: Directory containing the sources.json index

    Returns:
        Dictionary mapping source names to filtered datasets
    """
    with open(os.path.join(indices_dir, "sources.json"), "r") as f:
        sources = json.load(f)

    source_datasets = {}
    for source_class in sources:
        safe_filename = "".join(
            c for c in source_class if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        indices = np.load(
            os.path.join(indices_dir, f"{safe_filename}_indices.npy")
        ).tolist()
        source_datasets[source_class] = dataset.select(indices)

    return source_datasets


def load_tulu_dataset(
    sub_dataset: str, n_samples: int = -1, filter_cache_dir: Path | None = None
) -> list[str]:
    """
    Load a Tulu sub-dataset and return text samples.

    Args:
        sub_dataset: Sub-dataset key (e.g., "0", "1", ..., "18") or full dataset name
        n_samples: Number of samples to randomly select (-1 for all)
        filter_cache_dir: Cache directory for source indices

    Returns:
        List of text strings (first message content from each conversation)
    """
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    filter_cache_dir = (
        FILTER_CACHE_DIR if filter_cache_dir is None else filter_cache_dir
    )

    # Build source index if needed
    if not os.path.exists(filter_cache_dir / "sources.json"):
        save_source_indices(dataset, filter_cache_dir)

    # Load filtered datasets by source
    dataset = load_source_datasets(dataset, filter_cache_dir)
    print(f"Available sources: {list(dataset.keys())}")

    # Select the requested sub-dataset
    # Map numeric key to full dataset name if needed
    actual_dataset_name = TULU_SET_MAP.get(sub_dataset, sub_dataset)
    dataset = dataset[actual_dataset_name]

    # Random sampling
    random.seed(42)
    indices = (
        random.sample(range(len(dataset)), n_samples)
        if n_samples != -1
        else range(len(dataset))
    )
    dataset = dataset.select(indices)

    # Extract first message content
    text_samples = [item["messages"][0]["content"] for item in dataset]

    return text_samples
