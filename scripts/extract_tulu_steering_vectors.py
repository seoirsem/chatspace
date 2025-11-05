"""Extract steering vectors from Tulu dataset subsets.

This script loads a Tulu sub-dataset, runs inference with activation capture,
and extracts steering vectors by averaging the last token position across samples.

Usage:
    uv run python scripts/extract_tulu_steering_vectors.py --subset 0 --n-samples 100
    uv run python scripts/extract_tulu_steering_vectors.py --subset 4 --n-samples -1  # all samples
"""

import argparse
import asyncio
import logging
from pathlib import Path

import torch
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import VLLMSteerModel, VLLMSteeringConfig
from chatspace.identification.dataset import load_tulu_dataset, TULU_SET_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def extract_steering_vectors(
    subset: str,
    n_samples: int = 100,
    model_name: str = "allenai/Llama-3.1-Tulu-3-8B",
    output_dir: Path = Path("/workspace/tulu_steering_vectors"),
    batch_size: int = 16,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> None:
    """Extract steering vectors from Tulu dataset subset.

    Args:
        subset: Sub-dataset key (e.g., "0", "1", ..., "18")
        n_samples: Number of samples to process (-1 for all)
        model_name: Model name or path
        output_dir: Directory to save steering vectors
        batch_size: Number of prompts to process in parallel
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization fraction
    """
    # 1. Load dataset
    logger.info(f"Loading Tulu subset '{subset}' ({TULU_SET_MAP.get(subset, 'unknown')})")
    logger.info(f"Requesting {n_samples} samples (-1 = all)")

    text_samples = load_tulu_dataset(sub_dataset=subset, n_samples=n_samples)

    logger.info(f"Loaded {len(text_samples)} text samples")
    if len(text_samples) == 0:
        logger.error("No samples loaded. Exiting.")
        return

    # 2. Initialize model
    logger.info(f"Initializing model: {model_name}")

    # We need to bootstrap all layers for capture
    # We'll determine layer count after initialization
    model = VLLMSteerModel(
        VLLMSteeringConfig(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    )

    # Get layer count
    num_layers = model.layer_count
    logger.info(f"Model has {num_layers} layers")

    # All layers for capture
    capture_layers = list(range(num_layers))

    # 3. Run inference with capture
    logger.info(f"Running inference with capture on {len(text_samples)} samples")
    logger.info(f"Capturing all {len(capture_layers)} layers")

    # Use zero-token generation: max_tokens=1 means we only process the prompt
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,  # Deterministic (not that it matters for captures)
    )

    # Process in batches
    all_handles = []
    for i in range(0, len(text_samples), batch_size):
        batch = text_samples[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(text_samples) + batch_size - 1)//batch_size} ({len(batch)} samples)")

        _, handles = await model.generate(
            prompts=batch,
            sampling_params=sampling_params,
            capture_layers=capture_layers,
        )

        all_handles.extend(handles)

    # 4. Fetch all captures
    logger.info("Fetching captures from workers...")
    await model.fetch_captures_batch(all_handles)

    # 5. Extract last token position and compute mean per layer
    logger.info("Extracting last token position and computing layer-wise means...")

    # Accumulate last-token activations per layer
    layer_activations = {layer_idx: [] for layer_idx in capture_layers}

    for handle in all_handles:
        for layer_idx in capture_layers:
            # Get captured hidden states for this layer
            # Format: handle.captures[layer_idx][0]["hidden"]
            # Shape: [seq_len, hidden_size]
            hidden_states = handle.captures[layer_idx][0]["hidden"]

            # Extract last token position
            last_token_activation = hidden_states[-1, :]  # shape: [hidden_size]

            layer_activations[layer_idx].append(last_token_activation)

    # Compute mean per layer
    logger.info("Computing mean vectors per layer...")
    steering_vectors = {}
    for layer_idx in capture_layers:
        # Stack all samples for this layer
        stacked = torch.stack(layer_activations[layer_idx])  # shape: [n_samples, hidden_size]

        # Compute mean
        mean_vector = stacked.mean(dim=0)  # shape: [hidden_size]

        steering_vectors[layer_idx] = mean_vector

        # Log statistics
        norm = torch.linalg.norm(mean_vector).item()
        logger.info(f"Layer {layer_idx:2d}: mean L2 norm = {norm:.4f}")

    # 6. Save steering vectors
    output_path = output_dir / subset
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving steering vectors to {output_path}")

    # Save metadata
    metadata = {
        "subset": subset,
        "subset_name": TULU_SET_MAP.get(subset, "unknown"),
        "n_samples": len(text_samples),
        "model_name": model_name,
        "num_layers": num_layers,
        "extraction_method": "last_token_mean",
    }

    metadata_path = output_path / "metadata.json"
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # Save individual layer vectors
    for layer_idx, vector in steering_vectors.items():
        vector_path = output_path / f"layer_{layer_idx:02d}.pt"
        torch.save(
            {
                "vector": vector.cpu(),
                "metadata": metadata,
            },
            vector_path
        )

    logger.info(f"Saved {len(steering_vectors)} layer vectors to {output_path}")

    # Save all layers in one file for convenience
    all_layers_path = output_path / "all_layers.pt"
    torch.save(
        {
            "vectors": {layer_idx: vec.cpu() for layer_idx, vec in steering_vectors.items()},
            "metadata": metadata,
        },
        all_layers_path
    )
    logger.info(f"Saved combined file to {all_layers_path}")

    logger.info("Done!")


def main():
    # Format the subset mapping for help text
    subset_mapping = "\n".join([
        f"  {key}: {name}" for key, name in TULU_SET_MAP.items()
    ])

    parser = argparse.ArgumentParser(
        description="Extract steering vectors from Tulu dataset subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help=f"Tulu sub-dataset key (0-18). Available subsets:\n{subset_mapping}"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to process (-1 for all, default: 100)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/Llama-3.1-Tulu-3-8B",
        help="Model name or path (default: allenai/Llama-3.1-Tulu-3-8B)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/tulu_steering_vectors"),
        help="Output directory (default: /workspace/tulu_steering_vectors)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for parallel inference (default: 16)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)"
    )

    args = parser.parse_args()

    # Validate subset
    if args.subset not in TULU_SET_MAP:
        logger.error(f"Invalid subset '{args.subset}'. Available: {', '.join(TULU_SET_MAP.keys())}")
        return

    logger.info("=" * 80)
    logger.info("Tulu Steering Vector Extraction")
    logger.info("=" * 80)
    logger.info(f"Subset: {args.subset} ({TULU_SET_MAP[args.subset]})")
    logger.info(f"Samples: {args.n_samples} (-1 = all)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info("=" * 80)

    asyncio.run(
        extract_steering_vectors(
            subset=args.subset,
            n_samples=args.n_samples,
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    )


if __name__ == "__main__":
    main()
