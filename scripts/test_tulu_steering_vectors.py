"""Test steering vectors extracted from Tulu datasets.

This script loads extracted steering vectors and compares model outputs
with and without steering applied. Results are saved to disk.

Recommended steering scales:
    0.1-0.5:  Subtle steering (good starting point)
    0.5-1.0:  Moderate steering
    1.0-2.0:  Strong steering
    2.0-5.0:  Very strong (may degrade coherence)
    >5.0:     Extreme (likely breaks model)

Recommended layers (for 32-layer models like Tulu 3 8B):
    Early (0-10):     May affect basic processing, less semantic
    Middle (11-21):   Best for semantic steering (try 16-20)
    Late (22-31):     Closer to output, may affect style/format

Usage:
    uv run python scripts/test_tulu_steering_vectors.py --vectors-dir /workspace/tulu_steering_vectors/0 --layer 16
    uv run python scripts/test_tulu_steering_vectors.py --vectors-dir /workspace/tulu_steering_vectors/0 --layer 16 --scale 1.5

Output:
    {vectors_dir}/inference_layer_{layer}_scale_{scale}/results.json
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import torch
from vllm import SamplingParams

from chatspace.generation.vllm_steer_model import VLLMSteerModel, VLLMSteeringConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


DEFAULT_TEST_PROMPTS = [
    "What is 2 + 2?",
    "Explain the concept of recursion in programming.",
    "Write a poem about artificial intelligence.",
    "How do I sort a list in Python?",
    "What are the main causes of climate change?",
]


async def test_steering_vectors(
    vectors_dir: Path,
    layer: int,
    scale: float = 1.0,
    model_name: str = "allenai/Llama-3.1-Tulu-3-8B",
    test_prompts: list[str] | None = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    output_dir: Path | None = None,
) -> None:
    """Test steering vectors by comparing outputs with and without steering.

    Args:
        vectors_dir: Directory containing extracted steering vectors
        layer: Layer index to apply steering to
        scale: Steering scale/magnitude
        model_name: Model name or path
        test_prompts: List of test prompts (uses defaults if None)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        tensor_parallel_size: Number of GPUs for tensor parallelism
        output_dir: Directory to save results (default: vectors_dir/inference_layer_{layer}_scale_{scale})
    """
    # 1. Load metadata and steering vector
    metadata_path = vectors_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info("=" * 80)
    logger.info("Steering Vector Metadata")
    logger.info("=" * 80)
    logger.info(f"Subset: {metadata.get('subset')} ({metadata.get('subset_name')})")
    logger.info(f"Samples: {metadata.get('n_samples')}")
    logger.info(f"Model: {metadata.get('model_name')}")
    logger.info(f"Num layers: {metadata.get('num_layers')}")
    logger.info(f"Extraction method: {metadata.get('extraction_method')}")
    logger.info("=" * 80)

    # Check if the specified layer exists
    vector_path = vectors_dir / f"layer_{layer:02d}.pt"
    if not vector_path.exists():
        logger.error(f"Vector file not found: {vector_path}")
        logger.info(f"Available layers: 0-{metadata.get('num_layers', '?') - 1}")
        return

    # Load steering vector
    logger.info(f"Loading steering vector for layer {layer} from {vector_path}")
    vector_data = torch.load(vector_path)
    steering_vector = vector_data["vector"]

    vector_norm = torch.linalg.norm(steering_vector).item()
    logger.info(f"Steering vector shape: {steering_vector.shape}")
    logger.info(f"Steering vector L2 norm: {vector_norm:.4f}")
    logger.info(f"Steering scale: {scale}")
    logger.info(f"Effective magnitude: {vector_norm * scale:.4f}")

    # 2. Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = VLLMSteerModel(
        VLLMSteeringConfig(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            bootstrap_layers=(layer,),
        )
    )

    # 3. Prepare test prompts
    if test_prompts is None:
        test_prompts = DEFAULT_TEST_PROMPTS

    logger.info(f"Testing with {len(test_prompts)} prompts")

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # 4. Generate without steering (baseline)
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASELINE (No Steering)")
    logger.info("=" * 80)

    baseline_outputs = await model.generate(
        prompts=test_prompts,
        sampling_params=sampling_params,
    )

    for i, (prompt, output) in enumerate(zip(test_prompts, baseline_outputs)):
        logger.info(f"\n[Prompt {i+1}] {prompt}")
        logger.info(f"[Output] {output}")

    # 5. Apply steering vector
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"STEERED (Layer {layer}, Scale {scale})")
    logger.info("=" * 80)

    # Set the steering vector
    await model.set_layer_vector(layer, steering_vector * scale)

    steered_outputs = await model.generate(
        prompts=test_prompts,
        sampling_params=sampling_params,
    )

    for i, (prompt, output) in enumerate(zip(test_prompts, steered_outputs)):
        logger.info(f"\n[Prompt {i+1}] {prompt}")
        logger.info(f"[Output] {output}")

    # 6. Side-by-side comparison
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON")
    logger.info("=" * 80)

    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Prompt {i+1}: {prompt}")
        logger.info(f"{'=' * 80}")
        logger.info(f"\nBaseline:\n{baseline_outputs[i]}")
        logger.info(f"\nSteered:\n{steered_outputs[i]}")
        logger.info("")

    # 7. Save results to file
    if output_dir is None:
        output_dir = vectors_dir / f"inference_layer_{layer}_scale_{scale}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")

    # Save JSON with all results
    results = {
        "metadata": {
            "vectors_dir": str(vectors_dir),
            "subset": metadata.get("subset"),
            "subset_name": metadata.get("subset_name"),
            "layer": layer,
            "scale": scale,
            "model_name": model_name,
            "vector_norm": vector_norm,
            "effective_magnitude": vector_norm * scale,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        "results": [
            {
                "prompt_idx": i,
                "prompt": prompt,
                "baseline": baseline_outputs[i],
                "steered": steered_outputs[i],
            }
            for i, prompt in enumerate(test_prompts)
        ]
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Test steering vectors extracted from Tulu datasets"
    )
    parser.add_argument(
        "--vectors-dir",
        type=Path,
        required=True,
        help="Directory containing extracted steering vectors (e.g., /workspace/tulu_steering_vectors/0)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to apply steering to (e.g., 16 for middle layer of 32-layer model)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Steering scale/magnitude (default: 0.5, recommended: 0.1-2.0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/Llama-3.1-Tulu-3-8B",
        help="Model name or path (default: allenai/Llama-3.1-Tulu-3-8B)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Custom test prompts (default: uses built-in prompts)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (default: vectors_dir/inference_layer_{layer}_scale_{scale})"
    )

    args = parser.parse_args()

    # Validate vectors directory
    if not args.vectors_dir.exists():
        logger.error(f"Vectors directory does not exist: {args.vectors_dir}")
        return

    logger.info("=" * 80)
    logger.info("Tulu Steering Vector Test")
    logger.info("=" * 80)
    logger.info(f"Vectors directory: {args.vectors_dir}")
    logger.info(f"Target layer: {args.layer}")
    logger.info(f"Steering scale: {args.scale}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info("=" * 80)

    asyncio.run(
        test_steering_vectors(
            vectors_dir=args.vectors_dir,
            layer=args.layer,
            scale=args.scale,
            model_name=args.model,
            test_prompts=args.prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
