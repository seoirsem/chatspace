"""Test contrastive steering vectors computed from two Tulu datasets.

This script loads two Tulu subsets (target and baseline), computes the contrastive
steering vector on-the-fly as mean(target) - mean(baseline), and tests the effect
on generation.

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
    # Math steering with default test prompts
    uv run python scripts/test_tulu_contrastive_steering.py \
        --target-subset 0 --baseline-subset 5 --layer 16

    # Math steering with prompts from the math dataset
    uv run python scripts/test_tulu_contrastive_steering.py \
        --target-subset 0 --baseline-subset 5 --layer 16 \
        --inference-dataset 0 --inference-n-samples 50

    # Code steering with prompts from code dataset
    uv run python scripts/test_tulu_contrastive_steering.py \
        --target-subset 8 --baseline-subset 5 --layer 18 --scale 1.0 \
        --inference-dataset 8 --inference-n-samples 20

Output:
    Default prompts: {output_dir}/{target}_{baseline}/layer_{layer}_scale_{scale}/results.json
    Dataset prompts: {output_dir}/{target}_{baseline}/layer_{layer}_scale_{scale}/out_{dataset}_{n}.json
"""

import argparse
import asyncio
import json
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


DEFAULT_TEST_PROMPTS = [
    "What is 2 + 2?",
    "Explain the concept of recursion in programming.",
    "Write a poem about artificial intelligence.",
    "How do I sort a list in Python?",
    "What are the main causes of climate change?",
]


async def test_contrastive_steering(
    target_subset: str,
    baseline_subset: str,
    layer: int,
    scale: float = 0.5,
    model_name: str = "allenai/Llama-3.1-Tulu-3-8B",
    n_samples: int = 100,
    test_prompts: list[str] | None = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    output_dir: Path | None = None,
) -> None:
    """Test contrastive steering computed from two Tulu datasets.

    Args:
        target_subset: Target dataset subset key (e.g., "0" for math)
        baseline_subset: Baseline dataset subset key (e.g., "5" for FLAN)
        layer: Layer index to apply steering to
        scale: Steering scale/magnitude
        model_name: Model name or path
        n_samples: Number of samples to use for computing steering vector
        test_prompts: List of test prompts (uses defaults if None)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        tensor_parallel_size: Number of GPUs for tensor parallelism
        output_dir: Output directory for results
    """
    # 1. Load datasets
    logger.info("=" * 80)
    logger.info("Loading Datasets")
    logger.info("=" * 80)
    logger.info(f"Target: {target_subset} ({TULU_SET_MAP.get(target_subset, 'unknown')})")
    logger.info(f"Baseline: {baseline_subset} ({TULU_SET_MAP.get(baseline_subset, 'unknown')})")
    logger.info(f"Samples per dataset: {n_samples}")

    target_samples = load_tulu_dataset(sub_dataset=target_subset, n_samples=n_samples)
    baseline_samples = load_tulu_dataset(sub_dataset=baseline_subset, n_samples=n_samples)

    logger.info(f"Loaded {len(target_samples)} target samples")
    logger.info(f"Loaded {len(baseline_samples)} baseline samples")

    # 2. Initialize model
    logger.info("")
    logger.info("=" * 80)
    logger.info("Initializing Model")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")

    model = VLLMSteerModel(
        VLLMSteeringConfig(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            bootstrap_layers=(layer,),
        )
    )

    num_layers = model.layer_count
    logger.info(f"Model has {num_layers} layers")
    logger.info(f"Target layer: {layer}")

    # Setup cache directory
    if output_dir is None:
        cache_dir = Path("/workspace/tulu_contrastive_steering") / "dataset_cache"
    else:
        cache_dir = output_dir / "dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 3. Capture activations for target dataset
    logger.info("")
    logger.info("=" * 80)
    logger.info("Capturing Target Activations")
    logger.info("=" * 80)

    # Check cache for target dataset
    target_cache_file = cache_dir / f"{target_subset}_{n_samples}_layer_{layer}.pt"

    if target_cache_file.exists():
        logger.info(f"Loading target activations from cache: {target_cache_file}")
        cache_data = torch.load(target_cache_file)
        target_mean = cache_data["mean_vector"]
        target_norm = cache_data["norm"]
        logger.info(f"Target mean vector L2 norm: {target_norm:.4f} (cached)")
    else:
        logger.info("Cache not found, capturing target activations...")
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

        # Format as chat messages
        target_messages = [[{"role": "user", "content": text}] for text in target_samples]

        _, target_handles = await model.chat(
            messages=target_messages,
            sampling_params=sampling_params,
            capture_layers=[layer],
        )

        await model.fetch_captures_batch(target_handles)

        # Extract last token from each target sample
        target_activations = []
        for handle in target_handles:
            hidden_states = handle.captures[layer][0]["hidden"]
            last_token = hidden_states[-1, :]
            target_activations.append(last_token)

        target_mean = torch.stack(target_activations).mean(dim=0)
        target_norm = torch.linalg.norm(target_mean).item()
        logger.info(f"Target mean vector L2 norm: {target_norm:.4f}")

        # Save to cache
        torch.save({
            "mean_vector": target_mean.cpu(),
            "norm": target_norm,
            "subset": target_subset,
            "n_samples": n_samples,
            "layer": layer,
            "model_name": model_name,
        }, target_cache_file)
        logger.info(f"Saved target activations to cache: {target_cache_file}")

    # 4. Capture activations for baseline dataset
    logger.info("")
    logger.info("=" * 80)
    logger.info("Capturing Baseline Activations")
    logger.info("=" * 80)

    # Check cache for baseline dataset
    baseline_cache_file = cache_dir / f"{baseline_subset}_{n_samples}_layer_{layer}.pt"

    if baseline_cache_file.exists():
        logger.info(f"Loading baseline activations from cache: {baseline_cache_file}")
        cache_data = torch.load(baseline_cache_file)
        baseline_mean = cache_data["mean_vector"]
        baseline_norm = cache_data["norm"]
        logger.info(f"Baseline mean vector L2 norm: {baseline_norm:.4f} (cached)")
    else:
        logger.info("Cache not found, capturing baseline activations...")
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

        # Format as chat messages
        baseline_messages = [[{"role": "user", "content": text}] for text in baseline_samples]

        _, baseline_handles = await model.chat(
            messages=baseline_messages,
            sampling_params=sampling_params,
            capture_layers=[layer],
        )

        await model.fetch_captures_batch(baseline_handles)

        # Extract last token from each baseline sample
        baseline_activations = []
        for handle in baseline_handles:
            hidden_states = handle.captures[layer][0]["hidden"]
            last_token = hidden_states[-1, :]
            baseline_activations.append(last_token)

        baseline_mean = torch.stack(baseline_activations).mean(dim=0)
        baseline_norm = torch.linalg.norm(baseline_mean).item()
        logger.info(f"Baseline mean vector L2 norm: {baseline_norm:.4f}")

        # Save to cache
        torch.save({
            "mean_vector": baseline_mean.cpu(),
            "norm": baseline_norm,
            "subset": baseline_subset,
            "n_samples": n_samples,
            "layer": layer,
            "model_name": model_name,
        }, baseline_cache_file)
        logger.info(f"Saved baseline activations to cache: {baseline_cache_file}")

    # 5. Compute contrastive steering vector
    logger.info("")
    logger.info("=" * 80)
    logger.info("Computing Contrastive Steering Vector")
    logger.info("=" * 80)

    # Ensure vectors are on the same device (cached vectors are on CPU)
    target_mean = target_mean.to(device="cpu")
    baseline_mean = baseline_mean.to(device="cpu")

    steering_vector = target_mean - baseline_mean
    vector_norm = torch.linalg.norm(steering_vector).item()

    logger.info(f"Contrastive vector L2 norm: {vector_norm:.4f}")
    logger.info(f"Steering scale: {scale}")
    logger.info(f"Effective magnitude: {vector_norm * scale:.4f}")

    # 6. Prepare test prompts
    inference_dataset_name = None
    if test_prompts is None:
        test_prompts = DEFAULT_TEST_PROMPTS
        logger.info(f"Testing with {len(test_prompts)} default prompts")
    else:
        # test_prompts is a tuple: (dataset_subset, n_samples) or list of prompts
        if isinstance(test_prompts, tuple):
            inference_dataset_name, inference_n = test_prompts
            logger.info(f"Loading {inference_n} inference prompts from dataset {inference_dataset_name}")
            test_prompts = load_tulu_dataset(sub_dataset=inference_dataset_name, n_samples=inference_n)
            logger.info(f"Loaded {len(test_prompts)} prompts from {TULU_SET_MAP.get(inference_dataset_name, 'unknown')}")
        else:
            logger.info(f"Testing with {len(test_prompts)} custom prompts")

    test_sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # 7. Generate without steering (baseline)
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASELINE (No Steering)")
    logger.info("=" * 80)

    # Explicitly clear any steering to ensure clean baseline
    await model.clear_all_vectors()

    # Format as chat messages
    test_messages = [[{"role": "user", "content": prompt}] for prompt in test_prompts]

    baseline_responses = await model.chat(
        messages=test_messages,
        sampling_params=test_sampling_params,
    )

    baseline_outputs = [resp.full_text() for resp in baseline_responses]

    for i, (prompt, output) in enumerate(zip(test_prompts, baseline_outputs)):
        logger.info(f"\n[Prompt {i+1}] {prompt}")
        logger.info(f"[Output] {output}")

    # 8. Apply steering vector
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"STEERED (Layer {layer}, Scale {scale})")
    logger.info("=" * 80)

    await model.set_layer_vector(layer, steering_vector * scale)

    steered_responses = await model.chat(
        messages=test_messages,
        sampling_params=test_sampling_params,
    )

    steered_outputs = [resp.full_text() for resp in steered_responses]

    for i, (prompt, output) in enumerate(zip(test_prompts, steered_outputs)):
        logger.info(f"\n[Prompt {i+1}] {prompt}")
        logger.info(f"[Output] {output}")

    # 9. Side-by-side comparison
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

    # 10. Save results
    if output_dir is None:
        output_dir = Path("/workspace/tulu_contrastive_steering")

    # Append subset and layer/scale info
    output_dir = output_dir / f"{target_subset}_{baseline_subset}" / f"layer_{layer}_scale_{scale}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")

    # Save JSON with all results
    results = {
        "metadata": {
            "target_subset": target_subset,
            "target_subset_name": TULU_SET_MAP.get(target_subset, "unknown"),
            "baseline_subset": baseline_subset,
            "baseline_subset_name": TULU_SET_MAP.get(baseline_subset, "unknown"),
            "n_samples": n_samples,
            "layer": layer,
            "scale": scale,
            "model_name": model_name,
            "target_norm": target_norm,
            "baseline_norm": baseline_norm,
            "contrastive_vector_norm": vector_norm,
            "effective_magnitude": vector_norm * scale,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "inference_dataset": inference_dataset_name,
            "inference_dataset_name": TULU_SET_MAP.get(inference_dataset_name, None) if inference_dataset_name else None,
            "inference_n_samples": inference_n if inference_dataset_name else None,
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

    # Determine output filename
    if inference_dataset_name is not None:
        results_filename = f"out_{inference_dataset_name}_{inference_n}.json"
    else:
        results_filename = "results.json"

    results_path = output_dir / results_filename
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    logger.info("Done!")


def main():
    # Format the subset mapping for help text
    subset_mapping = "\n".join([
        f"  {key}: {name}" for key, name in TULU_SET_MAP.items()
    ])

    parser = argparse.ArgumentParser(
        description="Test contrastive steering computed from two Tulu datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target-subset",
        type=str,
        required=True,
        help=f"Target dataset subset key (e.g., 0 for math). Available:\n{subset_mapping}"
    )
    parser.add_argument(
        "--baseline-subset",
        type=str,
        required=True,
        help=f"Baseline dataset subset key (e.g., 5 for FLAN). Available:\n{subset_mapping}"
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
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per dataset for computing steering vector (default: 100)"
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
        help="Custom test prompts (default: uses built-in prompts, overridden by --inference-dataset)"
    )
    parser.add_argument(
        "--inference-dataset",
        type=str,
        help=f"Use prompts from a Tulu dataset subset for inference (e.g., '0' for math). Available:\n{subset_mapping}"
    )
    parser.add_argument(
        "--inference-n-samples",
        type=int,
        default=10,
        help="Number of prompts to sample from inference dataset (default: 10)"
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
        help="Base output directory (default: /workspace/tulu_contrastive_steering). Results saved to {output_dir}/{target}_{baseline}/layer_{layer}_scale_{scale}/"
    )

    args = parser.parse_args()

    # Validate subsets
    if args.target_subset not in TULU_SET_MAP:
        logger.error(f"Invalid target subset '{args.target_subset}'. Available: {', '.join(TULU_SET_MAP.keys())}")
        return
    if args.baseline_subset not in TULU_SET_MAP:
        logger.error(f"Invalid baseline subset '{args.baseline_subset}'. Available: {', '.join(TULU_SET_MAP.keys())}")
        return
    if args.inference_dataset and args.inference_dataset not in TULU_SET_MAP:
        logger.error(f"Invalid inference dataset '{args.inference_dataset}'. Available: {', '.join(TULU_SET_MAP.keys())}")
        return

    # Determine test prompts source
    if args.inference_dataset:
        test_prompts = (args.inference_dataset, args.inference_n_samples)
    else:
        test_prompts = args.prompts

    logger.info("=" * 80)
    logger.info("Tulu Contrastive Steering Test")
    logger.info("=" * 80)
    logger.info(f"Target: {args.target_subset} ({TULU_SET_MAP[args.target_subset]})")
    logger.info(f"Baseline: {args.baseline_subset} ({TULU_SET_MAP[args.baseline_subset]})")
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Scale: {args.scale}")
    logger.info(f"Model: {args.model}")
    logger.info("=" * 80)

    asyncio.run(
        test_contrastive_steering(
            target_subset=args.target_subset,
            baseline_subset=args.baseline_subset,
            layer=args.layer,
            scale=args.scale,
            model_name=args.model,
            n_samples=args.n_samples,
            test_prompts=test_prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
