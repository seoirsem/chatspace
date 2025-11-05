# Chunky Tulu Contrastive Steering

Documentation for the Tulu contrastive steering vector extraction and testing workflow.

## Overview

This workflow computes contrastive steering vectors from Tulu dataset subsets and tests their effect on model generation. The key script is:

```bash
scripts/test_tulu_contrastive_steering.py
```

**What it does:**
1. Loads two Tulu dataset subsets (target and baseline)
2. Captures activations from both datasets at a specified layer
3. Computes contrastive steering vector: `target_mean - baseline_mean`
4. Tests the steering effect by generating with and without the vector
5. Caches activations for fast re-runs with different parameters

## Example Command

```bash
uv run python scripts/test_tulu_contrastive_steering.py \
    --target-subset 12 \
    --baseline-subset 0 \
    --layer 16 \
    --inference-dataset 0 \
    --inference-n-samples 50 \
    --scale 2.0 \
    --output-dir /workspace-vast/seoirsem/chunky/steering_vectors
```

## Parameter Breakdown

### Core Steering Parameters

**`--target-subset 12`**
- Dataset to steer TOWARD (CoCoNot in this example)
- The model will exhibit MORE characteristics of this dataset
- See TULU_SET_MAP below for all options

**`--baseline-subset 0`**
- Dataset to steer AWAY FROM (Math in this example)
- Used as the neutral reference point
- Common choices: 0 (math), 5 (FLAN - generic)

**`--layer 16`**
- Which transformer layer to extract/apply steering at
- For 32-layer models (Tulu 3 8B):
  - Layers 0-10: Early, basic processing
  - **Layers 11-21: Middle, best for semantic steering** ⭐
  - Layers 22-31: Late, affects style/format
- Recommended: Start with 16-18

**`--scale 2.0`**
- Steering magnitude multiplier
- Recommended ranges:
  - 0.1-0.5: Subtle steering
  - 0.5-1.0: Moderate steering
  - 1.0-2.0: Strong steering
  - 2.0-5.0: Very strong (may degrade coherence)
  - >5.0: Extreme (likely breaks model)

### Inference Parameters

**`--inference-dataset 0`**
- Which dataset to sample test prompts from
- If omitted, uses 5 default built-in prompts
- Use the same subset as target to test domain-specific steering

**`--inference-n-samples 50`**
- Number of prompts to sample from inference dataset
- Default: 10
- -1 means use all samples (can be slow)

### Other Parameters

**`--n-samples 100`** (default)
- Number of samples per dataset for computing the steering vector
- More samples = more stable vector but slower first run
- Cached after first computation

**`--output-dir /workspace-vast/seoirsem/chunky/steering_vectors`**
- Base directory for outputs
- Default: `/workspace/tulu_contrastive_steering`

**`--max-tokens 100`** (default)
- Maximum tokens to generate during testing

**`--temperature 0.7`** (default)
- Sampling temperature for generation

**`--model allenai/Llama-3.1-Tulu-3-8B`** (default)
- Model to use

**`--tensor-parallel-size 1`** (default)
- Number of GPUs for tensor parallelism

## TULU_SET_MAP - Available Datasets

| Key | Dataset Name |
|-----|--------------|
| 0 | ai2-adapt-dev/numinamath_tir_math_decontaminated (Math) |
| 1 | ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k (Safety) |
| 2 | ai2-adapt-dev/oasst1_converted (Conversational) |
| 3 | ai2-adapt-dev/tulu_v3.9_table_gpt_5k (Tables) |
| 4 | allenai/tulu-3-sft-personas-math-grade (Math Personas) |
| 5 | ai2-adapt-dev/flan_v2_converted (FLAN - Generic Instructions) |
| 6 | ai2-adapt-dev/tulu_v3.9_sciriff_10k (Scientific) |
| 7 | ai2-adapt-dev/no_robots_converted (No Robots) |
| 8 | ai2-adapt-dev/personahub_code_v2_34999 (Code) |
| 9 | ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k (Algebra) |
| 10 | ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k (Jailbreak) |
| 11 | ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k (GSM8K Math) |
| 12 | ai2-adapt-dev/coconot_converted (CoCoNot) |
| 13 | ai2-adapt-dev/tulu_hard_coded_repeated_10 (Hard Coded) |
| 14 | ai2-adapt-dev/tulu_v3.9_wildchat_100k (Wild Chat) |
| 15 | ai2-adapt-dev/tulu_v3.9_aya_100k (Aya Multilingual) |
| 16 | ai2-adapt-dev/evol_codealpaca_heval_decontaminated (Code Evolution) |
| 17 | ai2-adapt-dev/personahub_math_v5_regen_149960 (Math Personas Large) |
| 18 | ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980 (IF Data) |

## Output Structure

### Cache Directory
```
{output_dir}/dataset_cache/
├── 12_100_layer_16.pt  # Target subset 12, 100 samples, layer 16
├── 0_100_layer_16.pt   # Baseline subset 0, 100 samples, layer 16
└── ...
```

**Cache format:**
- Filename: `{subset}_{n_samples}_layer_{layer}.pt`
- Contents: `mean_vector`, `norm`, `subset`, `n_samples`, `layer`, `model_name`
- Re-used across runs with same parameters

### Results Directory
```
{output_dir}/{target}_{baseline}/layer_{layer}_scale_{scale}/
├── out_{inference_dataset}_{inference_n}.json  # When using --inference-dataset
└── results.json                                 # When using default prompts
```

**Example from command above:**
```
/workspace-vast/seoirsem/chunky/steering_vectors/12_0/layer_16_scale_2.0/
└── out_0_50.json
```

### Results JSON Format
```json
{
  "metadata": {
    "target_subset": "12",
    "target_subset_name": "ai2-adapt-dev/coconot_converted",
    "baseline_subset": "0",
    "baseline_subset_name": "ai2-adapt-dev/numinamath_tir_math_decontaminated",
    "n_samples": 100,
    "layer": 16,
    "scale": 2.0,
    "contrastive_vector_norm": 7.25,
    "effective_magnitude": 14.5,
    "inference_dataset": "0",
    "inference_n_samples": 50,
    ...
  },
  "results": [
    {
      "prompt_idx": 0,
      "prompt": "...",
      "baseline": "...",  // Unsteered response
      "steered": "..."    // Steered response
    },
    ...
  ]
}
```

## Workflow Summary

1. **Load Datasets**
   - Target dataset (e.g., CoCoNot)
   - Baseline dataset (e.g., Math)
   - Optional: Inference dataset for test prompts

2. **Compute Steering Vector**
   - Check cache for pre-computed activations
   - If not cached:
     - Capture activations from both datasets at specified layer
     - Extract last token position from each prompt
     - Compute mean per dataset
     - Save to cache
   - Compute: `steering_vector = target_mean - baseline_mean`

3. **Test Inference**
   - Generate baseline outputs (no steering)
   - Apply steering vector to specified layer
   - Generate steered outputs (with steering)
   - Compare side-by-side

4. **Save Results**
   - JSON file with metadata and all prompt/response pairs
   - Separate files for different inference datasets

## Understanding Steering Direction

**The vector points FROM baseline TOWARD target:**

```
steering_vector = target_mean - baseline_mean
```

**Example: Math steering**
```bash
--target-subset 0 --baseline-subset 5  # Math - FLAN
# Effect: Model becomes MORE math-focused, LESS generic
```

**Example: Code steering**
```bash
--target-subset 8 --baseline-subset 5  # Code - FLAN
# Effect: Model becomes MORE code-focused, LESS generic
```

**Negative scale reverses direction:**
```bash
--target-subset 0 --baseline-subset 5 --scale -1.0
# Effect: Model becomes LESS math-focused, MORE generic
```

## Common Use Cases

### 1. Test Math Steering on Math Problems
```bash
uv run python scripts/test_tulu_contrastive_steering.py \
    --target-subset 0 \
    --baseline-subset 5 \
    --layer 16 \
    --scale 1.0 \
    --inference-dataset 0 \
    --inference-n-samples 100
```

### 2. Test Code Steering on Code Problems
```bash
uv run python scripts/test_tulu_contrastive_steering.py \
    --target-subset 8 \
    --baseline-subset 5 \
    --layer 18 \
    --scale 0.5 \
    --inference-dataset 8 \
    --inference-n-samples 50
```

### 3. Layer Sweep with Cached Activations
```bash
# First run: computes and caches activations
uv run python scripts/test_tulu_contrastive_steering.py \
    --target-subset 0 --baseline-subset 5 --layer 16 --scale 1.0

# Subsequent runs: INSTANT (loads from cache)
uv run python scripts/test_tulu_contrastive_steering.py \
    --target-subset 0 --baseline-subset 5 --layer 18 --scale 1.0

uv run python scripts/test_tulu_contrastive_steering.py \
    --target-subset 0 --baseline-subset 5 --layer 20 --scale 1.0
```

### 4. Scale Sweep with Cached Activations
```bash
# All runs after first are instant
for scale in 0.1 0.5 1.0 2.0 5.0; do
    uv run python scripts/test_tulu_contrastive_steering.py \
        --target-subset 0 --baseline-subset 5 --layer 16 --scale $scale
done
```

## Other Scripts (Not Currently Used)

### scripts/extract_tulu_steering_vectors.py
**Status**: Superseded by contrastive on-the-fly computation

This script was for pre-computing and saving steering vectors to disk. The contrastive approach computes vectors on-the-fly and caches activations instead, which is more flexible.

### scripts/test_tulu_steering_vectors.py
**Status**: Superseded by contrastive version

This script was for testing pre-extracted vectors. We're now using `test_tulu_contrastive_steering.py` which computes contrastive vectors and tests them in one workflow.

## Performance Notes

**First run with new dataset/layer:**
- Captures activations for both datasets
- Time: ~2-5 minutes depending on n_samples
- Saves to cache

**Subsequent runs with same dataset/layer:**
- Loads from cache
- Time: ~30 seconds for inference only
- Testing different scales or inference datasets is instant

**Cache benefits:**
- Reuse across different scales
- Reuse across different inference datasets
- Reuse across different test prompts
- Only re-compute when changing: dataset, n_samples, layer, or model

## Troubleshooting

**Weird baseline outputs:**
- Ensure model uses chat template (already implemented)
- Check that `clear_all_vectors()` is called before baseline (already implemented)

**No steering effect:**
- Try different layers (16-20 for semantic, 30-31 for style)
- Increase scale (but watch for coherence degradation)
- Check target/baseline are sufficiently different datasets
- Verify cache isn't stale (delete cache files if needed)

**Out of memory:**
- Reduce `--n-samples`
- Reduce `--inference-n-samples`
- Reduce `--max-tokens`
- Use `--tensor-parallel-size 2` or higher

## Tips

1. **Start with default prompts** (no `--inference-dataset`) to quickly test steering
2. **Use middle layers (16-20)** for semantic steering
3. **Start with moderate scale (0.5-1.0)** and adjust based on results
4. **Use FLAN (subset 5) as baseline** for most experiments (neutral, generic)
5. **Cache is your friend** - test many parameters quickly after first run
6. **Compare metadata norms** - if target_norm ≈ baseline_norm, datasets may be too similar
