# NeSyDM Ablation Study

This directory contains the ablation study for investigating the "Neurosymbolic Diffusion Models" (NeSyDM) on the HalfMNIST task.

## Background: The HalfMNIST Challenge

HalfMNIST is a restricted addition task (digits 0-4) with missing combinations that creates a challenging reasoning shortcut problem:

- **Anchors**: The model sees (0,0)->0 and (0,1)->1
- **Ambiguity**: The model sees (2,3)->5 but NEVER sees (2,0) or (3,0)
- **The Shortcut**: Mathematically, mapping 2->4 and 3->1 is perfectly valid for the training data (4+1=5)

Standard NeSy models (with independence assumption) learn the shortcut and fail on out-of-distribution (OOD) data.

## Research Questions

1. **Visual Bias Hypothesis**: Does NeSyDM rely on specific visual patterns of digits 0 and 1?
2. **Entropy Importance**: Is entropy regularization critical for avoiding shortcuts?
3. **Denoising Importance**: Is the w-denoising loss critical?

## Ablation Experiments

### 1. Permutation Ablation (Test Visual Bias)

We test if NeSyDM succeeds because it exploits the simple visual structure of digits 0 and 1 (anchors).

**Available Permutations:**
- `identity`: Original HalfMNIST with digits [0,1,2,3,4]
- `shift5`: Uses digits [5,6,7,8,9] instead - harder visual patterns
- `shuffle`: Uses [7,2,9,4,1] - random shuffle
- `reverse`: Uses [4,3,2,1,0] - reversed ordering
- `swap01`: Uses [1,0,2,3,4] - only swap anchor digits
- `mid`: Uses [3,4,5,6,7] - middle digits

**Run Command:**
```bash
# Run full permutation ablation
python ablation_study.py --experiment permutation

# Run single permutation
python ablation_study.py --dataset permutedhalfmnist --digit_permutation shift5
```

### 2. Entropy Ablation

Test if entropy regularization is essential for preventing shortcuts.

```bash
# Run entropy ablation
python ablation_study.py --experiment entropy

# Run without entropy
python ablation_study.py --dataset halfmnist --no_entropy True
```

### 3. Denoising Ablation

Test if the w-denoising loss is essential.

```bash
# Run denoising ablation
python ablation_study.py --experiment denoising

# Run without denoising
python ablation_study.py --dataset halfmnist --no_denoising True
```

### 4. Full Ablation Suite

Run all ablations:
```bash
python ablation_study.py --experiment all
```

## Key Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | halfmnist | Dataset: halfmnist, permutedhalfmnist |
| `--digit_permutation` | str | identity | Permutation for permutedhalfmnist |
| `--no_entropy` | bool | False | Disable entropy regularization |
| `--no_denoising` | bool | False | Disable w-denoising loss |
| `--experiment` | str | single | Which ablation: all, permutation, entropy, denoising, single |
| `--epochs` | int | 500 | Number of training epochs |
| `--entropy_weight` | float | 1.6 | Entropy regularization weight |
| `--w_denoise_weight` | float | 0.0000015 | W-denoising loss weight |

## Expected Results

### If NeSyDM is Robust (Good Sign)
- **Permutation ablation**: OOD accuracy should remain high across all permutations
- **Entropy ablation**: OOD accuracy should drop significantly without entropy

### If NeSyDM Exploits Visual Bias (Concerning)
- **Permutation ablation**: OOD accuracy drops when using shift5 or shuffle
- This would indicate the model relies on simple visual structure of 0/1

## Files

- `ablation_study.py` - Main ablation study script
- `datasets/permuted_halfmnist.py` - Permuted HalfMNIST dataset
- `datasets/halfmnist.py` - Original HalfMNIST dataset (for comparison)

## Metrics

- **OOD Accuracy (accuracy_w)**: Accuracy on out-of-distribution digit combinations
- **Test Accuracy**: Accuracy on held-out in-distribution test set
- **ECE**: Expected Calibration Error

## Citation

This ablation study investigates claims from:
- "Independence Is Not an Issue in Neurosymbolic AI"
- "Neurosymbolic Diffusion Models"
