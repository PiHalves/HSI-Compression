# HSI-Compression: Deep Learning for Hyperspectral Image Compression

**Where deep learning meets Earth observation satellites: Compression and analysis of hyperspectral images**

By Michał Rajzer and Jakub Sanecki

## Abstract

Hyperspectral images (HSIs) provide detailed spectral information across hundreds of bands but generate significantly larger data volumes than traditional RGB images, creating challenges for storage and transmission in resource-constrained environments such as satellites. 

This thesis explores deep learning-based compression methods for HSIs, implementing and evaluating four models: LineRWKV for lossless compression, and CAE2D1D, CAE3D, and RCGDNAE for lossy compression.
The models were trained and tested on the HySpecNet11k dataset, with performance assessed using metrics such as Bits Per Pixel Per Band (BPPPB), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index Measure (SSIM).

Another important aspect of this work is the evaluation of the feasibility of these models for downstream tasks, specifically semantic segmentation, which was assessed using a separate model trained on the original, uncompressed data.
The results indicate that deep learning-based compression techniques can effectively reduce HSI data sizes while maintaining differing quality for both visual assessment and semantic segmentation tasks depending on the model with some models excelling in visual quality and others in segmentation performance.

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Usage](#usage)
  - [Training](#training)
  - [Testing/Validation](#testingvalidation)
  - [Segmentation Baseline](#segmentation-baseline)
  - [Compression Impact on Segmentation](#compression-impact-on-segmentation)
- [Configuration Files](#configuration-files)
- [Output Files](#output-files)
- [Examples](#examples)
- [Compiling the PDF](#compiling-the-pdf)

---

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HSI-Compression.git
cd HSI-Compression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the HySpecNet11k dataset and place it in your data directory.

---

## Project Structure

```
HSI-Compression/
├── Code/
│   ├── main.py                      # Main entry point for all operations
│   ├── config/
│   │   └── lineRWKV_config.json     # LineRWKV model configuration
│   ├── lossless/
│   │   ├── lineRWKV.py              # LineRWKV model implementation
│   │   └── lineRWKV_trainer.py      # LineRWKV training logic
│   ├── lossy/
│   │   ├── rcae2D1D.py              # 2D+1D autoencoder
│   │   ├── rcae3D.py                # 3D autoencoder
│   │   ├── RCGDNAE.py               # Rate-Controllable GDN autoencoder
│   │   └── RCGDNAE_trainer.py       # RCGDNAE training logic
│   ├── segmentation/
│   │   ├── unet.py                  # U-Net segmentation model
│   │   └── small_seg.py             # Lightweight segmentation model
│   ├── metrics/
│   │   └── metrics.py               # Evaluation metrics
│   ├── TFDataloader/
│   │   └── TFdataloader.py          # TensorFlow data pipeline
│   └── utils/
│       ├── histogram.py             # Error histogram plotting
│       ├── util.py                  # Utility functions
│       └── validation.py            # Validation utilities
├── models/                          # Saved model weights
├── requirements.txt
└── README.md
```

---

## Supported Models

### Compression Models

1. **rcae2D1D** - 2D+1D Residual Convolutional Autoencoder
   - Lossy compression
   - Input shape: `(H, W, C)` where C=202 bands
   - Suitable for spectral-spatial compression

2. **rcae2D** - 3D Residual Convolutional Autoencoder  
   - Lossy compression
   - Input shape: `(H, W, 1, C)`
   - 3D convolutions for volumetric compression

3. **RCGDNAE** - Rate-Controllable Generalized Divisive Normalization Autoencoder
   - Lossy compression with rate control
   - Uses KLT transform for decorrelation
   - Configurable compression rate

4. **LineRWKV** - Line-based RWKV for Lossless Compression
   - Lossless compression using RWKV architecture
   - Sequential processing per spectral line
   - No information loss

### Segmentation Models

5. **UNET** - U-Net for Semantic Segmentation
   - 4-class segmentation
   - Input shape: `(128, 128, 202, 1)`

6. **small_seg** - Lightweight Segmentation Model
   - Memory-efficient alternative to U-Net
   - 8 base filters, depth=3
   - Ideal for 12GB GPU memory

---

## Usage

The main entry point is `Code/main.py`. All operations are controlled via command-line arguments.

### Training

Train a compression model:

```bash
python Code/main.py \
  --mode train \
  --model RCGDNAE \
  --split easy \
  --data_dir /path/to/HySpecNet11k \
  --output_dir ./models \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 0.001
```

Train a segmentation model:

```bash
python Code/main.py \
  --mode train \
  --model small_seg \
  --split easy \
  --data_dir /path/to/HySpecNet11k \
  --output_dir ./models \
  --epochs 50 \
  --batch_size 8
```

### Testing/Validation

Test a trained compression model and generate reconstruction error histogram:

```bash
python Code/main.py \
  --mode test \
  --model RCGDNAE \
  --split easy \
  --data_dir /path/to/HySpecNet11k \
  --checkpoint ./models/RCGDNAE_best.weights.h5 \
  --batch_size 16 \
  --histogram \
  --histogram_save_path ./output/histogram.png \
  --histogram_error_type signed
```

Save reconstructed arrays for downstream evaluation:

```bash
python Code/main.py \
  --mode test \
  --model RCGDNAE \
  --split easy \
  --data_dir /path/to/HySpecNet11k \
  --checkpoint ./models/RCGDNAE_best.weights.h5 \
  --save_arrays \
  --save_arrays_path ./output/arrays \
  --save_masks
```

### Segmentation Baseline

Evaluate segmentation performance on original (uncompressed) data:

```bash
python Code/main.py \
  --seg_baseline \
  --data_dir /path/to/HySpecNet11k \
  --split easy \
  --seg_checkpoint ./models/small_seg_best.weights.h5 \
  --seg_num_classes 4 \
  --batch_size 4 \
  --output_dir ./output
```

This computes:
- Confusion matrix
- Accuracy, Precision, Recall, F1-score
- IoU (Intersection over Union)
- Dice coefficient
- Per-class metrics
- AUC-ROC

### Compression Impact on Segmentation

Evaluate how compression affects downstream segmentation task:

```bash
python Code/main.py \
  --eval_segmentation \
  --load_arrays_path ./output/arrays/RCGDNAE_manifest.json \
  --seg_checkpoint ./models/small_seg_best.weights.h5 \
  --seg_num_classes 4 \
  --output_dir ./output \
  --seg_results_path ./output/segmentation_impact.json
```

This compares:
- Original images segmentation vs ground truth
- Reconstructed images segmentation vs ground truth  
- Degradation metrics (accuracy, F1, IoU, Dice)
- Prediction agreement between original and reconstructed

---

## Configuration Files

### LineRWKV Configuration (`Code/config/lineRWKV_config.json`)

```json
{
  "data": {
    "root_dir": "/path/to/data",
    "mode": "easy",
    "batch_size": 16,
    "data_mode": 2
  },
  "model": {
    "n_embd": 256,
    "n_layer": 12,
    "num_bands": 202
  },
  "trainer": {
    "learning_rate": 0.0001,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0
  }
}
```

### RCGDNAE Configuration (similar structure)

Model-specific parameters can be adjusted in the respective config files or via command-line arguments.

---

## Output Files

### Model Checkpoints

- `{model_name}_best.weights.h5` - Best performing model weights
- `{model_name}_epoch_{n}.weights.h5` - Per-epoch checkpoints

### Reconstruction Arrays

When using `--save_arrays`:
- `{model_name}_batch_{i}_originals.npy` - Original images
- `{model_name}_batch_{i}_reconstructed.npy` - Reconstructed images
- `{model_name}_batch_{i}_masks.npy` - Ground truth masks (if `--save_masks`)
- `{model_name}_manifest.json` - Metadata and batch index

### Evaluation Results

- `baseline_segmentation_results.json` - Segmentation metrics on original data
- `{model_name}_segmentation_vs_gt.json` - Compression impact on segmentation
- Histogram plots (PNG format)

### Metrics in JSON Output

```json
{
  "accuracy": 0.9234,
  "f1_macro": 0.8891,
  "mean_iou": 0.8123,
  "dice_coefficient": 0.8456,
  "per_class": {
    "ppv": [0.91, 0.88, 0.85, 0.90],
    "recall": [0.89, 0.87, 0.83, 0.88],
    "f1_score": [0.90, 0.875, 0.84, 0.89],
    "iou": [0.82, 0.78, 0.73, 0.80]
  }
}
```

---

## Examples

### Example 1: Train and Evaluate RCGDNAE

```bash
# Train
python Code/main.py --mode train --model RCGDNAE --split easy \
  --data_dir ./data --epochs 100 --batch_size 16

# Test with histogram
python Code/main.py --mode test --model RCGDNAE --split easy \
  --data_dir ./data \
  --checkpoint ./models/RCGDNAE_best.weights.h5 \
  --histogram --histogram_save_path ./output/rcgdnae_error.png

# Save arrays for segmentation evaluation
python Code/main.py --mode test --model RCGDNAE --split easy \
  --data_dir ./data \
  --checkpoint ./models/RCGDNAE_best.weights.h5 \
  --save_arrays --save_arrays_path ./output/rcgdnae_arrays \
  --save_masks
```

### Example 2: Full Segmentation Pipeline

```bash
# 1. Train segmentation model on original data
python Code/main.py --mode train --model small_seg --split easy \
  --data_dir ./data --epochs 50 --batch_size 8

# 2. Baseline evaluation
python Code/main.py --seg_baseline --split easy \
  --data_dir ./data \
  --seg_checkpoint ./models/small_seg_best.weights.h5

# 3. Test compression model and save arrays
python Code/main.py --mode test --model RCGDNAE --split easy \
  --data_dir ./data \
  --checkpoint ./models/RCGDNAE_best.weights.h5 \
  --save_arrays --save_arrays_path ./output/rcgdnae_arrays \
  --save_masks

# 4. Evaluate compression impact
python Code/main.py --eval_segmentation \
  --load_arrays_path ./output/rcgdnae_arrays \
  --seg_checkpoint ./models/small_seg_best.weights.h5
```

### Example 3: Compare Multiple Compression Models

```bash
# Test each model and save arrays
for MODEL in rcae2D1D RCGDNAE LineRWKV; do
  python Code/main.py --mode test --model $MODEL --split easy \
    --data_dir ./data \
    --checkpoint ./models/${MODEL}_best.weights.h5 \
    --save_arrays --save_arrays_path ./output/${MODEL}_arrays \
    --save_masks
done

# Evaluate segmentation impact for each
for MODEL in rcae2D1D RCGDNAE LineRWKV; do
  python Code/main.py --eval_segmentation \
    --load_arrays_path ./output/${MODEL}_arrays \
    --seg_checkpoint ./models/small_seg_best.weights.h5 \
    --seg_results_path ./output/${MODEL}_seg_impact.json
done
```

---

## Command-Line Arguments Reference

### Core Arguments

- `--mode`: Operation mode (`train`, `validate`, `test`)
- `--model`: Model to use (`rcae2D1D`, `rcae2D`, `RCGDNAE`, `LineRWKV`, `UNET`, `small_seg`)
- `--split`: Dataset split (`easy`, `hard`)
- `--data_dir`: Path to HySpecNet11k dataset
- `--output_dir`: Directory for outputs (default: `./output`)

### Training Arguments

- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--checkpoint`: Path to checkpoint (for resuming or testing)

### Histogram Arguments

- `--histogram`: Generate reconstruction error histogram
- `--histogram_save_path`: Path to save histogram image
- `--histogram_error_type`: Error type (`signed`, `absolute`, `squared`)
- `--histogram_bins`: Number of bins (default: 200)

### Array Saving Arguments

- `--save_arrays`: Save original/reconstructed arrays
- `--save_arrays_path`: Directory to save arrays
- `--save_masks`: Also save ground truth masks

### Segmentation Arguments

- `--seg_baseline`: Run segmentation on original data only
- `--eval_segmentation`: Evaluate compression impact on segmentation
- `--seg_checkpoint`: Path to segmentation model checkpoint
- `--seg_num_classes`: Number of classes (default: 4)
- `--seg_results_path`: Path to save results JSON
- `--load_arrays_path`: Path to saved arrays for evaluation

---

## Compiling the PDF

### Dependencies

Install the dependencies (Arch Linux example):

#### LaTeX

```bash
yay -Sy texlive
```

#### Fonts

```bash
yay -Sy ttf-ms-win11-auto
```

#### Make

```bash
yay -Sy make
```

### Compiling

```bash
make
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{rajzer2026hsi,
  title={Where deep learning meets Earth observation satellites: Compression and analysis of hyperspectral images},
  author={Rajzer, Michał and Sanecki, Jakub},
  year={2026},
  school={Your University}
}
```

---

## License

[Add your license here]

---

## Acknowledgments

This work uses the HySpecNet11k dataset for hyperspectral image analysis.

## Running the code

Preferably use

```zsh

```
