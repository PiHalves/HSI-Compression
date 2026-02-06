import argparse
import os
import json
import tensorflow as tf
import numpy as np
from TFDataloader.TFdataloader import TFHySpecNetLoader
from lossless.lineRWKV_trainer import create_line_rwkv_trainer
from lossy.RCGDNAE_trainer import create_rcgdnae_trainer
from utils.histogram import plot_reconstruction_error_histogram
from metrics.metrics import SegmentationMetrics


def validate_model(model, dataset):
    """Generic validation stub (prints average MSE over dataset)."""
    mse_metric = tf.keras.metrics.Mean()
    for batch in dataset:
        preds = model(batch, training=False)
        # Handle models that return (reconstructed, latent)
        reconstructed = preds[0] if isinstance(preds, (tuple, list)) else preds
        mse = tf.reduce_mean(tf.square(batch - reconstructed))
        mse_metric.update_state(mse)
    print(f"Validation MSE: {mse_metric.result().numpy():.6f}")


def reshape_for_model(images, model_name):
    """
    Reshape images from segmentation format (B, H, W, C, 1) to model-specific format.

    Args:
        images: Images in segmentation format (B, 128, 128, 202, 1)
        model_name: Name of the model to determine target format

    Returns:
        Reshaped images in model-specific format
    """
    # Segmentation format: (B, 128, 128, 202, 1)
    # Remove the last dimension first
    if len(images.shape) == 5 and images.shape[-1] == 1:
        images = tf.squeeze(images, axis=-1)  # (B, 128, 128, 202)

    if model_name in ['rcae2D1D', 'RCGDNAE', 'LineRWKV']:
        # These models expect (B, H, W, C) -> data_mode=2 format
        return images  # Already (B, 128, 128, 202)
    elif model_name == 'rcae2D':
        # rcae2D expects (B, H, W, 1, C) -> data_mode=1 format
        return tf.expand_dims(images, axis=3)  # (B, 128, 128, 1, 202)
    else:
        return images


def reshape_to_segmentation_format(images):
    """
    Reshape images to segmentation format (B, H, W, C, 1) for consistent storage.

    Args:
        images: Images in model-specific format

    Returns:
        Images in segmentation format (B, 128, 128, 202, 1)
    """
    # Handle different input formats
    if len(images.shape) == 4:
        # (B, H, W, C) -> add channel dim -> (B, H, W, C, 1)
        return np.expand_dims(images, axis=-1)
    elif len(images.shape) == 5 and images.shape[3] == 1:
        # (B, H, W, 1, C) rcae2D format -> (B, H, W, C, 1)
        images = np.squeeze(images, axis=3)  # (B, H, W, C)
        return np.expand_dims(images, axis=-1)  # (B, H, W, C, 1)
    elif len(images.shape) == 5 and images.shape[-1] == 1:
        # Already in segmentation format
        return images
    else:
        # Unknown format, try to make it work
        return np.expand_dims(images, axis=-1) if len(images.shape) == 4 else images


def save_reconstruction_arrays_with_masks(model, data_dir, split, batch_size, save_path, model_name='model'):
    """
    Save original and reconstructed images along with ground truth masks.
    Always uses data_mode=3 to get (images, masks) pairs and reshapes for each model.

    Args:
        model: The trained compression model
        data_dir: Root directory for the dataset
        split: Dataset split ('easy' or 'hard')
        batch_size: Batch size for loading
        save_path: Directory path to save the arrays
        model_name: Name prefix for the saved files (also determines reshape logic)

    Saves:
        - {model_name}_batch_{i}_originals.npy: Original images (in segmentation format)
        - {model_name}_batch_{i}_reconstructed.npy: Reconstructed images (in segmentation format)
        - {model_name}_batch_{i}_masks.npy: Ground truth masks
        - {model_name}_manifest.json: Manifest file listing all batch files
    """
    os.makedirs(save_path, exist_ok=True)

    # Always use data_mode=3 to get (images, masks) pairs
    print(f"Loading dataset with data_mode=3 for masks...")
    data_loader = TFHySpecNetLoader(
        root_dir=data_dir,
        mode=split,
        split='test',
        batch_size=batch_size,
        data_mode=3  # Returns (images, masks) with segmentation format
    )

    print(f"Saving reconstruction arrays with masks to: {save_path}")
    print(f"  Model: {model_name}")
    batch_count = 0
    total_images = 0
    batch_files = []

    for batch in data_loader.dataset:
        images, masks = batch
        masks_np = masks.numpy() if hasattr(masks, 'numpy') else np.array(masks)

        # Reshape images for the model
        model_input = reshape_for_model(images, model_name)

        # Get model predictions
        preds = model(model_input, training=False)
        # Handle models that return (reconstructed, latent) or just reconstructed
        reconstructed = preds[0] if isinstance(preds, (tuple, list)) else preds

        # Convert to numpy
        orig_np = images.numpy() if hasattr(images, 'numpy') else np.array(images)
        recon_np = reconstructed.numpy() if hasattr(
            reconstructed, 'numpy') else np.array(reconstructed)

        # Reshape reconstructed to segmentation format for consistent storage
        recon_np = reshape_to_segmentation_format(recon_np)

        # Save this batch
        orig_batch_path = os.path.join(
            save_path, f'{model_name}_batch_{batch_count:04d}_originals.npy')
        recon_batch_path = os.path.join(
            save_path, f'{model_name}_batch_{batch_count:04d}_reconstructed.npy')
        masks_batch_path = os.path.join(
            save_path, f'{model_name}_batch_{batch_count:04d}_masks.npy')

        np.save(orig_batch_path, orig_np)
        np.save(recon_batch_path, recon_np)
        np.save(masks_batch_path, masks_np)

        batch_info = {
            'batch_idx': batch_count,
            'originals': os.path.basename(orig_batch_path),
            'reconstructed': os.path.basename(recon_batch_path),
            'masks': os.path.basename(masks_batch_path),
            'num_images': orig_np.shape[0],
            'shape': list(orig_np.shape)
        }

        batch_files.append(batch_info)

        total_images += orig_np.shape[0]
        batch_count += 1

        if batch_count % 10 == 0:
            print(
                f"  Saved batch {batch_count} ({total_images} images so far)...")

    # Save manifest file with metadata
    manifest = {
        'model_name': model_name,
        'total_batches': batch_count,
        'total_images': total_images,
        'has_masks': True,
        'batches': batch_files
    }
    manifest_path = os.path.join(save_path, f'{model_name}_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(
        f"\nSaved {total_images} images with masks across {batch_count} batches")
    print(f"Manifest saved to: {manifest_path}")

    return manifest


def save_reconstruction_arrays(model, dataset, save_path, model_name='model', has_masks=False):
    """
    Save original and reconstructed images as numpy arrays for segmentation evaluation.
    Saves each batch incrementally to avoid memory issues with large datasets.

    Args:
        model: The trained compression model
        dataset: TensorFlow dataset with original images (or (images, masks) tuples if has_masks=True)
        save_path: Directory path to save the arrays
        model_name: Name prefix for the saved files
        has_masks: If True, dataset yields (images, masks) tuples and masks will be saved

    Saves:
        - {model_name}_batch_{i}_originals.npy: Original images for batch i
        - {model_name}_batch_{i}_reconstructed.npy: Reconstructed images for batch i
        - {model_name}_batch_{i}_masks.npy: Ground truth masks for batch i (if has_masks=True)
        - {model_name}_manifest.json: Manifest file listing all batch files
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"Saving reconstruction arrays incrementally to: {save_path}")
    if has_masks:
        print("  Including ground truth masks")
    batch_count = 0
    total_images = 0
    batch_files = []

    for batch in dataset:
        # Handle datasets with masks
        if has_masks:
            images, masks = batch
            masks_np = masks.numpy() if hasattr(masks, 'numpy') else np.array(masks)
        else:
            images = batch
            masks_np = None

        # Get model predictions
        preds = model(images, training=False)
        # Handle models that return (reconstructed, latent) or just reconstructed
        reconstructed = preds[0] if isinstance(preds, (tuple, list)) else preds

        # Convert to numpy
        orig_np = images.numpy() if hasattr(images, 'numpy') else np.array(images)
        recon_np = reconstructed.numpy() if hasattr(
            reconstructed, 'numpy') else np.array(reconstructed)

        # Save this batch immediately
        orig_batch_path = os.path.join(
            save_path, f'{model_name}_batch_{batch_count:04d}_originals.npy')
        recon_batch_path = os.path.join(
            save_path, f'{model_name}_batch_{batch_count:04d}_reconstructed.npy')

        np.save(orig_batch_path, orig_np)
        np.save(recon_batch_path, recon_np)

        batch_info = {
            'batch_idx': batch_count,
            'originals': os.path.basename(orig_batch_path),
            'reconstructed': os.path.basename(recon_batch_path),
            'num_images': orig_np.shape[0],
            'shape': list(orig_np.shape)
        }

        # Save masks if available
        if has_masks and masks_np is not None:
            masks_batch_path = os.path.join(
                save_path, f'{model_name}_batch_{batch_count:04d}_masks.npy')
            np.save(masks_batch_path, masks_np)
            batch_info['masks'] = os.path.basename(masks_batch_path)

        batch_files.append(batch_info)

        total_images += orig_np.shape[0]
        batch_count += 1

        if batch_count % 10 == 0:
            print(
                f"  Saved batch {batch_count} ({total_images} images so far)...")

    # Save manifest file with metadata
    manifest = {
        'model_name': model_name,
        'total_batches': batch_count,
        'total_images': total_images,
        'has_masks': has_masks,
        'batches': batch_files
    }
    manifest_path = os.path.join(save_path, f'{model_name}_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {total_images} images across {batch_count} batches")
    print(f"Manifest saved to: {manifest_path}")

    return manifest


def load_saved_arrays(arrays_path):
    """
    Load original and reconstructed arrays from file.
    Supports both legacy single-file format and new incremental batch format.

    Args:
        arrays_path: Path to .npz file, manifest .json file, or directory containing batch files

    Returns:
        Tuple of (originals, reconstructed) numpy arrays
    """
    if arrays_path.endswith('.npz'):
        data = np.load(arrays_path)
        originals = data['originals']
        reconstructed = data['reconstructed']
    elif arrays_path.endswith('.json'):
        # Load from manifest file (new incremental format)
        with open(arrays_path, 'r') as f:
            manifest = json.load(f)

        base_dir = os.path.dirname(arrays_path)
        originals_list = []
        reconstructed_list = []

        print(f"Loading {manifest['total_batches']} batches from manifest...")
        for batch_info in manifest['batches']:
            orig_path = os.path.join(base_dir, batch_info['originals'])
            recon_path = os.path.join(base_dir, batch_info['reconstructed'])
            originals_list.append(np.load(orig_path))
            reconstructed_list.append(np.load(recon_path))

        originals = np.concatenate(originals_list, axis=0)
        reconstructed = np.concatenate(reconstructed_list, axis=0)
    else:
        # Assume directory - check for manifest first, then legacy format
        manifest_files = [f for f in os.listdir(
            arrays_path) if f.endswith('_manifest.json')]

        if manifest_files:
            # Use manifest to load batches
            manifest_path = os.path.join(arrays_path, manifest_files[0])
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            originals_list = []
            reconstructed_list = []

            print(
                f"Loading {manifest['total_batches']} batches from manifest...")
            for batch_info in manifest['batches']:
                orig_path = os.path.join(arrays_path, batch_info['originals'])
                recon_path = os.path.join(
                    arrays_path, batch_info['reconstructed'])
                originals_list.append(np.load(orig_path))
                reconstructed_list.append(np.load(recon_path))

            originals = np.concatenate(originals_list, axis=0)
            reconstructed = np.concatenate(reconstructed_list, axis=0)
        else:
            # Legacy format - single .npy files
            files = os.listdir(arrays_path)
            orig_file = [f for f in files if 'original' in f.lower()
                         and f.endswith('.npy') and 'batch' not in f.lower()]
            recon_file = [f for f in files if 'reconstruct' in f.lower()
                          and f.endswith('.npy') and 'batch' not in f.lower()]

            if orig_file and recon_file:
                originals = np.load(os.path.join(arrays_path, orig_file[0]))
                reconstructed = np.load(
                    os.path.join(arrays_path, recon_file[0]))
            else:
                raise ValueError(
                    f"Could not find original/reconstructed arrays in {arrays_path}")

    print(f"Loaded arrays:")
    print(f"  Originals shape: {originals.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")

    return originals, reconstructed


def iterate_saved_arrays(arrays_path, include_masks=False):
    """
    Generator that yields (originals, reconstructed) or (originals, reconstructed, masks) batch by batch.
    Memory-efficient incremental loading for large datasets.

    Args:
        arrays_path: Path to .npz file, manifest .json file, or directory containing batch files
        include_masks: If True and masks are available, yields (orig, recon, masks) tuples

    Yields:
        Tuple of (originals_batch, reconstructed_batch) or (originals_batch, reconstructed_batch, masks_batch)
    """
    if arrays_path.endswith('.npz'):
        # Legacy format - load all at once (can't incrementally load .npz)
        data = np.load(arrays_path)
        if include_masks and 'masks' in data:
            yield data['originals'], data['reconstructed'], data['masks']
        else:
            yield data['originals'], data['reconstructed']
    elif arrays_path.endswith('.json'):
        # Load from manifest file (new incremental format)
        with open(arrays_path, 'r') as f:
            manifest = json.load(f)

        base_dir = os.path.dirname(arrays_path)
        has_masks = manifest.get('has_masks', False)
        print(
            f"Iterating through {manifest['total_batches']} batches from manifest...")

        for i, batch_info in enumerate(manifest['batches']):
            orig_path = os.path.join(base_dir, batch_info['originals'])
            recon_path = os.path.join(base_dir, batch_info['reconstructed'])

            if include_masks and has_masks and 'masks' in batch_info:
                masks_path = os.path.join(base_dir, batch_info['masks'])
                yield np.load(orig_path), np.load(recon_path), np.load(masks_path)
            else:
                yield np.load(orig_path), np.load(recon_path)
    else:
        # Assume directory - check for manifest first, then legacy format
        manifest_files = [f for f in os.listdir(
            arrays_path) if f.endswith('_manifest.json')]

        if manifest_files:
            # Use manifest to load batches incrementally
            manifest_path = os.path.join(arrays_path, manifest_files[0])
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            has_masks = manifest.get('has_masks', False)
            print(
                f"Iterating through {manifest['total_batches']} batches from manifest...")

            for i, batch_info in enumerate(manifest['batches']):
                orig_path = os.path.join(arrays_path, batch_info['originals'])
                recon_path = os.path.join(
                    arrays_path, batch_info['reconstructed'])

                if include_masks and has_masks and 'masks' in batch_info:
                    masks_path = os.path.join(arrays_path, batch_info['masks'])
                    yield np.load(orig_path), np.load(recon_path), np.load(masks_path)
                else:
                    yield np.load(orig_path), np.load(recon_path)
        else:
            # Legacy format - single .npy files, load all at once
            files = os.listdir(arrays_path)
            orig_file = [f for f in files if 'original' in f.lower()
                         and f.endswith('.npy') and 'batch' not in f.lower()]
            recon_file = [f for f in files if 'reconstruct' in f.lower()
                          and f.endswith('.npy') and 'batch' not in f.lower()]

            if orig_file and recon_file:
                originals = np.load(os.path.join(arrays_path, orig_file[0]))
                reconstructed = np.load(
                    os.path.join(arrays_path, recon_file[0]))
                yield originals, reconstructed
            else:
                raise ValueError(
                    f"Could not find original/reconstructed arrays in {arrays_path}")


def get_manifest_info(arrays_path):
    """
    Get metadata about saved arrays without loading them.

    Args:
        arrays_path: Path to arrays directory or manifest file

    Returns:
        Dict with 'total_batches', 'total_images', 'model_name'
    """
    if arrays_path.endswith('.json'):
        with open(arrays_path, 'r') as f:
            manifest = json.load(f)
        model_name = manifest.get('model_name', os.path.basename(
            arrays_path).replace('_manifest.json', ''))
        return {
            'total_batches': manifest['total_batches'],
            'total_images': manifest['total_images'],
            'model_name': model_name
        }
    elif arrays_path.endswith('.npz'):
        data = np.load(arrays_path)
        return {
            'total_batches': 1,
            'total_images': data['originals'].shape[0],
            'model_name': os.path.basename(arrays_path).replace('_pairs.npz', '').replace('.npz', '')
        }
    else:
        manifest_files = [f for f in os.listdir(
            arrays_path) if f.endswith('_manifest.json')]
        if manifest_files:
            manifest_path = os.path.join(arrays_path, manifest_files[0])
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            model_name = manifest.get(
                'model_name', manifest_files[0].replace('_manifest.json', ''))
            return {
                'total_batches': manifest['total_batches'],
                'total_images': manifest['total_images'],
                'model_name': model_name,
                'has_masks': manifest.get('has_masks', False)
            }
        else:
            return {
                'total_batches': 1,
                'total_images': None,
                'model_name': os.path.basename(arrays_path.rstrip('/')),
                'has_masks': False
            }


def prepare_for_segmentation(images):
    """
    Prepare images for segmentation model input.

    The segmentation model expects input shape (batch, H, W, D, 1).
    Compression models may output (batch, H, W, C) or (batch, H, W, D, C).

    Args:
        images: Input images array

    Returns:
        Images reshaped for segmentation model
    """
    if len(images.shape) == 4:
        # Shape is (N, H, W, C) - need to add channel dimension
        images = np.expand_dims(images, axis=-1)  # (N, H, W, D, 1)
    return images


def run_segmentation_on_arrays(model, images, batch_size=4):
    """
    Run segmentation model on images.

    Args:
        model: Segmentation model
        images: Input images (N, H, W, D, 1)
        batch_size: Batch size for inference

    Returns:
        Tuple of (predictions, probabilities)
    """
    n_images = images.shape[0]
    all_probs = []

    print(f"Running segmentation on {n_images} images...")

    for i in range(0, n_images, batch_size):
        batch = images[i:i+batch_size]
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        probs = model(batch_tensor, training=False)
        all_probs.append(probs.numpy())

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{n_images} images...")

    probabilities = np.concatenate(all_probs, axis=0)
    predictions = np.argmax(probabilities, axis=-1)

    return predictions, probabilities


def evaluate_segmentation_impact(orig_preds, orig_probs, recon_preds, recon_probs,
                                 num_classes=4, output_path=None):
    """
    Compute metrics comparing segmentation on original vs reconstructed images.

    Args:
        orig_preds: Predictions from original images (N, H, W)
        orig_probs: Probabilities from original images (N, H, W, num_classes)
        recon_preds: Predictions from reconstructed images (N, H, W)
        recon_probs: Probabilities from reconstructed images (N, H, W, num_classes)
        num_classes: Number of segmentation classes
        output_path: Optional path to save results JSON

    Returns:
        Dictionary with comparison metrics
    """
    metrics = {}

    orig_flat = orig_preds.flatten()
    recon_flat = recon_preds.flatten()

    print("\n" + "="*60)
    print("SEGMENTATION COMPARISON METRICS")
    print("="*60)

    # Prediction agreement
    print("\n--- Prediction Agreement ---")
    agreement = np.mean(orig_flat == recon_flat)
    metrics['prediction_agreement'] = float(agreement)
    print(
        f"Overall prediction agreement: {agreement:.4f} ({agreement*100:.2f}%)")

    # Confusion matrix
    print("\n--- Using Original Predictions as Reference ---")
    cm = SegmentationMetrics.confusion_matrix(
        orig_flat, recon_flat, num_classes)
    metrics['confusion_matrix'] = cm.tolist()
    print(f"Confusion Matrix:\n{cm}")

    # Core metrics
    accuracy = SegmentationMetrics.accuracy(orig_flat, recon_flat)
    metrics['accuracy'] = float(accuracy)
    print(f"Accuracy: {accuracy:.4f}")

    ppv_macro = SegmentationMetrics.ppv(orig_flat, recon_flat, average='macro')
    ppv_weighted = SegmentationMetrics.ppv(
        orig_flat, recon_flat, average='weighted')
    metrics['ppv_macro'] = float(ppv_macro)
    metrics['ppv_weighted'] = float(ppv_weighted)
    print(
        f"PPV (Precision) - Macro: {ppv_macro:.4f}, Weighted: {ppv_weighted:.4f}")

    recall_macro = SegmentationMetrics.recall(
        orig_flat, recon_flat, average='macro')
    recall_weighted = SegmentationMetrics.recall(
        orig_flat, recon_flat, average='weighted')
    metrics['recall_macro'] = float(recall_macro)
    metrics['recall_weighted'] = float(recall_weighted)
    print(
        f"Recall - Macro: {recall_macro:.4f}, Weighted: {recall_weighted:.4f}")

    f1_macro = SegmentationMetrics.f1_score(
        orig_flat, recon_flat, average='macro')
    f1_weighted = SegmentationMetrics.f1_score(
        orig_flat, recon_flat, average='weighted')
    metrics['f1_macro'] = float(f1_macro)
    metrics['f1_weighted'] = float(f1_weighted)
    print(f"F1-score - Macro: {f1_macro:.4f}, Weighted: {f1_weighted:.4f}")

    iou_macro = SegmentationMetrics.iou(orig_flat, recon_flat, average='macro')
    metrics['iou_macro'] = float(iou_macro)
    print(f"IoU (Jaccard) - Macro: {iou_macro:.4f}")

    miou = SegmentationMetrics.mean_iou(orig_flat, recon_flat)
    metrics['mean_iou'] = float(miou)
    print(f"Mean IoU: {miou:.4f}")

    dice = SegmentationMetrics.dice_coefficient(
        orig_flat, recon_flat, average='macro')
    metrics['dice_coefficient'] = float(dice)
    print(f"Dice Coefficient: {dice:.4f}")

    # Per-class metrics
    print("\n--- Per-Class Metrics ---")
    per_class = SegmentationMetrics.per_class_metrics(
        orig_flat, recon_flat, num_classes)
    metrics['per_class'] = {
        'ppv': [float(v) for v in per_class['ppv']],
        'recall': [float(v) for v in per_class['recall']],
        'f1_score': [float(v) for v in per_class['f1_score']],
        'iou': [float(v) for v in per_class['iou']],
        'support': [int(v) for v in per_class['support']]
    }

    for c in range(num_classes):
        print(f"  Class {c}: PPV={per_class['ppv'][c]:.4f}, "
              f"Recall={per_class['recall'][c]:.4f}, "
              f"F1={per_class['f1_score'][c]:.4f}, "
              f"IoU={per_class['iou'][c]:.4f}, "
              f"Support={per_class['support'][c]}")

    # AUC-ROC
    print("\n--- AUC-ROC ---")
    try:
        auc = SegmentationMetrics.auc_roc(
            orig_flat, recon_probs.reshape(-1, recon_probs.shape[-1]))
        metrics['auc_roc'] = float(auc)
        print(f"AUC-ROC (macro): {auc:.4f}")
    except Exception as e:
        print(f"Could not compute AUC-ROC: {e}")
        metrics['auc_roc'] = None

    # Probability analysis
    print("\n--- Probability Analysis ---")
    prob_diff = np.abs(orig_probs - recon_probs)
    mean_prob_diff = np.mean(prob_diff)
    max_prob_diff = np.max(prob_diff)
    metrics['mean_probability_difference'] = float(mean_prob_diff)
    metrics['max_probability_difference'] = float(max_prob_diff)
    print(f"Mean probability difference: {mean_prob_diff:.6f}")
    print(f"Max probability difference: {max_prob_diff:.6f}")

    # Per-image analysis
    print("\n--- Per-Image Analysis ---")
    n_images = orig_preds.shape[0]
    per_image_agreement = []
    for i in range(n_images):
        img_agreement = np.mean(orig_preds[i] == recon_preds[i])
        per_image_agreement.append(img_agreement)

    per_image_agreement = np.array(per_image_agreement)
    metrics['per_image_agreement_mean'] = float(np.mean(per_image_agreement))
    metrics['per_image_agreement_std'] = float(np.std(per_image_agreement))
    metrics['per_image_agreement_min'] = float(np.min(per_image_agreement))
    metrics['per_image_agreement_max'] = float(np.max(per_image_agreement))

    print(f"Per-image agreement - Mean: {np.mean(per_image_agreement):.4f}, "
          f"Std: {np.std(per_image_agreement):.4f}")
    print(
        f"  Min: {np.min(per_image_agreement):.4f}, Max: {np.max(per_image_agreement):.4f}")

    worst_idx = np.argmin(per_image_agreement)
    best_idx = np.argmax(per_image_agreement)
    metrics['worst_image_idx'] = int(worst_idx)
    metrics['best_image_idx'] = int(best_idx)
    print(
        f"  Worst image: idx={worst_idx}, agreement={per_image_agreement[worst_idx]:.4f}")
    print(
        f"  Best image: idx={best_idx}, agreement={per_image_agreement[best_idx]:.4f}")

    # Save results if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(
            output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return metrics


def load_config(filepath):
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_latest_model_weights(models_dir):
    """Find best or latest epoch weights in a models directory."""
    try:
        best_path = os.path.join(models_dir, 'lineRWKV_best.weights.h5')
        if os.path.exists(best_path):
            return best_path

        weight_files = [
            f for f in os.listdir(models_dir)
            if f.startswith('lineRWKV_epoch_') and f.endswith('.weights.h5')
        ]
        if weight_files:
            weight_files.sort(
                key=lambda x: int(x.split('_')[2].split('.')[0])
            )
            return os.path.join(models_dir, weight_files[-1])
        return None
    except Exception:
        return None


def find_latest_rcgdnae_weights(models_dir):
    """Find best or latest epoch weights for RCGDNAE in a models directory."""
    try:
        best_path = os.path.join(models_dir, 'RCGDNAE_best.weights.h5')
        if os.path.exists(best_path):
            return best_path

        weight_files = [
            f for f in os.listdir(models_dir)
            if f.startswith('RCGDNAE_epoch_') and f.endswith('.weights.h5')
        ]
        if weight_files:
            weight_files.sort(
                key=lambda x: int(x.split('_')[2].split('.')[0])
            )
            return os.path.join(models_dir, weight_files[-1])
        return None
    except Exception:
        return None


def main(args):
    # Handle baseline segmentation on original data (no compression comparison)
    if args.seg_baseline:
        from segmentation.small_seg import small_segmenter

        tf.config.run_functions_eagerly(True)
        print("Note: Running in eager mode for inference")

        print("\n" + "="*60)
        print("BASELINE SEGMENTATION ON ORIGINAL DATA")
        print("="*60)

        # Load dataset
        print("\nLoading dataset...")
        data_loader = TFHySpecNetLoader(
            root_dir=args.data_dir,
            mode=args.split if args.split else 'easy',
            split='test',
            batch_size=args.batch_size,
            # Returns (image, mask) pairs with correct shape for segmentation
            data_mode=3
        )

        # Load segmentation model
        print("\nLoading segmentation model...")
        seg_model = small_segmenter(
            input_shape=(128, 128, 202, 1),
            num_classes=args.seg_num_classes,
            base_filters=8,
            depth=3,
            dropout_rate=0.1
        )

        # Build model
        dummy_input = tf.random.normal([1, 128, 128, 202, 1])
        _ = seg_model(dummy_input, training=False)

        # Load weights
        seg_model.load_weights(args.seg_checkpoint)
        print(f"Loaded segmentation weights from: {args.seg_checkpoint}")

        # Run segmentation on all images
        print("\nRunning segmentation on original images...")
        all_preds = []
        all_probs = []
        all_masks = []
        batch_idx = 0
        total_images = 0

        for batch in data_loader.dataset:
            batch_idx += 1
            # data_mode=3 returns (images, masks) tuple
            images, masks = batch
            batch_np = images.numpy() if hasattr(images, 'numpy') else np.array(images)
            masks_np = masks.numpy() if hasattr(masks, 'numpy') else np.array(masks)

            batch_tensor = tf.convert_to_tensor(batch_np, dtype=tf.float32)
            probs = seg_model(batch_tensor, training=False).numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.append(preds)
            all_probs.append(probs)
            all_masks.append(masks_np)
            total_images += preds.shape[0]

            if batch_idx % 10 == 0:
                print(
                    f"  Processed {batch_idx} batches ({total_images} images)...", end='\r')
        print(
            f"{'='*20} Processing complete: {total_images} images segmented. {'='*20}")

        # Concatenate results
        predictions = np.concatenate(all_preds, axis=0)
        probabilities = np.concatenate(all_probs, axis=0)
        ground_truth = np.concatenate(all_masks, axis=0)

        print(f"\nTotal images segmented: {predictions.shape[0]}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
        print(f"Probabilities shape: {probabilities.shape}")

        # Flatten for metrics computation
        preds_flat = predictions.flatten()
        gt_flat = ground_truth.flatten().astype(np.int32)

        # Compute all segmentation metrics
        metrics = {}

        print("\n" + "="*60)
        print("BASELINE SEGMENTATION METRICS (Predictions vs Ground Truth)")
        print("="*60)

        # Confusion matrix
        print("\n--- Confusion Matrix ---")
        cm = SegmentationMetrics.confusion_matrix(
            gt_flat, preds_flat, args.seg_num_classes)
        metrics['confusion_matrix'] = cm.tolist()
        print(f"Confusion Matrix:\n{cm}")

        # Core metrics
        print("\n--- Core Metrics ---")
        accuracy = SegmentationMetrics.accuracy(gt_flat, preds_flat)
        metrics['accuracy'] = float(accuracy)
        print(f"Accuracy: {accuracy:.4f}")

        ppv_macro = SegmentationMetrics.ppv(
            gt_flat, preds_flat, average='macro')
        ppv_weighted = SegmentationMetrics.ppv(
            gt_flat, preds_flat, average='weighted')
        metrics['ppv_macro'] = float(ppv_macro)
        metrics['ppv_weighted'] = float(ppv_weighted)
        print(
            f"PPV (Precision) - Macro: {ppv_macro:.4f}, Weighted: {ppv_weighted:.4f}")

        recall_macro = SegmentationMetrics.recall(
            gt_flat, preds_flat, average='macro')
        recall_weighted = SegmentationMetrics.recall(
            gt_flat, preds_flat, average='weighted')
        metrics['recall_macro'] = float(recall_macro)
        metrics['recall_weighted'] = float(recall_weighted)
        print(
            f"Recall - Macro: {recall_macro:.4f}, Weighted: {recall_weighted:.4f}")

        f1_macro = SegmentationMetrics.f1_score(
            gt_flat, preds_flat, average='macro')
        f1_weighted = SegmentationMetrics.f1_score(
            gt_flat, preds_flat, average='weighted')
        metrics['f1_macro'] = float(f1_macro)
        metrics['f1_weighted'] = float(f1_weighted)
        print(f"F1-score - Macro: {f1_macro:.4f}, Weighted: {f1_weighted:.4f}")

        iou_macro = SegmentationMetrics.iou(
            gt_flat, preds_flat, average='macro')
        metrics['iou_macro'] = float(iou_macro)
        print(f"IoU (Jaccard) - Macro: {iou_macro:.4f}")

        miou = SegmentationMetrics.mean_iou(gt_flat, preds_flat)
        metrics['mean_iou'] = float(miou)
        print(f"Mean IoU: {miou:.4f}")

        dice = SegmentationMetrics.dice_coefficient(
            gt_flat, preds_flat, average='macro')
        metrics['dice_coefficient'] = float(dice)
        print(f"Dice Coefficient: {dice:.4f}")

        # Per-class metrics
        print("\n--- Per-Class Metrics ---")
        per_class = SegmentationMetrics.per_class_metrics(
            gt_flat, preds_flat, args.seg_num_classes)
        metrics['per_class'] = {
            'ppv': [float(v) for v in per_class['ppv']],
            'recall': [float(v) for v in per_class['recall']],
            'f1_score': [float(v) for v in per_class['f1_score']],
            'iou': [float(v) for v in per_class['iou']],
            'support': [int(v) for v in per_class['support']]
        }

        for c in range(args.seg_num_classes):
            print(f"  Class {c}: PPV={per_class['ppv'][c]:.4f}, "
                  f"Recall={per_class['recall'][c]:.4f}, "
                  f"F1={per_class['f1_score'][c]:.4f}, "
                  f"IoU={per_class['iou'][c]:.4f}, "
                  f"Support={per_class['support'][c]}")

        # AUC-ROC
        print("\n--- AUC-ROC ---")
        try:
            auc = SegmentationMetrics.auc_roc(
                gt_flat, probabilities.reshape(-1, probabilities.shape[-1]))
            metrics['auc_roc'] = float(auc)
            print(f"AUC-ROC (macro): {auc:.4f}")
        except Exception as e:
            print(f"Could not compute AUC-ROC: {e}")
            metrics['auc_roc'] = None

        # Class distribution
        print("\n--- Class Distribution ---")
        print("Ground Truth:")
        gt_unique, gt_counts = np.unique(ground_truth, return_counts=True)
        gt_total = ground_truth.size
        metrics['gt_class_distribution'] = {}
        for cls, count in zip(gt_unique, gt_counts):
            percentage = 100 * count / gt_total
            metrics['gt_class_distribution'][int(cls)] = {
                'count': int(count), 'percentage': float(percentage)}
            print(f"  Class {cls}: {count:,} pixels ({percentage:.2f}%)")

        print("\nPredictions:")
        pred_unique, pred_counts = np.unique(predictions, return_counts=True)
        pred_total = predictions.size
        metrics['pred_class_distribution'] = {}
        for cls, count in zip(pred_unique, pred_counts):
            percentage = 100 * count / pred_total
            metrics['pred_class_distribution'][int(cls)] = {
                'count': int(count), 'percentage': float(percentage)}
            print(f"  Class {cls}: {count:,} pixels ({percentage:.2f}%)")

        # Per-image analysis
        print("\n--- Per-Image Analysis ---")
        n_images = predictions.shape[0]
        per_image_accuracy = []
        for i in range(n_images):
            img_accuracy = np.mean(predictions[i] == ground_truth[i])
            per_image_accuracy.append(img_accuracy)

        per_image_accuracy = np.array(per_image_accuracy)
        metrics['per_image_accuracy_mean'] = float(np.mean(per_image_accuracy))
        metrics['per_image_accuracy_std'] = float(np.std(per_image_accuracy))
        metrics['per_image_accuracy_min'] = float(np.min(per_image_accuracy))
        metrics['per_image_accuracy_max'] = float(np.max(per_image_accuracy))

        print(f"Per-image accuracy - Mean: {np.mean(per_image_accuracy):.4f}, "
              f"Std: {np.std(per_image_accuracy):.4f}")
        print(
            f"  Min: {np.min(per_image_accuracy):.4f}, Max: {np.max(per_image_accuracy):.4f}")

        worst_idx = np.argmin(per_image_accuracy)
        best_idx = np.argmax(per_image_accuracy)
        metrics['worst_image_idx'] = int(worst_idx)
        metrics['best_image_idx'] = int(best_idx)
        print(
            f"  Worst image: idx={worst_idx}, accuracy={per_image_accuracy[worst_idx]:.4f}")
        print(
            f"  Best image: idx={best_idx}, accuracy={per_image_accuracy[best_idx]:.4f}")

        # Confidence statistics
        print("\n--- Confidence Statistics ---")
        max_probs = np.max(probabilities, axis=-1)
        metrics['confidence_stats'] = {
            'mean': float(np.mean(max_probs)),
            'min': float(np.min(max_probs)),
            'max': float(np.max(max_probs)),
            'std': float(np.std(max_probs))
        }
        print(f"  Mean confidence: {np.mean(max_probs):.4f}")
        print(f"  Min confidence: {np.min(max_probs):.4f}")
        print(f"  Max confidence: {np.max(max_probs):.4f}")
        print(f"  Std confidence: {np.std(max_probs):.4f}")

        # Add metadata
        metrics['total_images'] = int(n_images)
        metrics['predictions_shape'] = list(predictions.shape)

        # Save results
        output_path = args.seg_results_path or os.path.join(
            args.output_dir, 'baseline_segmentation_results.json')
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(
            output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Optionally save predictions
        save_preds_path = os.path.join(
            args.output_dir, 'baseline_predictions.npy')
        np.save(save_preds_path, predictions)
        print(f"Predictions saved to: {save_preds_path}")

        print("\n" + "="*60)
        print("BASELINE SEGMENTATION COMPLETE")
        print("="*60)
        return

    # Handle standalone segmentation evaluation mode first (doesn't need mode/model/split)
    if args.eval_segmentation:
        if not args.seg_checkpoint:
            raise ValueError(
                "seg_checkpoint must be provided when --eval_segmentation is set.")
        if not args.load_arrays_path:
            raise ValueError(
                "load_arrays_path must be provided when --eval_segmentation is set.")

        from segmentation.small_seg import small_segmenter

        tf.config.run_functions_eagerly(True)
        print("Note: Running in eager mode for inference")

        print("\n" + "="*60)
        print("SEGMENTATION EVALUATION vs GROUND TRUTH (Incremental)")
        print("="*60)

        # Get manifest info without loading all data
        manifest_info = get_manifest_info(args.load_arrays_path)
        print(f"\nDataset info:")
        print(f"  Model name: {manifest_info['model_name']}")
        print(f"  Total batches: {manifest_info['total_batches']}")
        if manifest_info['total_images']:
            print(f"  Total images: {manifest_info['total_images']}")
        print(
            f"  Has ground truth masks: {manifest_info.get('has_masks', False)}")

        if not manifest_info.get('has_masks', False):
            raise ValueError(
                "Saved arrays do not contain ground truth masks. "
                "Please re-save arrays using data_mode=3 with --save_masks flag, "
                "or use --seg_baseline to run on dataset directly.")

        # Load segmentation model
        print("\nLoading segmentation model...")
        seg_model = small_segmenter(
            input_shape=(128, 128, 202, 1),
            num_classes=args.seg_num_classes,
            base_filters=8,
            depth=3,
            dropout_rate=0.1
        )

        # Build model
        dummy_input = tf.random.normal([1, 128, 128, 202, 1])
        _ = seg_model(dummy_input, training=False)

        # Load weights
        seg_model.load_weights(args.seg_checkpoint)
        print(f"Loaded segmentation weights from: {args.seg_checkpoint}")

        # Process incrementally - collect predictions batch by batch
        print("\nProcessing batches incrementally...")
        all_orig_preds = []
        all_orig_probs = []
        all_recon_preds = []
        all_recon_probs = []
        all_gt_masks = []

        batch_idx = 0
        for batch_data in iterate_saved_arrays(args.load_arrays_path, include_masks=True):
            orig_batch, recon_batch, masks_batch = batch_data
            batch_idx += 1
            print(
                f"  Processing batch {batch_idx}/{manifest_info['total_batches']}...", end='\r')

            # Prepare for segmentation
            orig_prepared = prepare_for_segmentation(orig_batch)
            recon_prepared = prepare_for_segmentation(recon_batch)

            # Run segmentation on this batch
            orig_tensor = tf.convert_to_tensor(orig_prepared, dtype=tf.float32)
            recon_tensor = tf.convert_to_tensor(
                recon_prepared, dtype=tf.float32)

            orig_probs = seg_model(orig_tensor, training=False).numpy()
            recon_probs = seg_model(recon_tensor, training=False).numpy()

            orig_preds = np.argmax(orig_probs, axis=-1)
            recon_preds = np.argmax(recon_probs, axis=-1)

            all_orig_preds.append(orig_preds)
            all_orig_probs.append(orig_probs)
            all_recon_preds.append(recon_preds)
            all_recon_probs.append(recon_probs)
            all_gt_masks.append(masks_batch)

            # Free memory
            del orig_batch, recon_batch, masks_batch, orig_prepared, recon_prepared
            del orig_tensor, recon_tensor

        # Concatenate all results
        print("\nConcatenating results...")
        orig_preds = np.concatenate(all_orig_preds, axis=0)
        orig_probs = np.concatenate(all_orig_probs, axis=0)
        recon_preds = np.concatenate(all_recon_preds, axis=0)
        recon_probs = np.concatenate(all_recon_probs, axis=0)
        gt_masks = np.concatenate(all_gt_masks, axis=0).astype(np.int32)

        print(f"  Total images processed: {orig_preds.shape[0]}")

        # Get model name
        model_name = manifest_info['model_name']

        # Flatten for metrics
        orig_flat = orig_preds.flatten()
        recon_flat = recon_preds.flatten()
        gt_flat = gt_masks.flatten()

        metrics = {}

        print("\n" + "="*60)
        print("ORIGINAL IMAGES vs GROUND TRUTH")
        print("="*60)

        # Metrics for original images
        print("\n--- Core Metrics (Original vs GT) ---")

        # Confusion matrix for original
        orig_cm = SegmentationMetrics.confusion_matrix(
            gt_flat, orig_flat, args.seg_num_classes)
        metrics['original_confusion_matrix'] = orig_cm.tolist()
        print(f"Confusion Matrix:\n{orig_cm}")

        orig_accuracy = SegmentationMetrics.accuracy(gt_flat, orig_flat)
        metrics['original_accuracy'] = float(orig_accuracy)
        print(f"Accuracy: {orig_accuracy:.4f}")

        orig_f1_macro = SegmentationMetrics.f1_score(
            gt_flat, orig_flat, average='macro')
        metrics['original_f1_macro'] = float(orig_f1_macro)
        print(f"F1-score (Macro): {orig_f1_macro:.4f}")

        orig_miou = SegmentationMetrics.mean_iou(gt_flat, orig_flat)
        metrics['original_mean_iou'] = float(orig_miou)
        print(f"Mean IoU: {orig_miou:.4f}")

        orig_dice = SegmentationMetrics.dice_coefficient(
            gt_flat, orig_flat, average='macro')
        metrics['original_dice'] = float(orig_dice)
        print(f"Dice Coefficient: {orig_dice:.4f}")

        # Per-class for original
        print("\n--- Per-Class Metrics (Original vs GT) ---")
        orig_per_class = SegmentationMetrics.per_class_metrics(
            gt_flat, orig_flat, args.seg_num_classes)
        metrics['original_per_class'] = {
            'ppv': [float(v) for v in orig_per_class['ppv']],
            'recall': [float(v) for v in orig_per_class['recall']],
            'f1_score': [float(v) for v in orig_per_class['f1_score']],
            'iou': [float(v) for v in orig_per_class['iou']],
            'support': [int(v) for v in orig_per_class['support']]
        }
        for c in range(args.seg_num_classes):
            print(f"  Class {c}: PPV={orig_per_class['ppv'][c]:.4f}, "
                  f"Recall={orig_per_class['recall'][c]:.4f}, "
                  f"F1={orig_per_class['f1_score'][c]:.4f}, "
                  f"IoU={orig_per_class['iou'][c]:.4f}")

        # AUC-ROC for original
        print("\n--- AUC-ROC (Original vs GT) ---")
        try:
            orig_auc = SegmentationMetrics.auc_roc(
                gt_flat, orig_probs.reshape(-1, orig_probs.shape[-1]))
            metrics['original_auc_roc'] = float(orig_auc)
            print(f"AUC-ROC (macro): {orig_auc:.4f}")
        except Exception as e:
            print(f"Could not compute AUC-ROC: {e}")
            metrics['original_auc_roc'] = None

        print("\n" + "="*60)
        print("RECONSTRUCTED IMAGES vs GROUND TRUTH")
        print("="*60)

        # Metrics for reconstructed images
        print("\n--- Core Metrics (Reconstructed vs GT) ---")

        # Confusion matrix for reconstructed
        recon_cm = SegmentationMetrics.confusion_matrix(
            gt_flat, recon_flat, args.seg_num_classes)
        metrics['reconstructed_confusion_matrix'] = recon_cm.tolist()
        print(f"Confusion Matrix:\n{recon_cm}")

        recon_accuracy = SegmentationMetrics.accuracy(gt_flat, recon_flat)
        metrics['reconstructed_accuracy'] = float(recon_accuracy)
        print(f"Accuracy: {recon_accuracy:.4f}")

        recon_f1_macro = SegmentationMetrics.f1_score(
            gt_flat, recon_flat, average='macro')
        metrics['reconstructed_f1_macro'] = float(recon_f1_macro)
        print(f"F1-score (Macro): {recon_f1_macro:.4f}")

        recon_miou = SegmentationMetrics.mean_iou(gt_flat, recon_flat)
        metrics['reconstructed_mean_iou'] = float(recon_miou)
        print(f"Mean IoU: {recon_miou:.4f}")

        recon_dice = SegmentationMetrics.dice_coefficient(
            gt_flat, recon_flat, average='macro')
        metrics['reconstructed_dice'] = float(recon_dice)
        print(f"Dice Coefficient: {recon_dice:.4f}")

        # Per-class for reconstructed
        print("\n--- Per-Class Metrics (Reconstructed vs GT) ---")
        recon_per_class = SegmentationMetrics.per_class_metrics(
            gt_flat, recon_flat, args.seg_num_classes)
        metrics['reconstructed_per_class'] = {
            'ppv': [float(v) for v in recon_per_class['ppv']],
            'recall': [float(v) for v in recon_per_class['recall']],
            'f1_score': [float(v) for v in recon_per_class['f1_score']],
            'iou': [float(v) for v in recon_per_class['iou']],
            'support': [int(v) for v in recon_per_class['support']]
        }
        for c in range(args.seg_num_classes):
            print(f"  Class {c}: PPV={recon_per_class['ppv'][c]:.4f}, "
                  f"Recall={recon_per_class['recall'][c]:.4f}, "
                  f"F1={recon_per_class['f1_score'][c]:.4f}, "
                  f"IoU={recon_per_class['iou'][c]:.4f}")

        # AUC-ROC for reconstructed
        print("\n--- AUC-ROC (Reconstructed vs GT) ---")
        try:
            recon_auc = SegmentationMetrics.auc_roc(
                gt_flat, recon_probs.reshape(-1, recon_probs.shape[-1]))
            metrics['reconstructed_auc_roc'] = float(recon_auc)
            print(f"AUC-ROC (macro): {recon_auc:.4f}")
        except Exception as e:
            print(f"Could not compute AUC-ROC: {e}")
            metrics['reconstructed_auc_roc'] = None

        print("\n" + "="*60)
        print("COMPRESSION IMPACT SUMMARY")
        print("="*60)

        # Compute degradation
        acc_diff = recon_accuracy - orig_accuracy
        f1_diff = recon_f1_macro - orig_f1_macro
        miou_diff = recon_miou - orig_miou
        dice_diff = recon_dice - orig_dice

        metrics['accuracy_degradation'] = float(acc_diff)
        metrics['f1_degradation'] = float(f1_diff)
        metrics['miou_degradation'] = float(miou_diff)
        metrics['dice_degradation'] = float(dice_diff)

        print(
            f"\nAccuracy:  Original={orig_accuracy:.4f}, Reconstructed={recon_accuracy:.4f}, Δ={acc_diff:+.4f}")
        print(
            f"F1-score:  Original={orig_f1_macro:.4f}, Reconstructed={recon_f1_macro:.4f}, Δ={f1_diff:+.4f}")
        print(
            f"Mean IoU:  Original={orig_miou:.4f}, Reconstructed={recon_miou:.4f}, Δ={miou_diff:+.4f}")
        print(
            f"Dice:      Original={orig_dice:.4f}, Reconstructed={recon_dice:.4f}, Δ={dice_diff:+.4f}")

        # AUC degradation (only if both were successfully computed)
        if metrics.get('original_auc_roc') is not None and metrics.get('reconstructed_auc_roc') is not None:
            auc_diff = metrics['reconstructed_auc_roc'] - \
                metrics['original_auc_roc']
            metrics['auc_roc_degradation'] = float(auc_diff)
            print(
                f"AUC-ROC:   Original={metrics['original_auc_roc']:.4f}, Reconstructed={metrics['reconstructed_auc_roc']:.4f}, Δ={auc_diff:+.4f}")

        # Agreement between original and reconstructed segmentation
        print("\n--- Original vs Reconstructed Segmentation Agreement ---")
        agreement = np.mean(orig_flat == recon_flat)
        metrics['orig_vs_recon_agreement'] = float(agreement)
        print(f"Prediction agreement: {agreement:.4f} ({agreement*100:.2f}%)")

        # Per-image analysis
        print("\n--- Per-Image Accuracy vs Ground Truth ---")
        n_images = orig_preds.shape[0]
        orig_per_image_acc = []
        recon_per_image_acc = []
        for i in range(n_images):
            orig_per_image_acc.append(np.mean(orig_preds[i] == gt_masks[i]))
            recon_per_image_acc.append(np.mean(recon_preds[i] == gt_masks[i]))

        orig_per_image_acc = np.array(orig_per_image_acc)
        recon_per_image_acc = np.array(recon_per_image_acc)

        metrics['original_per_image_accuracy'] = {
            'mean': float(np.mean(orig_per_image_acc)),
            'std': float(np.std(orig_per_image_acc)),
            'min': float(np.min(orig_per_image_acc)),
            'max': float(np.max(orig_per_image_acc))
        }
        metrics['reconstructed_per_image_accuracy'] = {
            'mean': float(np.mean(recon_per_image_acc)),
            'std': float(np.std(recon_per_image_acc)),
            'min': float(np.min(recon_per_image_acc)),
            'max': float(np.max(recon_per_image_acc))
        }

        print(
            f"Original - Mean: {np.mean(orig_per_image_acc):.4f}, Std: {np.std(orig_per_image_acc):.4f}")
        print(
            f"Reconstructed - Mean: {np.mean(recon_per_image_acc):.4f}, Std: {np.std(recon_per_image_acc):.4f}")

        # Add metadata
        metrics['total_images'] = int(n_images)
        metrics['model_name'] = model_name

        # Save results
        output_path = args.seg_results_path or os.path.join(
            args.output_dir, f'{model_name}_segmentation_vs_gt.json')
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(
            output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        print("\n" + "="*60)
        print("SEGMENTATION EVALUATION COMPLETE")
        print("="*60)
        return

    # Check if args are valid for non-segmentation modes
    if args.mode not in ['train', 'validate', 'test']:
        raise ValueError(
            "Invalid mode. Choose from 'train', 'validate', or 'test'.")
    if args.model not in ['rcae2D1D', 'rcae2D', 'RCGDNAE', 'LineRWKV', 'UNET', 'small_seg']:
        raise ValueError(
            "Invalid model. Choose from 'rcae2D1D', 'rcae2D', 'RCGDNAE', 'LineRWKV', 'UNET', or 'small_seg'.")
    if args.split not in ['easy', 'hard']:
        raise ValueError("Invalid split. Choose from 'easy' or 'hard'.")

    if args.mode in ['test', 'validate'] and not args.checkpoint:
        raise ValueError(
            "Checkpoint path must be provided for 'test' or 'validate' mode.")
    if args.histogram and args.mode not in ['test'] and args.histogram_error_type not in ['signed', 'absolute', 'squared']:
        raise ValueError(
            "Histogram error type must be 'signed', 'absolute', or 'squared' and mode must be 'test'.")
    if args.save_arrays and not args.save_arrays_path:
        raise ValueError(
            "save_arrays_path must be provided when --save_arrays is set.")
    if args.save_arrays and args.mode != 'test':
        raise ValueError(
            "--save_arrays can only be used with 'test' mode.")

    # Enable eager execution for validation/test mode to avoid @tf.function
    # graph caching issues after loading weights
    if args.mode in ['validate', 'test']:
        tf.config.run_functions_eagerly(True)
        print("Note: Running in eager mode for inference")
    # Helper metrics for autoencoders

    def compute_psnr(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        psnr = 20 * tf.math.log(1.0 / tf.sqrt(mse + 1e-10)) / tf.math.log(10.0)
        return psnr

    def compute_ssim(y_true, y_pred):
        rank = len(y_true.shape)
        if rank == 4:  # [b,h,w,c]
            b, h, w, c = tf.shape(y_true)[0], tf.shape(
                y_true)[1], tf.shape(y_true)[2], tf.shape(y_true)[3]
            y_true_reshaped = tf.reshape(tf.transpose(
                y_true, [0, 3, 1, 2]), [b * c, h, w, 1])
            y_pred_reshaped = tf.reshape(tf.transpose(
                y_pred, [0, 3, 1, 2]), [b * c, h, w, 1])
            ssim_vals = tf.image.ssim(
                y_true_reshaped, y_pred_reshaped, max_val=1.0)
            ssim_per_sample = tf.reshape(ssim_vals, [b, c])
            return tf.reduce_mean(ssim_per_sample)
        elif rank == 5:  # [b,h,w,d,c]
            b, h, w, d, c = tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[
                2], tf.shape(y_true)[3], tf.shape(y_true)[4]
            y_true_t = tf.transpose(y_true, [0, 4, 1, 2, 3])  # [b,c,h,w,d]
            y_pred_t = tf.transpose(y_pred, [0, 4, 1, 2, 3])
            y_true_reshaped = tf.reshape(y_true_t, [b * c * d, h, w, 1])
            y_pred_reshaped = tf.reshape(y_pred_t, [b * c * d, h, w, 1])
            ssim_vals = tf.image.ssim(
                y_true_reshaped, y_pred_reshaped, max_val=1.0)
            return tf.reduce_mean(ssim_vals)
        else:
            return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    def train_autoencoder(model, train_ds, val_ds, epochs, save_dir, prefix):
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        best_val = float('inf')
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(epochs):
            train_loss = tf.keras.metrics.Mean()
            val_loss = tf.keras.metrics.Mean()
            val_psnr = tf.keras.metrics.Mean()
            val_ssim = tf.keras.metrics.Mean()

            # Train loop
            for batch in train_ds:
                with tf.GradientTape() as tape:
                    outputs = model(batch, training=True)
                    reconstructed = outputs[0] if isinstance(
                        outputs, (tuple, list)) else outputs
                    loss = tf.reduce_mean(tf.square(batch - reconstructed))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                train_loss.update_state(loss)

            # Validation loop
            if val_ds is not None:
                for batch in val_ds:
                    outputs = model(batch, training=False)
                    reconstructed = outputs[0] if isinstance(
                        outputs, (tuple, list)) else outputs
                    loss = tf.reduce_mean(tf.square(batch - reconstructed))
                    val_loss.update_state(loss)
                    val_psnr.update_state(compute_psnr(batch, reconstructed))
                    val_ssim.update_state(compute_ssim(batch, reconstructed))

            # Save weights
            weights_path = os.path.join(
                save_dir, f"{prefix}_epoch_{epoch+1:03d}.weights.h5")
            model.save_weights(weights_path)

            # Track best
            current_val = val_loss.result().numpy(
            ) if val_ds is not None else train_loss.result().numpy()
            if current_val < best_val:
                best_val = current_val
                best_path = os.path.join(save_dir, f"{prefix}_best.weights.h5")
                model.save_weights(best_path)

            # Log
            if val_ds is not None:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.result().numpy():.6f} | "
                      f"Val Loss: {val_loss.result().numpy():.6f} | "
                      f"Val PSNR: {val_psnr.result().numpy():.2f} dB | "
                      f"Val SSIM: {val_ssim.result().numpy():.4f}")
            else:
                print(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.result().numpy():.6f}")

    def eval_autoencoder(model, eval_ds):
        loss_m = tf.keras.metrics.Mean()
        psnr_m = tf.keras.metrics.Mean()
        ssim_m = tf.keras.metrics.Mean()
        for batch in eval_ds:
            outputs = model(batch, training=False)
            reconstructed = outputs[0] if isinstance(
                outputs, (tuple, list)) else outputs
            loss = tf.reduce_mean(tf.square(batch - reconstructed))
            loss_m.update_state(loss)
            psnr_m.update_state(compute_psnr(batch, reconstructed))
            ssim_m.update_state(compute_ssim(batch, reconstructed))
        print("\n=== Autoencoder Evaluation Results ===")
        print(f"Loss: {loss_m.result().numpy():.6f}")
        print(f"PSNR: {psnr_m.result().numpy():.2f} dB")
        print(f"SSIM: {ssim_m.result().numpy():.4f}")

    if args.model == 'rcae2D1D':
        from Code.lossy.rcae2D1D import cae2D1D
        model = cae2D1D(src_channels=202, latent_channels=128)
        # Dataloaders: 2D1D expects [h,w,c] -> data_mode=2

        if args.mode == 'train':
            # Resume from checkpoint if provided
            train_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='train',
                batch_size=args.batch_size,
                data_mode=2
            )
            if args.checkpoint:
                try:
                    model.load_weights(args.checkpoint)
                    print(f"Loaded weights from: {args.checkpoint}")
                except Exception:
                    print("Warning: failed to load checkpoint; starting fresh")

            save_dir = os.path.join(args.output_dir, 'models')
            train_autoencoder(model, train_loader.dataset,
                              val_loader.dataset, args.epochs, save_dir, 'rcae2D1D')

        elif args.mode in ['validate', 'test']:
            val_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='test',
                batch_size=args.batch_size,
                data_mode=2
            )
            if not args.checkpoint:
                raise ValueError("Checkpoint is required for validation/test")
            model = tf.keras.models.load_model(args.checkpoint)
            if args.mode == 'test' and args.save_arrays:
                # Save original and reconstructed arrays for segmentation evaluation
                print("Saving reconstruction arrays for segmentation evaluation...")
                if args.save_masks:
                    # Use data_mode=3 to get masks, reshape for model
                    save_reconstruction_arrays_with_masks(
                        model=model,
                        data_dir=args.data_dir,
                        split=args.split,
                        batch_size=args.batch_size,
                        save_path=args.save_arrays_path,
                        model_name='rcae2D1D'
                    )
                else:
                    save_reconstruction_arrays(
                        model=model,
                        dataset=val_loader.dataset,
                        save_path=args.save_arrays_path,
                        model_name='rcae2D1D',
                        has_masks=False
                    )
            elif args.mode == 'test' and args.histogram:
                # Plot reconstruction error histogram
                print("Plotting reconstruction error histogram...")
                plot_reconstruction_error_histogram(
                    model=model,
                    dataset=val_loader.dataset,
                    bins=args.histogram_bins,
                    error_type=args.histogram_error_type,
                    title="rcae2D1D Reconstruction Error Histogram",
                    show=True,
                    save_path=args.histogram_save_path
                )
            else:
                eval_autoencoder(model, val_loader.dataset)
    elif args.model == 'rcae2D':
        from Code.lossy.rcae3D import ResidualConv3DAutoencoder
        model = ResidualConv3DAutoencoder(
            src_channels=202, latent_channels=128)
        print(args.histogram_save_path)
        # rcae2D expects [h,w,1,c] -> data_mode=1

        if args.mode == 'train':
            train_loader = TFHySpecNetLoader(

                root_dir=args.data_dir,
                mode=args.split,
                split='train',
                batch_size=args.batch_size,
                data_mode=1
            )
            if args.checkpoint:
                try:
                    model.load_weights(args.checkpoint)
                    print(f"Loaded weights from: {args.checkpoint}")
                except Exception:
                    print("Warning: failed to load checkpoint; starting fresh")
            save_dir = os.path.join(args.output_dir, 'models')
            train_autoencoder(model, train_loader.dataset,
                              val_loader.dataset, args.epochs, save_dir, 'rcae2D')

        elif args.mode in ['validate', 'test']:
            val_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='test',
                batch_size=args.batch_size,
                data_mode=1
            )
            if not args.checkpoint:
                raise ValueError("Checkpoint is required for validation/test")
            model = tf.keras.models.load_model(args.checkpoint)
            if args.mode == 'test' and args.save_arrays:
                # Save original and reconstructed arrays for segmentation evaluation
                print("Saving reconstruction arrays for segmentation evaluation...")
                if args.save_masks:
                    # Use data_mode=3 to get masks, reshape for model
                    save_reconstruction_arrays_with_masks(
                        model=model,
                        data_dir=args.data_dir,
                        split=args.split,
                        batch_size=args.batch_size,
                        save_path=args.save_arrays_path,
                        model_name='rcae2D'
                    )
                else:
                    save_reconstruction_arrays(
                        model=model,
                        dataset=val_loader.dataset,
                        save_path=args.save_arrays_path,
                        model_name='rcae2D',
                        has_masks=False
                    )
            elif args.mode == 'test' and args.histogram:
                # Plot reconstruction error histogram
                print("Plotting reconstruction error histogram...")
                plot_reconstruction_error_histogram(
                    model=model,
                    dataset=val_loader.dataset,
                    bins=args.histogram_bins,
                    error_type=args.histogram_error_type,
                    title="rcae2D Reconstruction Error Histogram",
                    show=True,
                    save_path=args.histogram_save_path
                )
            else:
                eval_autoencoder(model, val_loader.dataset)

    elif args.model == 'RCGDNAE':
        # Load configuration file for RCGDNAE (fallback to defaults)
        try:
            config = load_config(args.config)
        except Exception:
            config = {
                'data': {
                    'root_dir': args.data_dir,
                    'mode': args.split,
                    'num_parallel_calls': 32,
                    'prefetch_buffer': 32,
                    'data_mode': 2,
                    'batch_size': args.batch_size
                },
                'model': {
                    'img_height': 128,
                    'img_width': 128,
                    'bands': 202
                },
                'trainer': {
                    'learning_rate': args.learning_rate,
                    'weight_decay': 1e-5,
                    'lambda_rd': 0.01,
                    'use_mixed_precision': False,
                    'enable_performance_optimizations': True
                },
                'training': {
                    'epochs': args.epochs
                }
            }

        # Ensure CLI overrides are applied
        config['data']['root_dir'] = args.data_dir
        config['data']['mode'] = args.split
        config['data']['batch_size'] = args.batch_size
        config['trainer']['learning_rate'] = args.learning_rate
        if 'training' in config:
            config['training']['epochs'] = args.epochs

        # Create train/val loaders
        train_loader = TFHySpecNetLoader(
            root_dir=config['data']['root_dir'],
            mode=config['data']['mode'],
            split='train',
            batch_size=config['data'].get('batch_size', args.batch_size),
            num_parallel_calls=config['data'].get('num_parallel_calls', 32),
            prefetch_buffer=config['data'].get('prefetch_buffer', 32),
            data_mode=config['data'].get('data_mode', 2)
        )
        val_loader = TFHySpecNetLoader(
            root_dir=config['data']['root_dir'],
            mode=config['data']['mode'],
            split='val',
            batch_size=config['data'].get('batch_size', args.batch_size),
            num_parallel_calls=config['data'].get('num_parallel_calls', 32),
            prefetch_buffer=config['data'].get('prefetch_buffer', 32),
            data_mode=config['data'].get('data_mode', 2)
        )

        # Create trainer
        trainer = create_rcgdnae_trainer(
            dataloader_config=config['data'],
            model_config=config.get('model', {}),
            trainer_config=config.get('trainer', {}),
            klt_path=config.get('model', {}).get('klt_matrix_path')
        )

        # Training
        if args.mode == 'train':
            save_dir = os.path.join(args.output_dir, 'models')
            os.makedirs(save_dir, exist_ok=True)

            # Resume from checkpoint if provided
            if args.checkpoint:
                ckpt = args.checkpoint
                if os.path.isdir(ckpt):
                    # Try loading a SavedModel directory
                    try:
                        trainer.load_model(ckpt)
                        print(f"Loaded SavedModel from: {ckpt}")
                    except Exception:
                        # Fallback: look for weights in 'models' subdir
                        maybe_models = os.path.join(ckpt, 'models')
                        weights_path = find_latest_rcgdnae_weights(
                            maybe_models)
                        if weights_path and os.path.isfile(weights_path):
                            trainer.model.load_weights(weights_path)
                            print(f"Loaded weights from: {weights_path}")
                        else:
                            print(
                                "Warning: No valid checkpoint found; starting fresh")
                elif os.path.isfile(ckpt):
                    # Load direct weights file
                    try:
                        trainer.model.load_weights(ckpt)
                        print(f"Loaded weights from: {ckpt}")
                    except Exception:
                        print(
                            "Warning: Failed to load provided checkpoint, starting fresh")

            history = trainer.fit(
                train_dataset=train_loader,
                val_dataset=val_loader,
                epochs=args.epochs,
                save_path=save_dir,
                save_frequency=1,
                verbose=True
            )

        # Validation/Test
        elif args.mode in ['validate', 'test']:
            eval_loader = TFHySpecNetLoader(
                root_dir=config['data']['root_dir'],
                mode=config['data']['mode'],
                split='test',
                batch_size=config['data'].get('batch_size', args.batch_size),
                num_parallel_calls=config['data'].get(
                    'num_parallel_calls', 32),
                prefetch_buffer=config['data'].get('prefetch_buffer', 32),
                data_mode=config['data'].get('data_mode', 2)
            )

            ckpt = args.checkpoint
            if os.path.isdir(ckpt):
                # Try SavedModel first
                try:
                    trainer.load_model(ckpt)
                    print(f"Loaded SavedModel from: {ckpt}")
                except Exception:
                    # Fallback to weights in models subdir
                    maybe_models = os.path.join(ckpt, 'models')
                    weights_path = find_latest_rcgdnae_weights(maybe_models)
                    if weights_path and os.path.isfile(weights_path):
                        # Build once before loading weights
                        dummy = tf.zeros((1,
                                          config.get('model', {}).get(
                                              'img_height', 128),
                                          config.get('model', {}).get(
                                              'img_width', 128),
                                          config.get('model', {}).get('bands', 202)))
                        _ = trainer.model(dummy, training=False)
                        trainer.model.load_weights(weights_path)
                        print(f"Loaded weights for evaluation: {weights_path}")
                    else:
                        raise ValueError(
                            "No valid checkpoint found in directory for validation/test")
            elif os.path.isfile(ckpt):
                # Build once before loading weights
                dummy = tf.zeros((1,
                                  config.get('model', {}).get(
                                      'img_height', 128),
                                  config.get('model', {}).get(
                                      'img_width', 128),
                                  config.get('model', {}).get('bands', 202)))
                _ = trainer.model(dummy, training=False)
                trainer.model.load_weights(ckpt)
                print(f"Loaded weights for evaluation: {ckpt}")
            else:
                raise ValueError(
                    "Valid checkpoint path required for validation/test")

            # Run validation loop using trainer metrics
            trainer.reset_metrics(validation=True)
            if args.mode == 'test' and args.save_arrays:
                # Save original and reconstructed arrays for segmentation evaluation
                print("Saving reconstruction arrays for segmentation evaluation...")
                if args.save_masks:
                    # Use data_mode=3 to get masks, reshape for model
                    save_reconstruction_arrays_with_masks(
                        model=trainer.model,
                        data_dir=args.data_dir,
                        split=args.split,
                        batch_size=args.batch_size,
                        save_path=args.save_arrays_path,
                        model_name='RCGDNAE'
                    )
                else:
                    save_reconstruction_arrays(
                        model=trainer.model,
                        dataset=eval_loader.dataset,
                        save_path=args.save_arrays_path,
                        model_name='RCGDNAE',
                        has_masks=False
                    )
            elif args.mode == 'test' and args.histogram:
                # Plot reconstruction error histogram
                print("Plotting reconstruction error histogram...")
                plot_reconstruction_error_histogram(
                    model=trainer.model,
                    dataset=eval_loader.dataset,
                    bins=args.histogram_bins,
                    error_type=args.histogram_error_type,
                    title="RCGDNAE Reconstruction Error Histogram",
                    show=True,
                    save_path=args.histogram_save_path
                )
            else:
                for batch in (eval_loader.dataset if hasattr(eval_loader, 'dataset') else eval_loader):
                    trainer.val_step(batch)

                print("\n=== RCGDNAE Evaluation Results ===")
                print(f"Val Loss: {trainer.val_loss.result().numpy():.6f}")
                print(
                    f"Val Distortion: {trainer.val_distortion.result().numpy():.6f}")
                print(
                    f"Val Rate: {trainer.val_rate.result().numpy():.2f} bits")
                print(f"Val PSNR: {trainer.val_psnr.result().numpy():.2f} dB")
                print(f"Val SSIM: {trainer.val_ssim.result().numpy():.4f}")
                print(f"Val SA: {trainer.val_sa.result().numpy():.4f} rad")
    elif args.model == 'LineRWKV':
        # Load configuration file for LineRWKV (fallback to defaults)
        try:
            config = load_config(args.config)
        except Exception:
            config = {
                'data': {
                    'root_dir': args.data_dir,
                    'mode': args.split,
                    'num_parallel_calls': 32,
                    'prefetch_buffer': 32,
                    'data_mode': 2
                },
                'model': {
                    'input_channels': 202,
                    'dim': 128,
                    'num_layers': 4,
                    'time_decay': 0.99,
                    'prediction_mode': 'spectral'
                },
                'trainer': {
                    'learning_rate': args.learning_rate,
                    'weight_decay': 1e-5,
                    'loss_type': 'prediction_residual',
                    'use_mixed_precision': False
                },
                'training': {
                    'epochs': args.epochs
                }
            }

        # Ensure CLI overrides are applied
        config['data']['root_dir'] = args.data_dir
        config['data']['mode'] = args.split
        config['trainer']['learning_rate'] = args.learning_rate
        if 'training' in config:
            config['training']['epochs'] = args.epochs

        # Create train/val loaders consistent with trainer expectations
        train_loader = TFHySpecNetLoader(
            root_dir=config['data']['root_dir'],
            mode=config['data']['mode'],
            split='train',
            batch_size=args.batch_size,
            num_parallel_calls=config['data'].get('num_parallel_calls', 32),
            prefetch_buffer=config['data'].get('prefetch_buffer', 32),
            data_mode=config['data'].get('data_mode', 2)
        )
        val_loader = TFHySpecNetLoader(
            root_dir=config['data']['root_dir'],
            mode=config['data']['mode'],
            split='val',
            batch_size=args.batch_size,
            num_parallel_calls=config['data'].get('num_parallel_calls', 32),
            prefetch_buffer=config['data'].get('prefetch_buffer', 32),
            data_mode=config['data'].get('data_mode', 2)
        )

        trainer = create_line_rwkv_trainer(
            dataloader_config=config['data'],
            model_config=config.get('model', {}),
            trainer_config=config.get('trainer', {})
        )

        # Training
        if args.mode == 'train':
            save_dir = os.path.join(args.output_dir, 'models')
            os.makedirs(save_dir, exist_ok=True)

            # Build once to initialize variables before any weight loading
            dummy = tf.random.normal(
                [1, 32, 32, config.get('model', {}).get('input_channels', 202)])
            _ = trainer.model(dummy, training=False)

            # Resume from checkpoint if provided (weights file or run dir)
            if args.checkpoint:
                weights_path = args.checkpoint
                if os.path.isdir(weights_path):
                    maybe_models = os.path.join(weights_path, 'models')
                    weights_path = find_latest_model_weights(
                        maybe_models) or weights_path
                if os.path.isfile(weights_path):
                    try:
                        trainer.model.load_weights(weights_path)
                        print(f"Loaded weights from: {weights_path}")
                    except Exception:
                        print(
                            "Warning: Failed to load provided checkpoint, starting fresh")

            history = trainer.fit(
                train_loader.dataset,
                val_loader.dataset,
                epochs=args.epochs,
                save_path=save_dir,
                save_frequency=1,
                verbose=True
            )

        # Validation/Test
        elif args.mode in ['validate', 'test']:
            # Determine evaluation split ('val' is commonly used)
            eval_loader = TFHySpecNetLoader(
                root_dir=config['data']['root_dir'],
                mode=config['data']['mode'],
                split='test',
                batch_size=args.batch_size,
                num_parallel_calls=config['data'].get(
                    'num_parallel_calls', 32),
                prefetch_buffer=config['data'].get('prefetch_buffer', 32),
                data_mode=config['data'].get('data_mode', 2)
            )

            # Build and load weights
            dummy = tf.random.normal(
                [1, 32, 32, config.get('model', {}).get('input_channels', 202)])
            _ = trainer.model(dummy, training=False)

            weights_path = args.checkpoint
            if os.path.isdir(weights_path):
                maybe_models = os.path.join(weights_path, 'models')
                weights_path = find_latest_model_weights(
                    maybe_models) or weights_path

            if not (weights_path and os.path.isfile(weights_path)):
                raise ValueError(
                    "Valid checkpoint file (.h5) required for validation/test")

            trainer.model.load_weights(weights_path)
            print(f"Loaded weights for evaluation: {weights_path}")

            # Run validation loop using trainer metrics
            trainer.val_loss.reset_state()
            trainer.val_psnr.reset_state()
            trainer.val_ssim.reset_state()
            trainer.val_sa.reset_state()

            if args.mode == 'test' and args.save_arrays:
                # Save original and reconstructed arrays for segmentation evaluation
                print("Saving reconstruction arrays for segmentation evaluation...")
                if args.save_masks:
                    # Use data_mode=3 to get masks, reshape for model
                    save_reconstruction_arrays_with_masks(
                        model=trainer.model,
                        data_dir=args.data_dir,
                        split=args.split,
                        batch_size=args.batch_size,
                        save_path=args.save_arrays_path,
                        model_name='LineRWKV'
                    )
                else:
                    save_reconstruction_arrays(
                        model=trainer.model,
                        dataset=eval_loader.dataset,
                        save_path=args.save_arrays_path,
                        model_name='LineRWKV',
                        has_masks=False
                    )
            elif args.mode == 'test' and args.histogram:
                # Plot reconstruction error histogram
                print("Plotting reconstruction error histogram...")
                plot_reconstruction_error_histogram(
                    model=trainer.model,
                    dataset=eval_loader.dataset,
                    bins=args.histogram_bins,
                    error_type=args.histogram_error_type,
                    title="LineRWKV Reconstruction Error Histogram",
                    show=True,
                    save_path=args.histogram_save_path

                )
            else:
                for batch in eval_loader.dataset:
                    trainer.val_step(batch)

                print("\n=== Evaluation Results ===")
                print(f"Val Loss: {trainer.val_loss.result().numpy():.6f}")
                print(f"Val PSNR: {trainer.val_psnr.result().numpy():.2f} dB")
                print(f"Val SSIM: {trainer.val_ssim.result().numpy():.4f}")
                print(f"Val SA: {trainer.val_sa.result().numpy():.4f} rad")

    elif args.model == 'UNET':
        from segmentation.unet import UNET

        # Enable mixed precision for memory efficiency
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Using mixed precision (float16) for memory efficiency")

        # U-Net for spatial segmentation using data_mode=3
        # data_mode=3 returns (image, mask) pairs
        early_stopping_timer = 0

        def train_unet(model, train_ds, val_ds, epochs, save_dir, prefix, num_classes):
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False)
            best_val = float('inf')
            os.makedirs(save_dir, exist_ok=True)

            for epoch in range(epochs):
                train_loss = tf.keras.metrics.Mean()
                train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
                val_loss = tf.keras.metrics.Mean()
                val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
                val_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

                # Train loop
                for i, (images, masks) in enumerate(train_ds):
                    print(f"Batch {i}/{len(train_ds)}", end='\r')
                    with tf.GradientTape() as tape:
                        preds = model(images, training=True)
                        loss = loss_fn(masks, preds)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                    train_loss.update_state(loss)
                    train_acc.update_state(masks, preds)

                # Validation loop
                if val_ds is not None:
                    for i, (images, masks) in enumerate(val_ds):
                        print(f"Val batch {i}/{len(val_ds)}", end='\r')
                        preds = model(images, training=False)
                        loss = loss_fn(masks, preds)
                        val_loss.update_state(loss)
                        val_acc.update_state(masks, preds)
                        pred_classes = tf.argmax(preds, axis=-1)
                        val_iou.update_state(masks, pred_classes)

                # Save weights
                weights_path = os.path.join(
                    save_dir, f"{prefix}_epoch_{epoch+1:03d}.weights.h5")
                model.save_weights(weights_path)

                # Track best
                current_val = val_loss.result().numpy(
                ) if val_ds is not None else train_loss.result().numpy()
                if current_val < best_val:
                    best_val = current_val
                    best_path = os.path.join(
                        save_dir, f"{prefix}_best.weights.h5")
                    model.save_weights(best_path)
                else:
                    early_stopping_timer += 1
                    if early_stopping_timer >= 10:
                        print(
                            "Early stopping triggered due to no improvement in validation loss.")
                        break

                # Log
                if val_ds is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss.result().numpy():.4f} | "
                          f"Train Acc: {train_acc.result().numpy():.4f} | "
                          f"Val Loss: {val_loss.result().numpy():.4f} | "
                          f"Val Acc: {val_acc.result().numpy():.4f} | "
                          f"Val mIoU: {val_iou.result().numpy():.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss.result().numpy():.4f} | "
                          f"Train Acc: {train_acc.result().numpy():.4f}")

        def eval_unet(model, eval_ds, num_classes):
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False)
            loss_m = tf.keras.metrics.Mean()
            acc_m = tf.keras.metrics.SparseCategoricalAccuracy()
            iou_m = tf.keras.metrics.MeanIoU(num_classes=num_classes)

            for images, masks in eval_ds:
                preds = model(images, training=False)
                loss = loss_fn(masks, preds)
                loss_m.update_state(loss)
                acc_m.update_state(masks, preds)
                pred_classes = tf.argmax(preds, axis=-1)
                iou_m.update_state(masks, pred_classes)

            print("\n=== U-Net Segmentation Evaluation Results ===")
            print(f"Loss: {loss_m.result().numpy():.4f}")
            print(f"Accuracy: {acc_m.result().numpy():.4f}")
            print(f"Mean IoU: {iou_m.result().numpy():.4f}")

        # Create model with reduced filters for memory efficiency
        num_classes = 4
        model = UNET(
            input_shape=(128, 128, 202, 1),
            num_classes=num_classes,
            base_filters=16,  # Reduced from 32 for GPU memory
            depth=3  # Reduced from 4 for GPU memory
        )

        if args.mode == 'train':
            train_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='train',
                batch_size=args.batch_size,
                data_mode=3
            )
            val_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='val',
                batch_size=args.batch_size,
                data_mode=3
            )

            # Build model
            dummy_input = tf.random.normal([1, 128, 128, 202, 1])
            _ = model(dummy_input, training=False)

            # Resume from checkpoint if provided
            if args.checkpoint:
                try:
                    model.load_weights(args.checkpoint)
                    print(f"Loaded weights from: {args.checkpoint}")
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint: {e}")

            save_dir = os.path.join(args.output_dir, 'models')
            train_unet(model, train_loader.dataset, val_loader.dataset,
                       args.epochs, save_dir, 'UNET', num_classes)

        elif args.mode in ['validate', 'test']:
            eval_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='test',
                batch_size=args.batch_size,
                data_mode=3
            )

            # Build model
            dummy_input = tf.random.normal([1, 128, 128, 202, 1])
            _ = model(dummy_input, training=False)

            if not args.checkpoint:
                raise ValueError(
                    "Checkpoint required for validation/test mode")

            model.load_weights(args.checkpoint)
            print(f"Loaded weights from: {args.checkpoint}")

            eval_unet(model, eval_loader.dataset, num_classes)

    elif args.model == 'small_seg':
        from segmentation.small_seg import small_segmenter

        # Small segmenter for spatial segmentation using data_mode=3
        # data_mode=3 returns (image, mask) pairs
        early_stopping_counter = 0

        def train_small_seg(model, train_ds, val_ds, epochs, save_dir, prefix, num_classes):
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False)
            best_val = float('inf')
            patience = 15
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)

            for epoch in range(epochs):
                train_loss = tf.keras.metrics.Mean()
                train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
                val_loss = tf.keras.metrics.Mean()
                val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
                val_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

                # Train loop
                for i, (images, masks) in enumerate(train_ds):
                    print(f"Batch {i+1}/{len(train_ds)}", end='\r')
                    with tf.GradientTape() as tape:
                        preds = model(images, training=True)
                        loss = loss_fn(masks, preds)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                    train_loss.update_state(loss)
                    train_acc.update_state(masks, preds)

                # Validation loop
                if val_ds is not None:
                    for i, (images, masks) in enumerate(val_ds):
                        print(f"Val batch {i+1}/{len(val_ds)}", end='\r')
                        preds = model(images, training=False)
                        loss = loss_fn(masks, preds)
                        val_loss.update_state(loss)
                        val_acc.update_state(masks, preds)
                        pred_classes = tf.argmax(preds, axis=-1)
                        val_iou.update_state(masks, pred_classes)

                # Save weights
                weights_path = os.path.join(
                    save_dir, f"{prefix}_epoch_{epoch+1:03d}.weights.h5")
                model.save_weights(weights_path)

                # Track best with early stopping
                current_val = val_loss.result().numpy(
                ) if val_ds is not None else train_loss.result().numpy()
                if current_val < best_val:
                    best_val = current_val
                    patience_counter = 0
                    best_path = os.path.join(
                        save_dir, f"{prefix}_best.weights.h5")
                    model.save_weights(best_path)
                    print(f" [NEW BEST]")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"\nEarly stopping triggered after {patience} epochs without improvement.")
                        break

                # Log
                if val_ds is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss.result().numpy():.4f} | "
                          f"Train Acc: {train_acc.result().numpy():.4f} | "
                          f"Val Loss: {val_loss.result().numpy():.4f} | "
                          f"Val Acc: {val_acc.result().numpy():.4f} | "
                          f"Val mIoU: {val_iou.result().numpy():.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss.result().numpy():.4f} | "
                          f"Train Acc: {train_acc.result().numpy():.4f}")

        def eval_small_seg(model, eval_ds, num_classes):
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False)
            loss_m = tf.keras.metrics.Mean()
            acc_m = tf.keras.metrics.SparseCategoricalAccuracy()
            iou_m = tf.keras.metrics.MeanIoU(num_classes=num_classes)

            for images, masks in eval_ds:
                preds = model(images, training=False)
                loss = loss_fn(masks, preds)
                loss_m.update_state(loss)
                acc_m.update_state(masks, preds)
                pred_classes = tf.argmax(preds, axis=-1)
                iou_m.update_state(masks, pred_classes)

            print("\n=== Small Segmenter Evaluation Results ===")
            print(f"Loss: {loss_m.result().numpy():.4f}")
            print(f"Accuracy: {acc_m.result().numpy():.4f}")
            print(f"Mean IoU: {iou_m.result().numpy():.4f}")

        # Create model - memory efficient for 12GB GPU
        num_classes = 4
        model = small_segmenter(
            input_shape=(128, 128, 202, 1),
            num_classes=num_classes,
            base_filters=8,   # Small for memory efficiency
            depth=3,
            dropout_rate=0.1
        )

        if args.mode == 'train':
            train_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='train',
                batch_size=args.batch_size,
                data_mode=3
            )
            val_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='val',
                batch_size=args.batch_size,
                data_mode=3
            )

            # Build model
            dummy_input = tf.random.normal([1, 128, 128, 202, 1])
            _ = model(dummy_input, training=False)
            model.summary()

            # Resume from checkpoint if provided
            if args.checkpoint:
                try:
                    model.load_weights(args.checkpoint)
                    print(f"Loaded weights from: {args.checkpoint}")
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint: {e}")

            save_dir = os.path.join(args.output_dir, 'models')
            train_small_seg(model, train_loader.dataset, val_loader.dataset,
                            args.epochs, save_dir, 'small_seg', num_classes)

        elif args.mode in ['validate', 'test']:
            eval_loader = TFHySpecNetLoader(
                root_dir=args.data_dir,
                mode=args.split,
                split='test',
                batch_size=args.batch_size,
                data_mode=3
            )

            # Build model
            dummy_input = tf.random.normal([1, 128, 128, 202, 1])
            _ = model(dummy_input, training=False)

            if not args.checkpoint:
                raise ValueError(
                    "Checkpoint required for validation/test mode")

            model.load_weights(args.checkpoint)
            print(f"Loaded weights from: {args.checkpoint}")

            eval_small_seg(model, eval_loader.dataset, num_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train, validate, or test compression/segmentation models. Available models: rcae2D1D, rcae2D, RCGDNAE, LineRWKV, UNET.')
    parser.add_argument('--mode', type=str,
                        choices=['train', 'validate', 'test'],
                        help='Mode of operation: train, validate, or test.')
    parser.add_argument('--split', type=str,
                        choices=['easy', 'hard'],
                        help='Dataset split to use: easy or hard.')
    parser.add_argument('--model', type=str,
                        choices=['rcae2D1D', 'rcae2D',
                                 'RCGDNAE', 'LineRWKV', 'UNET', 'small_seg'],
                        help='Model to use: rcae2D1D, rcae2D, RCGDNAE, LineRWKV, UNET, or small_seg.')
    parser.add_argument('--config', type=str,
                        help='Path to the configuration file.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to the dataset directory. Required unless using --eval_segmentation.')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs and models. Default is ./output.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs. Default is 100.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training/validation/testing. Default is 32.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer. Default is 0.001.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a model checkpoint to resume training or for testing.')
    parser.add_argument('--histogram', action='store_true',
                        help='If set, plot reconstruction error histogram after validation/testing.')
    parser.add_argument('--histogram_save_path', type=str, default=None,
                        help='Path to save the reconstruction error histogram plot.')
    parser.add_argument('--histogram_data_path', type=str, default=None,
                        help='Path to save the reconstruction error histogram data.')
    parser.add_argument('--histogram_bins', type=int, default=200,
                        help='Number of bins for the histogram. Default is 200.')
    parser.add_argument('--histogram_error_type', type=str,
                        choices=['signed', 'absolute', 'squared'],
                        default='signed',
                        help='Type of reconstruction error for histogram: signed, absolute, or squared. Default is signed.')
    parser.add_argument('--save_arrays', action='store_true',
                        help='If set, save original and reconstructed images as numpy arrays for segmentation evaluation.')
    parser.add_argument('--save_arrays_path', type=str, default=None,
                        help='Directory path to save the numpy arrays. Required if --save_arrays is set.')
    parser.add_argument('--save_masks', action='store_true',
                        help='If set alongside --save_arrays, also save ground truth masks (requires data_mode=3).')
    parser.add_argument('--eval_segmentation', action='store_true',
                        help='If set, evaluate segmentation impact on original vs reconstructed images.')
    parser.add_argument('--seg_baseline', action='store_true',
                        help='If set, run segmentation on original dataset only (no compression comparison).')
    parser.add_argument('--seg_checkpoint', type=str, default=None,
                        help='Path to segmentation model checkpoint for --eval_segmentation or --seg_baseline.')
    parser.add_argument('--seg_num_classes', type=int, default=4,
                        help='Number of segmentation classes. Default is 4.')
    parser.add_argument('--seg_results_path', type=str, default=None,
                        help='Path to save segmentation results JSON.')
    parser.add_argument('--load_arrays_path', type=str, default=None,
                        help='Path to load saved arrays (.npz or directory) for standalone segmentation evaluation.')

    args = parser.parse_args()

    # Validate that data_dir is provided when not using eval_segmentation or seg_baseline
    if not args.eval_segmentation and not args.seg_baseline and not args.data_dir:
        parser.error(
            "--data_dir is required unless using --eval_segmentation or --seg_baseline mode.")

    # Validate seg_baseline requirements
    if args.seg_baseline:
        if not args.seg_checkpoint:
            parser.error(
                "--seg_checkpoint is required when using --seg_baseline.")
        if not args.data_dir:
            parser.error("--data_dir is required when using --seg_baseline.")

    main(args)
