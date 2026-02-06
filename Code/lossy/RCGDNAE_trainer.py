"""
Trainer for RCGDNAE (Rate-Constrained GDN Autoencoder) lossy compression model.
Inspired by LineRWKVTrainer, adapted for TFHySpecNetLoader.
"""

import tensorflow as tf
import numpy as np
import os
import time
import math
from datetime import datetime

from lossy.RCGDNAE import (
    FullCompressor, rate_distortion_loss, gaussian_likelihood,
    IMAGE_SIZE, CHANNELS, LATENT_CHANNELS, HYPER_CHANNELS, LD,
    load_klt_from_file
)


class RCGDNAETrainer:
    """
    Trainer for RCGDNAE lossy compression model for hyperspectral images.
    """

    def __init__(self,
                 model_config=None,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 lambda_rd=0.01,
                 use_mixed_precision=False,
                 enable_performance_optimizations=True,
                 klt_matrix_path=None):
        """
        Initialize the RCGDNAE trainer.

        Args:
            model_config: Configuration for the FullCompressor model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for AdamW optimizer
            lambda_rd: Lambda parameter for rate-distortion tradeoff
            use_mixed_precision: Whether to use mixed precision training
            enable_performance_optimizations: Whether to enable TF optimizations
        """

        # Apply performance optimizations
        if enable_performance_optimizations:
            print("Applying performance optimizations...")
            self._apply_performance_config(use_mixed_precision)
        else:
            print("Performance optimizations disabled")

        # Model configuration
        self.model_config = model_config or {
            'img_height': IMAGE_SIZE,
            'img_width': IMAGE_SIZE,
            'bands': CHANNELS
        }

        # Remove klt_matrix_path if present (it's already been processed to klt_matrix/klt_mean)
        self.model_config.pop('klt_matrix_path', None)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_rd = lambda_rd

        # Initialize model
        self.model = FullCompressor(**self.model_config)

        # Build model with dummy input
        dummy_input = tf.zeros((1,
                                self.model_config.get(
                                    'img_height', IMAGE_SIZE),
                                self.model_config.get('img_width', IMAGE_SIZE),
                                self.model_config.get('bands', CHANNELS)))
        _ = self.model(dummy_input, training=False)
        print(
            f"Model built with {len(self.model.trainable_variables)} trainable variables")

        # Setup mixed precision if requested
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        # Setup optimizer
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        # Metrics for training
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_distortion = tf.keras.metrics.Mean(name='train_distortion')
        self.train_rate = tf.keras.metrics.Mean(name='train_rate')
        self.train_psnr = tf.keras.metrics.Mean(name='train_psnr')
        self.train_ssim = tf.keras.metrics.Mean(name='train_ssim')
        self.train_sa = tf.keras.metrics.Mean(name='train_sa')
        # Additional losses tracking
        self.train_mse = tf.keras.metrics.Mean(name='train_mse')
        self.train_mae = tf.keras.metrics.Mean(name='train_mae')

        # Metrics for validation
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_distortion = tf.keras.metrics.Mean(name='val_distortion')
        self.val_rate = tf.keras.metrics.Mean(name='val_rate')
        self.val_psnr = tf.keras.metrics.Mean(name='val_psnr')
        self.val_ssim = tf.keras.metrics.Mean(name='val_ssim')
        self.val_sa = tf.keras.metrics.Mean(name='val_sa')
        # Additional losses tracking
        self.val_mse = tf.keras.metrics.Mean(name='val_mse')
        self.val_mae = tf.keras.metrics.Mean(name='val_mae')

    def _apply_performance_config(self, use_mixed_precision):
        """Apply TensorFlow performance optimizations"""
        try:
            # Enable XLA compilation
            tf.config.optimizer.set_jit(True)

            # Memory growth for GPUs
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Configured {len(gpus)} GPU(s) with memory growth")

            if use_mixed_precision:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision enabled")

        except Exception as e:
            print(f"Performance configuration warning: {e}")

    def compute_rate_distortion_loss(self, x, x_hat, y, mu, sigma):
        """
        Compute rate-distortion loss for compression.

        Args:
            x: Original input
            x_hat: Reconstructed input
            y: Latent representation
            mu: Mean from entropy model
            sigma: Standard deviation from entropy model

        Returns:
            total_loss, distortion, rate
        """
        # Switch distortion to MAE (mean absolute error)
        distortion = tf.reduce_mean(tf.abs(x - x_hat))

        # Discretized Gaussian PMF via CDF differences to estimate bits
        # Stabilize sigma to avoid degenerate/negative differential entropy
        sigma = tf.maximum(sigma, tf.cast(1e-3, sigma.dtype))
        upper = (y - mu + tf.cast(0.5, y.dtype)) / sigma
        lower = (y - mu - tf.cast(0.5, y.dtype)) / sigma
        inv_sqrt2 = tf.math.rsqrt(tf.cast(2.0, y.dtype))
        cdf_u = 0.5 * (1.0 + tf.math.erf(upper * inv_sqrt2))
        cdf_l = 0.5 * (1.0 + tf.math.erf(lower * inv_sqrt2))
        pmf = tf.clip_by_value(cdf_u - cdf_l, 1e-12, 1.0)

        rate = tf.reduce_mean(-tf.math.log(pmf)) / math.log(2)

        # Rate-distortion loss: D + λ * R (R is non-negative with PMF)
        total_loss = distortion + self.lambda_rd * tf.maximum(rate, 0.0)

        return total_loss, distortion, rate

    def compute_psnr(self, y_true, y_pred):
        """Compute PSNR metric"""
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        max_val = 1.0  # Assume normalized data
        psnr = 20 * tf.math.log(max_val /
                                tf.sqrt(mse + 1e-10)) / tf.math.log(10.0)
        return psnr

    def compute_ssim(self, y_true, y_pred):
        """Compute SSIM (Structural Similarity Index) metric"""
        if len(tf.shape(y_true)) == 4:  # [batch, height, width, channels]
            batch_size = tf.shape(y_true)[0]
            height = tf.shape(y_true)[1]
            width = tf.shape(y_true)[2]
            channels = tf.shape(y_true)[3]

            # Reshape to [batch*channels, height, width, 1] for tf.image.ssim
            y_true_reshaped = tf.reshape(tf.transpose(y_true, [0, 3, 1, 2]),
                                         [batch_size * channels, height, width, 1])
            y_pred_reshaped = tf.reshape(tf.transpose(y_pred, [0, 3, 1, 2]),
                                         [batch_size * channels, height, width, 1])

            ssim_values = tf.image.ssim(
                y_true_reshaped, y_pred_reshaped, max_val=1.0)
            ssim_per_sample = tf.reshape(ssim_values, [batch_size, channels])
            ssim = tf.reduce_mean(ssim_per_sample)
        else:
            ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        return ssim

    def compute_spectral_angle(self, y_true, y_pred):
        """Compute Spectral Angle (SA) metric for hyperspectral data"""
        if len(tf.shape(y_true)) == 4:
            y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
            y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        else:
            y_true_flat = y_true
            y_pred_flat = y_pred

        y_true_norm = tf.nn.l2_normalize(y_true_flat, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred_flat, axis=-1)

        cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)

        spectral_angles = tf.acos(cos_sim)
        return tf.reduce_mean(spectral_angles)

    @tf.function
    def train_step(self, x_batch):
        """
        Training step for RCGDNAE compression model.

        Args:
            x_batch: Batch of input images

        Returns:
            loss, distortion, rate, psnr, ssim, sa  
        """

        with tf.GradientTape() as tape:
            # Forward pass
            x_hat, y, mu, sigma = self.model(x_batch, training=True)

            # Loss computation
            loss, distortion, rate = self.compute_rate_distortion_loss(
                x_batch, x_hat, y, mu, sigma
            )

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        # Compute metrics
        psnr = self.compute_psnr(x_batch, x_hat)
        ssim = self.compute_ssim(x_batch, x_hat)
        sa = self.compute_spectral_angle(x_batch, x_hat)
        # Loss components for tracking
        mse = tf.reduce_mean(tf.square(x_batch - x_hat))
        mae = tf.reduce_mean(tf.abs(x_batch - x_hat))

        # Update metrics
        self.train_loss.update_state(loss)
        self.train_distortion.update_state(distortion)
        self.train_rate.update_state(rate)
        self.train_psnr.update_state(psnr)
        self.train_ssim.update_state(ssim)
        self.train_sa.update_state(sa)
        self.train_mse.update_state(mse)
        self.train_mae.update_state(mae)
        return loss, distortion, rate, psnr, ssim, sa

    @tf.function
    def val_step(self, x_batch):
        """Validation step"""
        x_hat, y, mu, sigma = self.model(x_batch, training=False)
        loss, distortion, rate = self.compute_rate_distortion_loss(
            x_batch, x_hat, y, mu, sigma
        )
        psnr = self.compute_psnr(x_batch, x_hat)
        ssim = self.compute_ssim(x_batch, x_hat)
        sa = self.compute_spectral_angle(x_batch, x_hat)
        # Loss components for tracking
        mse = tf.reduce_mean(tf.square(x_batch - x_hat))
        mae = tf.reduce_mean(tf.abs(x_batch - x_hat))

        self.val_loss.update_state(loss)
        self.val_distortion.update_state(distortion)
        self.val_rate.update_state(rate)
        self.val_psnr.update_state(psnr)
        self.val_ssim.update_state(ssim)
        self.val_sa.update_state(sa)
        self.val_mse.update_state(mse)
        self.val_mae.update_state(mae)

        return loss, distortion, rate, psnr, ssim, sa

    def reset_metrics(self, validation=False):
        """Reset all metrics"""
        self.train_loss.reset_state()
        self.train_distortion.reset_state()
        self.train_rate.reset_state()
        self.train_psnr.reset_state()
        self.train_ssim.reset_state()
        self.train_sa.reset_state()
        self.train_mse.reset_state()
        self.train_mae.reset_state()

        if validation:
            self.val_loss.reset_state()
            self.val_distortion.reset_state()
            self.val_rate.reset_state()
            self.val_psnr.reset_state()
            self.val_ssim.reset_state()
            self.val_sa.reset_state()
            self.val_mse.reset_state()
            self.val_mae.reset_state()

    def train_epoch(self, train_dataset, val_dataset=None):
        """
        Train for one epoch.

        Args:
            train_dataset: TFHySpecNetLoader dataset for training
            val_dataset: Optional validation dataset

        Returns:
            Dictionary with epoch metrics
        """
        # Reset metrics
        self.reset_metrics(validation=val_dataset is not None)

        batch_start_time = time.time()
        num_batches = 0

        # Training loop - handle both TFHySpecNetLoader and raw tf.data.Dataset
        dataset = train_dataset.dataset if hasattr(
            train_dataset, 'dataset') else train_dataset
        total_batches = len(dataset)

        for batch in dataset:
            self.train_step(batch)
            num_batches += 1
            elapsed = time.time() - batch_start_time
            eta = (elapsed / num_batches) * (total_batches -
                                             num_batches) if isinstance(total_batches, int) else 0
            print(f"\rBatch {num_batches}/{total_batches} | "
                  f"Loss: {self.train_loss.result():.4f} | "
                  f"D(MAE): {self.train_distortion.result():.4f} | "
                  f"MSE: {self.train_mse.result():.6f} | "
                  f"MAE: {self.train_mae.result():.6f} | "
                  f"R: {self.train_rate.result():.2f} | "
                  f"PSNR: {self.train_psnr.result():.2f} | "
                  f"SSIM: {self.train_ssim.result():.4f} | "
                  f"SA: {self.train_sa.result():.4f} | "
                  f"Time: {elapsed:.2f}s | "
                  f"ETA: {eta:.2f}s", end='\r')

        print()  # New line after progress
        # Validation loop
        if val_dataset is not None:
            val_data = val_dataset.dataset if hasattr(
                val_dataset, 'dataset') else val_dataset
            total_batches = len(val_data)
            num_val_batches = 0
            for batch in val_data:
                self.val_step(batch)
                num_val_batches += 1
                elapsed = time.time() - batch_start_time
                eta = (elapsed / (num_batches + num_val_batches)) * (
                    total_batches - num_batches - num_val_batches) if isinstance(total_batches, int) else 0
                print(f"\rValidation Batch {num_val_batches}/{total_batches} | "
                      f"Loss: {self.val_loss.result():.4f} | "
                      f"D(MAE): {self.val_distortion.result():.4f} | "
                      f"MSE: {self.val_mse.result():.6f} | "
                      f"MAE: {self.val_mae.result():.6f} | "
                      f"R: {self.val_rate.result():.2f} | "
                      f"PSNR: {self.val_psnr.result():.2f} | "
                      f"SSIM: {self.val_ssim.result():.4f} | "
                      f"SA: {self.val_sa.result():.4f} | "
                      f"Time: {elapsed:.2f}s | "
                      f"ETA: {eta:.2f}s", end='\r')
            print(f"Validation - Loss: {self.val_loss.result():.4f}, "
                  f"D(MAE): {self.val_distortion.result():.4f}, "
                  f"MSE: {self.val_mse.result():.6f}, "
                  f"MAE: {self.val_mae.result():.6f}, "
                  f"R: {self.val_rate.result():.2f}, "
                  f"PSNR: {self.val_psnr.result():.2f}, "
                  f"SSIM: {self.val_ssim.result():.4f}, "
                  f"SA: {self.val_sa.result():.4f}")
        print()  # New line after validation progress
        print(f"Epoch completed in {time.time() - batch_start_time:.1f}s")
        # Collect results
        results = {
            'train_loss': self.train_loss.result().numpy(),
            'train_distortion': self.train_distortion.result().numpy(),
            'train_mse': self.train_mse.result().numpy(),
            'train_mae': self.train_mae.result().numpy(),
            'train_rate': self.train_rate.result().numpy(),
            'train_psnr': self.train_psnr.result().numpy(),
            'train_ssim': self.train_ssim.result().numpy(),
            'train_sa': self.train_sa.result().numpy(),
            'num_batches': num_batches
        }

        if val_dataset is not None:
            results.update({
                'val_loss': self.val_loss.result().numpy(),
                'val_distortion': self.val_distortion.result().numpy(),
                'val_mse': self.val_mse.result().numpy(),
                'val_mae': self.val_mae.result().numpy(),
                'val_rate': self.val_rate.result().numpy(),
                'val_psnr': self.val_psnr.result().numpy(),
                'val_ssim': self.val_ssim.result().numpy(),
                'val_sa': self.val_sa.result().numpy()
            })

        return results

    def fit(self,
            train_dataset,
            val_dataset=None,
            epochs=100,
            save_path=None,
            save_frequency=10,
            verbose=True):
        """
        Complete training loop.

        Args:
            train_dataset: TFHySpecNetLoader for training
            val_dataset: Optional TFHySpecNetLoader for validation
            epochs: Number of epochs to train
            save_path: Path to save checkpoints
            save_frequency: Save every N epochs
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

        best_val_loss = float('inf')
        history = {
            'train_loss': [], 'train_distortion': [], 'train_mse': [], 'train_mae': [], 'train_rate': [],
            'train_psnr': [], 'train_ssim': [], 'train_sa': [],
            'val_loss': [], 'val_distortion': [], 'val_mse': [], 'val_mae': [], 'val_rate': [],
            'val_psnr': [], 'val_ssim': [], 'val_sa': []
        }

        for epoch in range(epochs):
            start_time = time.time()

            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-" * 50)

            # Train epoch
            epoch_results = self.train_epoch(train_dataset, val_dataset)

            # Record history
            history['train_loss'].append(epoch_results['train_loss'])
            history['train_distortion'].append(
                epoch_results['train_distortion'])
            history['train_rate'].append(epoch_results['train_rate'])
            history['train_psnr'].append(epoch_results['train_psnr'])
            history['train_ssim'].append(epoch_results['train_ssim'])
            history['train_sa'].append(epoch_results['train_sa'])
            history['train_mse'].append(epoch_results['train_mse'])
            history['train_mae'].append(epoch_results['train_mae'])
            if val_dataset:
                history['val_loss'].append(epoch_results['val_loss'])
                history['val_distortion'].append(
                    epoch_results['val_distortion'])
                history['val_rate'].append(epoch_results['val_rate'])
                history['val_psnr'].append(epoch_results['val_psnr'])
                history['val_ssim'].append(epoch_results['val_ssim'])
                history['val_sa'].append(epoch_results['val_sa'])
                history['val_mse'].append(epoch_results['val_mse'])
                history['val_mae'].append(epoch_results['val_mae'])

            epoch_time = time.time() - start_time

            # Print summary
            if verbose:
                print(f"Train - Loss: {epoch_results['train_loss']:.4f}, "
                      f"D(MAE): {epoch_results['train_distortion']:.4f}, "
                      f"MSE: {epoch_results['train_mse']:.6f}, "
                      f"MAE: {epoch_results['train_mae']:.6f}, "
                      f"R: {epoch_results['train_rate']:.2f} bits, "
                      f"PSNR: {epoch_results['train_psnr']:.2f} dB, "
                      f"SSIM: {epoch_results['train_ssim']:.4f}, "
                      f"SA: {epoch_results['train_sa']:.4f}")

                if val_dataset:
                    print(f"Val   - Loss: {epoch_results['val_loss']:.4f}, "
                          f"D(MAE): {epoch_results['val_distortion']:.4f}, "
                          f"MSE: {epoch_results['val_mse']:.6f}, "
                          f"MAE: {epoch_results['val_mae']:.6f}, "
                          f"R: {epoch_results['val_rate']:.2f} bits, "
                          f"PSNR: {epoch_results['val_psnr']:.2f} dB, "
                          f"SSIM: {epoch_results['val_ssim']:.4f}, "
                          f"SA: {epoch_results['val_sa']:.4f}")

                print(f"Time: {epoch_time:.1f}s")

            # Save checkpoint
            if save_path and (epoch + 1) % save_frequency == 0:
                checkpoint_path = os.path.join(
                    save_path, f"checkpoint_epoch_{epoch + 1}")
                self.save_model(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            # Save best model
            if val_dataset and epoch_results['val_loss'] < best_val_loss:
                best_val_loss = epoch_results['val_loss']
                if save_path:
                    best_path = os.path.join(save_path, "best_model")
                    self.save_model(best_path)
                    print(
                        f"New best model saved (val_loss: {best_val_loss:.4f})")

        return history

    def save_model(self, path):
        """Save the model"""
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def compress(self, x):
        """
        Compress input and return latent representation.

        Args:
            x: Input image(s)

        Returns:
            Dictionary with compressed representation and metrics
        """
        x_hat, y, mu, sigma = self.model(x, training=False)
        loss, distortion, rate = self.compute_rate_distortion_loss(
            x, x_hat, y, mu, sigma)
        psnr = self.compute_psnr(x, x_hat)

        return {
            'reconstructed': x_hat,
            'latent': y,
            'mu': mu,
            'sigma': sigma,
            'loss': loss,
            'distortion': distortion,
            'rate': rate,
            'psnr': psnr
        }


def create_rcgdnae_trainer(dataloader_config, model_config=None, trainer_config=None, klt_path=None):
    """
    Factory function to create a trainer compatible with TFHySpecNetLoader.

    Args:
        dataloader_config: Config for the TFHySpecNetLoader
        model_config: Config for the FullCompressor model
        trainer_config: Config for the trainer
        klt_path: Optional path to .npz file with KLT matrix and mean vector.
                  If provided, KLT preprocessing will be enabled.

    Returns:
        RCGDNAETrainer instance
    """
    # Default model config based on dataloader
    default_model_config = {
        'img_height': 128,
        'img_width': 128,
        'bands': dataloader_config.get('channels', 202)
    }

    default_trainer_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'lambda_rd': 0.01,
        'use_mixed_precision': False,
        'enable_performance_optimizations': True
    }

    # Update with provided configs
    if model_config:
        default_model_config.update(model_config)
    if trainer_config:
        default_trainer_config.update(trainer_config)

    # Load KLT if path provided (from klt_path argument or from model_config)
    klt_path = klt_path or default_model_config.pop('klt_matrix_path', None)
    if klt_path:
        print(f"Loading KLT from: {klt_path}")
        klt_matrix, klt_mean = load_klt_from_file(klt_path)
        default_model_config['klt_matrix'] = klt_matrix
        default_model_config['klt_mean'] = klt_mean
        print(
            f"KLT loaded: {klt_matrix.shape[0]} components from {klt_matrix.shape[1]} bands")

    # Create trainer
    trainer = RCGDNAETrainer(
        model_config=default_model_config,
        **default_trainer_config
    )

    return trainer


if __name__ == "__main__":
    # Example usage with TFHySpecNetLoader
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    from TFDataloader.TFdataloader import TFHySpecNetLoader

    # Configuration
    DATA_ROOT = "/path/to/hyspecnet"  # Update this path
    BATCH_SIZE = 4
    EPOCHS = 10

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = TFHySpecNetLoader(
        root_dir=DATA_ROOT,
        mode="easy",
        split="train",
        data_mode=2,  # [height, width, channels] format
        batch_size=BATCH_SIZE
    )

    val_loader = TFHySpecNetLoader(
        root_dir=DATA_ROOT,
        mode="easy",
        split="val",
        data_mode=2,
        batch_size=BATCH_SIZE
    )

    # Create trainer
    print("Creating trainer...")
    model_config = {
        'img_height': 128,
        'img_width': 128,
        'bands': 202  # Hyperspectral channels
    }

    trainer = RCGDNAETrainer(
        model_config=model_config,
        learning_rate=1e-4,
        lambda_rd=0.01
    )

    # Train
    print("Starting training...")
    history = trainer.fit(
        train_dataset=train_loader,
        val_dataset=val_loader,
        epochs=EPOCHS,
        save_path="./checkpoints/rcgdnae",
        save_frequency=5,
        verbose=True
    )

    print("✅ Training complete!")
