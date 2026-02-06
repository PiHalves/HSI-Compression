import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime

from .lineRWKV import LineRWKVPredictor, LineRWKVLosslessCodec
from .performance_config import configure_tensorflow_performance, print_performance_info, apply_late_optimizations


class LineRWKVTrainer:
    """
    Trainer for Line-based RWKV models for hyperspectral image prediction
    """

    def __init__(self,
                 model_config=None,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 loss_type='mse',
                 use_mixed_precision=False,
                 enable_performance_optimizations=True):

        # Apply performance optimizations first
        if enable_performance_optimizations:
            print("Applying performance optimizations...")
            try:
                config_applied = configure_tensorflow_performance(
                    enable_xla=True,
                    enable_mixed_precision=use_mixed_precision,
                    inter_op_parallelism=0,  # Use all cores
                    intra_op_parallelism=0,  # Use all cores
                    allow_memory_growth=True,
                    warn_on_late_config=False  # Don't warn, just handle gracefully
                )
            except Exception as e:
                print(
                    f"Full performance configuration failed, applying partial optimizations...")
                # Apply what we can
                config_applied = apply_late_optimizations(
                    enable_xla=True,
                    enable_mixed_precision=use_mixed_precision,
                    allow_memory_growth=True
                )

            # Show what was successfully configured
            configured_features = [k for k, v in config_applied.items() if v]
            if configured_features:
                print(
                    f"Successfully configured: {', '.join(configured_features)}")
            else:
                print("Performance optimizations were limited due to TensorFlow state")
        else:
            print("Performance optimizations disabled")

        self.model_config = model_config or {
            'input_channels': 202,
            'dim': 128,
            'num_layers': 4,
            'time_decay': 0.99,
            'prediction_mode': 'spectral'
        }

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type

        # Initialize model
        self.model = LineRWKVPredictor(**self.model_config)

        # Setup mixed precision if requested
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        # Setup optimizer
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        # Setup loss function
        self._setup_loss_function()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_psnr = tf.keras.metrics.Mean(name='train_psnr')
        self.train_ssim = tf.keras.metrics.Mean(name='train_ssim')
        self.train_sa = tf.keras.metrics.Mean(name='train_sa')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_psnr = tf.keras.metrics.Mean(name='val_psnr')
        self.val_ssim = tf.keras.metrics.Mean(name='val_ssim')
        self.val_sa = tf.keras.metrics.Mean(name='val_sa')

    def _setup_loss_function(self):
        """Setup loss function based on loss_type"""
        if self.loss_type == 'mse':
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.loss_type == 'mae':
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif self.loss_type == 'huber':
            self.loss_fn = tf.keras.losses.Huber()
        elif self.loss_type == 'prediction_residual':
            self.loss_fn = self._prediction_residual_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _prediction_residual_loss(self, y_true, y_pred):
        """
        Custom loss for prediction residual minimization
        Focuses on minimizing prediction errors for better compression
        """
        # Direct prediction loss
        pred_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        # Residual statistics loss (encourage low variance residuals)
        residuals = y_true - y_pred
        residual_variance = tf.reduce_mean(
            tf.square(residuals - tf.reduce_mean(residuals)))

        # Spectral continuity loss
        if len(tf.shape(y_true)) == 4:  # [batch, height, width, channels]
            spectral_diff = tf.reduce_mean(
                tf.square(y_true[:, :, :, 1:] - y_true[:, :, :, :-1]))
            pred_spectral_diff = tf.reduce_mean(
                tf.square(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
            spectral_loss = tf.square(spectral_diff - pred_spectral_diff)
        else:
            spectral_loss = 0.0

        total_loss = pred_loss + 0.1 * residual_variance + 0.05 * spectral_loss
        return total_loss

    def compute_psnr(self, y_true, y_pred):
        """Compute PSNR metric"""
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        # Assume data is normalized to [0, 1]
        max_val = 1.0
        psnr = 20 * tf.math.log(max_val / tf.sqrt(mse)) / tf.math.log(10.0)
        return psnr

    def compute_ssim(self, y_true, y_pred):
        """Compute SSIM (Structural Similarity Index) metric"""
        # For hyperspectral data, compute SSIM for each spatial slice and average
        if len(tf.shape(y_true)) == 4:  # [batch, height, width, channels]
            # Compute SSIM for each channel separately and average
            batch_size = tf.shape(y_true)[0]
            height = tf.shape(y_true)[1]
            width = tf.shape(y_true)[2]
            channels = tf.shape(y_true)[3]

            # Reshape to [batch*channels, height, width, 1] for tf.image.ssim
            y_true_reshaped = tf.reshape(tf.transpose(y_true, [0, 3, 1, 2]),
                                         [batch_size * channels, height, width, 1])
            y_pred_reshaped = tf.reshape(tf.transpose(y_pred, [0, 3, 1, 2]),
                                         [batch_size * channels, height, width, 1])

            # Compute SSIM for all channels
            ssim_values = tf.image.ssim(
                y_true_reshaped, y_pred_reshaped, max_val=1.0)

            # Reshape back to [batch, channels] and take mean
            ssim_per_sample = tf.reshape(ssim_values, [batch_size, channels])
            ssim = tf.reduce_mean(ssim_per_sample)
        else:
            # For other shapes, compute directly
            ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
            ssim = tf.reduce_mean(ssim)

        return ssim

    def compute_spectral_angle(self, y_true, y_pred):
        """Compute Spectral Angle (SA) metric for hyperspectral data"""
        # Flatten spatial dimensions to get [batch*height*width, channels]
        if len(tf.shape(y_true)) == 4:  # [batch, height, width, channels]
            y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
            y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        else:
            y_true_flat = y_true
            y_pred_flat = y_pred

        # Normalize vectors
        y_true_norm = tf.nn.l2_normalize(y_true_flat, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred_flat, axis=-1)

        # Compute dot product (cosine similarity)
        cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)

        # Clip to avoid numerical issues with arccos
        cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)

        # Compute spectral angle in radians
        spectral_angles = tf.acos(cos_sim)

        # Return mean spectral angle
        return tf.reduce_mean(spectral_angles)

    @tf.function
    def train_step(self, x_batch):
        """
        Training step for line-based RWKV
        Uses the input as both input and target for prediction training
        """
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x_batch, training=True)

            # Loss computation
            loss = self.loss_fn(x_batch, predictions)

            # Handle mixed precision
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Compute gradients
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            scaled_gradients = tape.gradient(
                scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        # Update metrics
        self.train_loss.update_state(loss)
        psnr = self.compute_psnr(x_batch, predictions)
        self.train_psnr.update_state(psnr)
        ssim = self.compute_ssim(x_batch, predictions)
        self.train_ssim.update_state(ssim)
        sa = self.compute_spectral_angle(x_batch, predictions)
        self.train_sa.update_state(sa)

        return loss, psnr, ssim, sa

    @tf.function
    def val_step(self, x_batch):
        """Validation step"""
        predictions = self.model(x_batch, training=False)
        loss = self.loss_fn(x_batch, predictions)
        psnr = self.compute_psnr(x_batch, predictions)
        ssim = self.compute_ssim(x_batch, predictions)
        sa = self.compute_spectral_angle(x_batch, predictions)

        self.val_loss.update_state(loss)
        self.val_psnr.update_state(psnr)
        self.val_ssim.update_state(ssim)
        self.val_sa.update_state(sa)

        return loss, psnr, ssim, sa

    def train_epoch(self, train_dataset, val_dataset=None):
        """Train for one epoch"""
        # Reset metrics
        print(f"Epoch start - resetting metrics")
        self.train_loss.reset_state()
        self.train_psnr.reset_state()
        self.train_ssim.reset_state()
        self.train_sa.reset_state()
        if val_dataset:
            self.val_loss.reset_state()
            self.val_psnr.reset_state()
            self.val_ssim.reset_state()
            self.val_sa.reset_state()

        batch_start_time = time.time()
        # Training loop
        num_batches = 0
        for batch in train_dataset:
            self.train_step(batch)
            num_batches += 1
            print(
                f"Trained on batch {num_batches}/{len(train_dataset)} of shape {batch.shape} Elapsed: {time.time() - batch_start_time:.2f}s To go: {(time.time() - batch_start_time) / num_batches * (len(train_dataset) - num_batches):.2f}s", end='\r')

        # Validation loop
        if val_dataset:
            for batch in val_dataset:
                self.val_step(batch)
                print(
                    f"Validated on batch {num_batches}/{len(val_dataset)}", end='\r')

        # Collect results
        results = {
            'train_loss': self.train_loss.result().numpy(),
            'train_psnr': self.train_psnr.result().numpy(),
            'train_ssim': self.train_ssim.result().numpy(),
            'train_sa': self.train_sa.result().numpy(),
            'num_batches': num_batches
        }

        if val_dataset:
            results.update({
                'val_loss': self.val_loss.result().numpy(),
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
        Complete training loop
        """
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'train_psnr': [],
            'train_ssim': [],
            'train_sa': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'val_sa': []
        }

        for epoch in range(epochs):
            start_time = time.time()

            # Train epoch
            epoch_results = self.train_epoch(train_dataset, val_dataset)

            # Record history
            history['train_loss'].append(epoch_results['train_loss'])
            history['train_psnr'].append(epoch_results['train_psnr'])
            history['train_ssim'].append(epoch_results['train_ssim'])
            history['train_sa'].append(epoch_results['train_sa'])

            if val_dataset:
                history['val_loss'].append(epoch_results['val_loss'])
                history['val_psnr'].append(epoch_results['val_psnr'])
                history['val_ssim'].append(epoch_results['val_ssim'])
                history['val_sa'].append(epoch_results['val_sa'])

            # Print progress
            if verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {epoch_results['train_loss']:.6f}")
                print(f"  Train PSNR: {epoch_results['train_psnr']:.2f} dB")
                if val_dataset:
                    print(f"  Val Loss: {epoch_results['val_loss']:.6f}")
                    print(f"  Val PSNR: {epoch_results['val_psnr']:.2f} dB")
                print(f"  Time: {epoch_time:.2f}s")
                print()

            # Save model
            if save_path and (epoch + 1) % save_frequency == 0:
                model_path = os.path.join(
                    save_path, f"lineRWKV_epoch_{epoch+1}.weights.h5")
                self.model.save_weights(model_path)

                # Save best model
                if val_dataset and epoch_results['val_loss'] < best_val_loss:
                    best_val_loss = epoch_results['val_loss']
                    best_path = os.path.join(
                        save_path, "lineRWKV_best.weights.h5")
                    self.model.save_weights(best_path)

        return history

    def predict_with_residuals(self, x):
        """
        Predict and compute residuals for compression analysis
        """
        predictions = self.model(x, training=False)
        residuals = x - predictions

        # Compute compression metrics
        mse = tf.reduce_mean(tf.square(residuals))
        residual_entropy = -tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.nn.softmax(residuals, axis=-1),
                logits=residuals
            )
        )

        return {
            'predictions': predictions,
            'residuals': residuals,
            'mse': mse,
            'residual_entropy': residual_entropy
        }

    def save_model(self, path):
        """Save complete model"""
        self.model.save(path)

    def load_model(self, path):
        """Load complete model"""
        self.model = tf.keras.models.load_model(path)


def create_line_rwkv_trainer(dataloader_config, model_config=None, trainer_config=None):
    """
    Factory function to create a trainer compatible with TFDataloader

    Args:
        dataloader_config: Config for the dataloader
        model_config: Config for the LineRWKV model
        trainer_config: Config for the trainer
    """
    # Default configs
    default_model_config = {
        'input_channels': 202,
        'dim': 128,
        'num_layers': 4,
        'time_decay': 0.99,
        'prediction_mode': 'spectral'
    }

    default_trainer_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'loss_type': 'prediction_residual',
        'use_mixed_precision': False
    }

    # Update with provided configs
    if model_config:
        default_model_config.update(model_config)
    if trainer_config:
        default_trainer_config.update(trainer_config)

    # Create trainer
    trainer = LineRWKVTrainer(
        model_config=default_model_config,
        **default_trainer_config
    )

    return trainer
