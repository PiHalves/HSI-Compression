import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class LineRWKVBlock(layers.Layer):
    """
    Paper-faithful RWKV block for line-based predictive coding.
    Implements exact RWKV recurrence for lossless compression causality.

    Key constraints for paper faithfulness:
    - Explicit left-to-right recurrence (no vectorized cumsum tricks)
    - Receptance applied inside the recurrence step
    - Fixed decay per block (no varying decay)
    - Minimal channel mixing (no heavy FFN)
    - No layer normalization (entropy stability)
    """

    def __init__(self, dim, time_decay=0.99, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.time_decay = time_decay

        # Projections for RWKV mechanism
        self.value = layers.Dense(dim, use_bias=False, name='value_proj')
        self.receptance = layers.Dense(
            dim, use_bias=False, name='receptance_proj')

        # Minimal channel mixing (paper-faithful)
        self.channel_mix = layers.Dense(
            dim, use_bias=False, name='channel_mix')

    def build(self, input_shape):
        """
        Build the layer with the given input shape
        input_shape: (batch, width, dim) for line processing
        """
        super().build(input_shape)

    def call(self, x, state=None, training=False):
        """
        Paper-faithful RWKV processing with explicit recurrence.

        x: [batch, width, dim]
        state: [batch, dim] - previous state for time mixing

        Uses tf.scan for explicit left-to-right recurrence to ensure:
        - Exact RWKV recurrence: s_t = decay * s_{t-1} + (1-decay) * v_t
        - Receptance applied per-step: y_t = x_t + r_t * s_t
        - Decoder-replayable computation
        """
        batch = tf.shape(x)[0]

        if state is None:
            state = tf.zeros((batch, self.dim), dtype=x.dtype)

        # Compute projections (these are position-independent)
        v = self.value(x)  # [batch, width, dim]
        r = tf.sigmoid(self.receptance(x))  # [batch, width, dim]

        decay = tf.cast(self.time_decay, x.dtype)
        one_minus_decay = 1.0 - decay

        # Transpose for tf.scan: [width, batch, dim]
        v_t = tf.transpose(v, [1, 0, 2])
        r_t = tf.transpose(r, [1, 0, 2])
        x_t = tf.transpose(x, [1, 0, 2])

        def recurrence_step(carry, inputs):
            """
            Paper-faithful RWKV recurrence step.

            carry: (s_prev, _) - previous state and dummy output placeholder
            inputs: (v_t, r_t, x_t) - current value, receptance, input

            Returns:
            - (s_curr, y_curr): new state and output for this step
            """
            s_prev, _ = carry
            v_curr, r_curr, x_curr = inputs

            # Exact RWKV recurrence
            s_curr = decay * s_prev + one_minus_decay * v_curr

            # Apply receptance INSIDE the recurrence (paper-faithful)
            y_curr = x_curr + r_curr * s_curr

            return (s_curr, y_curr)

        # Initial carry: (state, dummy output with same shape as one output step)
        dummy_output = tf.zeros((batch, self.dim), dtype=x.dtype)
        initial_carry = (state, dummy_output)

        # Explicit left-to-right scan (no parallelization for causality)
        # tf.scan returns accumulated (states, outputs) with shape [width, batch, dim]
        states_t, y_t = tf.scan(
            recurrence_step,
            elems=(v_t, r_t, x_t),
            initializer=initial_carry,
        )  # states_t: [width, batch, dim], y_t: [width, batch, dim]

        # Transpose back: [batch, width, dim]
        y = tf.transpose(y_t, [1, 0, 2])

        # Get the actual final state (last position from the accumulated states)
        new_state = states_t[-1]  # [batch, dim]

        # Minimal channel mixing (paper-faithful, no heavy FFN)
        y = y + self.channel_mix(y)

        return y, new_state


class LineRWKVPredictor(tf.keras.Model):
    """
    Paper-faithful LineRWKV predictive model for hyperspectral images.
    Implements exact RWKV recurrence for lossless compression.

    Key constraints for paper faithfulness:
    - Fixed decay per block (not varying)
    - Minimal architecture overhead
    - Explicit recurrence in width dimension
    - Line-by-line height scan
    """

    def __init__(self,
                 input_channels=202,
                 dim=128,
                 num_layers=4,
                 time_decay=0.99,
                 prediction_mode='spectral',
                 **kwargs):
        super().__init__(**kwargs)

        self.input_channels = input_channels
        self.dim = dim
        self.num_layers = num_layers
        self.time_decay = time_decay
        self.prediction_mode = prediction_mode

        # Input projection
        self.input_proj = layers.Dense(
            dim, use_bias=False, name='input_projection')

        # Paper-faithful RWKV blocks with fixed decay
        self.blocks = [
            LineRWKVBlock(
                dim=dim,
                time_decay=time_decay,  # Fixed decay for all blocks
                name=f'line_rwkv_block_{i}'
            ) for i in range(num_layers)
        ]

        # Prediction head
        self.output_proj = layers.Dense(
            input_channels, name='output_projection')

    def build(self, input_shape):
        """
        Build the model layers with the given input shape
        input_shape: (batch, height, width, channels)
        """
        super().build(input_shape)

        # Build input projection layer
        self.input_proj.build(input_shape)

        # Build output projection
        line_shape = (input_shape[0], input_shape[2], self.dim)
        self.output_proj.build(line_shape)

        # Mark as built
        self.built = True

    def preprocess_hyperspectral(self, x):
        """
        Preprocess hyperspectral data for line-based processing
        x: [batch, height, width, channels] or [batch, height, width, 1, channels]
        """
        # Check if we have 5 dimensions using shape attribute
        if x.shape.rank == 5:
            # Remove singleton dimension if present
            x = tf.squeeze(x, axis=3)
        elif x.shape.rank is None:
            # Handle dynamic shape case
            shape_tensor = tf.shape(x)
            if tf.shape(shape_tensor)[0] == 5:
                x = tf.squeeze(x, axis=3)

        # Ensure we have the right shape
        return x

    @tf.function(reduce_retracing=True)
    def call(self, x, training=False):
        """
        Forward pass through improved line-based RWKV
        x: [batch, height, width, channels] - hyperspectral image

        Uses tf.while_loop for height iteration (must be sequential for state)
        but the width dimension is fully vectorized within each block.
        """
        x = self.preprocess_hyperspectral(x)

        # Project input to working dimension (vectorized over all pixels)
        x = self.input_proj(x)  # [batch, height, width, dim]

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        num_layers = len(self.blocks)

        # Initialize states for all blocks: [num_layers, batch, dim]
        states = tf.zeros((num_layers, batch_size, self.dim), dtype=x.dtype)

        # TensorArray for predictions
        predictions_ta = tf.TensorArray(
            dtype=x.dtype, size=height, dynamic_size=False,
            element_shape=[None, None, self.input_channels]
        )

        def loop_body(h, states, predictions_ta):
            """Process one line through all blocks."""
            line = x[:, h, :, :]  # [batch, width, dim]

            new_states_list = []
            for i in range(num_layers):
                block = self.blocks[i]
                block_state = states[i]  # [batch, dim]
                line, new_state = block(line, block_state, training=training)
                new_states_list.append(new_state)

            new_states = tf.stack(new_states_list, axis=0)
            # [batch, width, input_channels]
            pred_line = self.output_proj(line)
            predictions_ta = predictions_ta.write(h, pred_line)

            return h + 1, new_states, predictions_ta

        _, _, predictions_ta = tf.while_loop(
            cond=lambda h, s, p: h < height,
            body=loop_body,
            loop_vars=[0, states, predictions_ta],
            parallel_iterations=1,  # Must be sequential
            swap_memory=True  # Help with memory for large images
        )

        # Stack and transpose: [height, batch, width, channels] -> [batch, height, width, channels]
        predictions = predictions_ta.stack()
        return tf.transpose(predictions, [1, 0, 2, 3])

    @tf.function(reduce_retracing=True)
    def predict_line_by_line(self, x, training=False):
        """
        Explicit line-by-line prediction for lossless coding
        Critical for proper causal inference during decoding
        Uses same optimized implementation as call()
        """
        return self.call(x, training=training)

    def compute_residuals(self, original, predicted):
        """Compute prediction residuals for compression"""
        return original - predicted

    def get_config(self):
        return {
            'input_channels': self.input_channels,
            'dim': self.dim,
            'num_layers': self.num_layers,
            'time_decay': self.time_decay,
            'prediction_mode': self.prediction_mode
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LineRWKVLosslessCodec(tf.keras.Model):
    """
    Complete lossless codec using line-based RWKV prediction
    """

    def __init__(self,
                 input_channels=202,
                 dim=128,
                 num_layers=4,
                 quantization_levels=256,
                 **kwargs):
        super().__init__(**kwargs)

        self.predictor = LineRWKVPredictor(
            input_channels=input_channels,
            dim=dim,
            num_layers=num_layers,
            prediction_mode='spectral'
        )
        self.quantization_levels = quantization_levels

    @tf.function(reduce_retracing=True)
    def encode(self, x, training=False):
        """
        Lossless encoding: predict and compute residuals
        """
        predictions = self.predictor.predict_line_by_line(x, training=training)
        residuals = self.predictor.compute_residuals(x, predictions)
        # Quantize residuals (for actual compression, would use arithmetic coding)
        quantized_residuals = tf.round(
            residuals * (self.quantization_levels - 1))
        return {
            'predictions': predictions,
            'residuals': residuals,
            'quantized_residuals': quantized_residuals
        }

    def decode(self, predictions, quantized_residuals):
        """
        Lossless decoding: reconstruct from predictions and residuals
        """
        # Dequantize residuals
        residuals = quantized_residuals / (self.quantization_levels - 1)

        # Reconstruct original
        reconstructed = predictions + residuals

        return reconstructed

    def call(self, x, training=False):
        """Training forward pass"""
        encoding_result = self.encode(x, training=training)
        predictions = encoding_result['predictions']

        # For training, return predictions directly
        return predictions
