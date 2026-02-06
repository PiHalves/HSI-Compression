
from tensorflow import keras
import tensorflow as tf
from keras import layers, Model


class small_segmenter(Model):
    """A smaller segmentation model for hyperspectral data.

    Takes 3D hyperspectral input (H, W, D, 1) and outputs 2D spatial 
    segmentation mask (H, W, num_classes).

    Architecture:
    - 3D convolutions to capture spatial-spectral features
    - Progressive spectral dimension reduction via strided convs
    - Final 2D convolutions (1x1) for per-pixel classification

    Memory estimate for batch_size=1, input (128, 128, 202, 1):
    - With base_filters=8, depth=3: ~1-2 GB
    - With base_filters=16, depth=3: ~3-4 GB
    """

    def __init__(
        self,
        input_shape: tuple = (128, 128, 202, 1),
        num_classes: int = 4,
        base_filters: int = 8,
        depth: int = 3,
        dropout_rate: float = 0.1,
        name: str = "small_segmenter",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.depth = depth
        self.dropout_rate = dropout_rate

        # Build the model layers
        self._build_layers()

    def _build_layers(self):
        """Build all model layers."""

        # Initial 3D conv to extract low-level features
        # Kernel (3,3,7) - small spatial, larger spectral receptive field
        self.initial_conv = layers.Conv3D(
            filters=self.base_filters,
            kernel_size=(3, 3, 7),
            padding='same',
            activation=None,
            kernel_initializer='he_normal',
            name='initial_conv3d'
        )
        self.initial_bn = layers.BatchNormalization(name='initial_bn')
        self.initial_act = layers.ReLU(name='initial_relu')

        # Encoder blocks: 3D convs with spectral pooling
        self.encoder_convs = []
        self.encoder_bns = []
        self.encoder_acts = []
        self.spectral_pools = []

        for i in range(self.depth):
            filters = self.base_filters * (2 ** i)

            # 3D conv block
            self.encoder_convs.append(layers.Conv3D(
                filters=filters,
                kernel_size=(3, 3, 5),
                padding='same',
                activation=None,
                kernel_initializer='he_normal',
                name=f'encoder_conv3d_{i}'
            ))
            self.encoder_bns.append(
                layers.BatchNormalization(name=f'encoder_bn_{i}'))
            self.encoder_acts.append(layers.ReLU(name=f'encoder_relu_{i}'))

            # Pool only in spectral dimension (keep spatial resolution)
            self.spectral_pools.append(layers.MaxPool3D(
                pool_size=(1, 1, 2),
                strides=(1, 1, 2),
                padding='same',
                name=f'spectral_pool_{i}'
            ))

        # Spectral collapse: 1x1x1 conv + mean over spectral axis
        self.collapse_conv = layers.Conv3D(
            filters=self.base_filters * (2 ** self.depth),
            kernel_size=(1, 1, 1),
            padding='same',
            activation=None,
            kernel_initializer='he_normal',
            name='collapse_conv3d'
        )
        self.collapse_bn = layers.BatchNormalization(name='collapse_bn')
        self.collapse_act = layers.ReLU(name='collapse_relu')

        # 2D refinement convolutions (operate on H x W)
        self.refine_conv1 = layers.Conv2D(
            filters=self.base_filters * 4,
            kernel_size=(3, 3),
            padding='same',
            activation=None,
            kernel_initializer='he_normal',
            name='refine_conv2d_1'
        )
        self.refine_bn1 = layers.BatchNormalization(name='refine_bn_1')
        self.refine_act1 = layers.ReLU(name='refine_relu_1')

        self.refine_conv2 = layers.Conv2D(
            filters=self.base_filters * 2,
            kernel_size=(3, 3),
            padding='same',
            activation=None,
            kernel_initializer='he_normal',
            name='refine_conv2d_2'
        )
        self.refine_bn2 = layers.BatchNormalization(name='refine_bn_2')
        self.refine_act2 = layers.ReLU(name='refine_relu_2')

        # Dropout for regularization
        self.dropout = layers.Dropout(self.dropout_rate, name='dropout')

        # Final classification layer (1x1 conv = dense per pixel)
        self.classifier = layers.Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            padding='same',
            activation='softmax',
            kernel_initializer='glorot_uniform',
            name='classifier'
        )

    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs: Tensor of shape (batch, H, W, D, 1)
            training: Boolean for training mode

        Returns:
            Tensor of shape (batch, H, W, num_classes)
        """
        # Initial feature extraction
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_act(x)

        # Encoder: 3D convs with spectral pooling
        for i in range(self.depth):
            x = self.encoder_convs[i](x)
            x = self.encoder_bns[i](x, training=training)
            x = self.encoder_acts[i](x)
            x = self.spectral_pools[i](x)

        # Collapse spectral dimension without creating layers in call()
        # (batch, H, W, D, C) -> (batch, H, W, D, C)
        x = self.collapse_conv(x)
        x = self.collapse_bn(x, training=training)
        x = self.collapse_act(x)

        # Remove spectral dimension: (batch, H, W, D, C) -> (batch, H, W, C)
        x = tf.reduce_mean(x, axis=-2)

        # 2D refinement
        x = self.refine_conv1(x)
        x = self.refine_bn1(x, training=training)
        x = self.refine_act1(x)

        x = self.refine_conv2(x)
        x = self.refine_bn2(x, training=training)
        x = self.refine_act2(x)

        # Dropout
        x = self.dropout(x, training=training)

        # Per-pixel classification
        output = self.classifier(x)

        return output

    def get_config(self):
        """Return model configuration."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_,
            'num_classes': self.num_classes,
            'base_filters': self.base_filters,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
        })
        return config

    def build_graph(self):
        """Build model graph for summary visualization."""
        x = keras.Input(shape=self.input_shape_, name='input')
        return Model(inputs=[x], outputs=self.call(x), name=self.name)


def estimate_memory_usage(model, batch_size=1, input_shape=(128, 128, 202, 1)):
    """Estimate GPU memory usage for the model.

    Args:
        model: The Keras model
        batch_size: Batch size for estimation
        input_shape: Input shape (H, W, D, C)

    Returns:
        Dictionary with memory estimates
    """
    # Count parameters
    total_params = model.count_params()
    param_memory_mb = (total_params * 4) / (1024 ** 2)  # float32 = 4 bytes

    # Estimate activation memory (rough estimate)
    # This is highly approximate - actual memory depends on TF internals
    h, w, d, c = input_shape
    activation_elements = batch_size * h * w * \
        d * model.base_filters * 4  # rough estimate
    activation_memory_mb = (activation_elements * 4) / (1024 ** 2)

    total_estimate_mb = param_memory_mb + activation_memory_mb

    return {
        'total_params': total_params,
        'param_memory_mb': param_memory_mb,
        'activation_memory_mb': activation_memory_mb,
        'total_estimate_mb': total_estimate_mb,
        'total_estimate_gb': total_estimate_mb / 1024
    }


if __name__ == "__main__":
    # Test the model
    print("Creating small_segmenter model...")

    model = small_segmenter(
        input_shape=(128, 128, 202, 1),
        num_classes=4,
        base_filters=8,
        depth=3,
        dropout_rate=0.1
    )

    # Build model with dummy input
    dummy_input = tf.random.normal((1, 128, 128, 202, 1))
    output = model(dummy_input, training=False)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output: (1, 128, 128, {model.num_classes})")

    # Model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    model.summary()

    # Memory estimation
    print("\n" + "="*60)
    print("MEMORY ESTIMATION")
    print("="*60)
    mem_info = estimate_memory_usage(model, batch_size=1)
    print(f"Total parameters: {mem_info['total_params']:,}")
    print(f"Parameter memory: {mem_info['param_memory_mb']:.2f} MB")
    print(
        f"Estimated activation memory: {mem_info['activation_memory_mb']:.2f} MB")
    print(f"Total estimate: {mem_info['total_estimate_gb']:.2f} GB")

    # Test with larger batch
    print("\n" + "="*60)
    print("BATCH SIZE MEMORY SCALING")
    print("="*60)
    for bs in [1, 2, 4, 8]:
        mem = estimate_memory_usage(model, batch_size=bs)
        print(f"Batch size {bs}: ~{mem['total_estimate_gb']:.2f} GB")
