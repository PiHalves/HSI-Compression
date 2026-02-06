# From https://www.mdpi.com/2072-4292/15/18/4422

import tensorflow as tf
import numpy as np
import math

# Import KLT functions for spectral decorrelation
from nonML.KLT import apply_klt_tf, inverse_klt

# ======================================================
# Hyperparameters
# ======================================================
IMAGE_SIZE = 128
CHANNELS = 202
LATENT_CHANNELS = 64
HYPER_CHANNELS = 32
LD = 0.01
BATCH_SIZE = 4
EPOCHS = 10

# ======================================================
# Custom Constraint for GDN
# ======================================================


class NonNegativeMinimum(tf.keras.constraints.Constraint):
    """Constrains weights to be greater than or equal to a minimum value."""

    def __init__(self, min_value=1e-6):
        self.min_value = min_value

    def __call__(self, w):
        return tf.maximum(w, self.min_value)

    def get_config(self):
        return {"min_value": self.min_value}


# ======================================================
# GDN / IGDN
# ======================================================


class GDN(tf.keras.layers.Layer):
    def __init__(self, inverse=False, epsilon=1e-6):
        super().__init__()
        self.inverse = inverse
        self.epsilon = epsilon

    def build(self, input_shape):
        C = input_shape[-1]
        self.beta = self.add_weight(shape=(C,), initializer="ones",
                                    constraint=NonNegativeMinimum(self.epsilon))
        self.gamma = self.add_weight(shape=(C, C), initializer="identity",
                                     constraint=NonNegativeMinimum(self.epsilon))

    @tf.function
    def call(self, x):
        gamma = tf.reshape(self.gamma, (1, 1, -1, x.shape[-1]))
        beta = tf.reshape(self.beta, (1, 1, 1, -1))
        norm = tf.nn.conv2d(tf.square(x), gamma, 1, "SAME")
        norm = tf.sqrt(norm + beta)
        return x * norm if self.inverse else x / norm


class IGDN(GDN):
    def __init__(self):
        super().__init__(inverse=True)

# ======================================================
# Masked Convolution (Context Model)
# ======================================================


class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=5):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        kh = kw = self.kernel_size
        in_ch = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(kh, kw, in_ch, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Mask (causal, no future pixels) - stored as non-trainable weight
        mask = np.ones((kh, kw, in_ch, self.filters), dtype=np.float32)
        mask[kh//2, kw//2 + 1:, :, :] = 0
        mask[kh//2 + 1:, :, :, :] = 0
        self.mask = self.add_weight(
            shape=(kh, kw, in_ch, self.filters),
            initializer=tf.constant_initializer(mask),
            trainable=False,
        )

        self.bias = self.add_weight(
            shape=(self.filters,), initializer="zeros", trainable=True
        )

    @tf.function
    def call(self, x):
        masked_kernel = self.kernel * self.mask
        x = tf.nn.conv2d(x, masked_kernel, 1, "SAME")
        return x + self.bias

# ======================================================
# Quantization
# ======================================================


def quantize(x, training):
    if training:
        return x + tf.random.uniform(tf.shape(x), -0.5, 0.5)
    return tf.round(x)

# ======================================================
# Gaussian Likelihood
# ======================================================


def gaussian_likelihood(y, mu, sigma):
    sigma = tf.maximum(sigma, 1e-6)
    return (1.0 / tf.sqrt(2.0 * math.pi * sigma**2)) * \
        tf.exp(-0.5 * ((y - mu) / sigma)**2)

# ======================================================
# Encoder / Decoder
# ======================================================


# ======================================================
# Hyperprior Networks
# ======================================================


def build_hyper_analysis():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(HYPER_CHANNELS, 3, 1,
                               'same', activation='relu'),
        tf.keras.layers.Conv2D(HYPER_CHANNELS, 5, 2,
                               'same', activation='relu'),
        tf.keras.layers.Conv2D(HYPER_CHANNELS, 5, 2, 'same'),
    ])


def build_hyper_synthesis():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(
            HYPER_CHANNELS, 5, 2, 'same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(
            HYPER_CHANNELS, 5, 2, 'same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(LATENT_CHANNELS * 2, 3, 1, 'same'),
    ])

# ======================================================
# Full Model (Hyperprior + Context)
# ======================================================


class FullCompressor(tf.keras.Model):
    def __init__(self, img_height=IMAGE_SIZE, img_width=IMAGE_SIZE, bands=CHANNELS,
                 klt_matrix=None, klt_mean=None, n_klt_components=None):
        """
        Initialize FullCompressor with optional KLT preprocessing.

        Args:
            img_height: Height of input images
            img_width: Width of input images
            bands: Number of spectral bands in input
            klt_matrix: KLT transformation matrix [n_components, bands]. If provided,
                       KLT is applied as first step of encoder.
            klt_mean: Mean vector for KLT [bands]. Required if klt_matrix is provided.
            n_klt_components: Number of KLT components (inferred from klt_matrix if not provided)
        """
        super().__init__()
        self.input_channels = bands  # Original input channels

        # KLT configuration
        self.use_klt = klt_matrix is not None
        if self.use_klt:
            if klt_mean is None:
                raise ValueError(
                    "klt_mean must be provided when klt_matrix is provided")
            # Store KLT parameters as non-trainable weights
            self.klt_matrix = tf.Variable(
                klt_matrix, trainable=False, dtype=tf.float32, name='klt_matrix')
            self.klt_mean = tf.Variable(
                klt_mean, trainable=False, dtype=tf.float32, name='klt_mean')
            # Number of channels after KLT transform
            self.channels = n_klt_components if n_klt_components else klt_matrix.shape[0]
            print(f"KLT enabled: {bands} -> {self.channels} components")
        else:
            self.klt_matrix = None
            self.klt_mean = None
            self.channels = bands  # Use the bands parameter, not the global CHANNELS

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.hyper_analysis = build_hyper_analysis()
        self.hyper_synthesis = build_hyper_synthesis()
        self.context = MaskedConv2D(LATENT_CHANNELS * 2)

    @tf.function
    def call(self, x, training=False):
        # Apply KLT as first step of encoder (spectral decorrelation)
        if self.use_klt:
            x_klt = apply_klt_tf(x, self.klt_matrix, self.klt_mean)
        else:
            x_klt = x

        y = self.encoder(x_klt)
        z = self.hyper_analysis(y)

        y_q = quantize(y, training)
        z_q = quantize(z, training)

        # Hyperprior μ, σ
        params_h = self.hyper_synthesis(z_q)
        mu_h, sigma_h = tf.split(params_h, 2, axis=-1)
        sigma_h = tf.nn.softplus(sigma_h)

        # Context μ, σ
        params_c = self.context(y_q)
        mu_c, sigma_c = tf.split(params_c, 2, axis=-1)
        sigma_c = tf.nn.softplus(sigma_c)

        # Combine
        mu = mu_h + mu_c
        sigma = sigma_h * sigma_c

        x_klt_hat = self.decoder(y_q)

        # Apply inverse KLT as last step of decoder (spectral reconstruction)
        if self.use_klt:
            x_hat = inverse_klt(x_klt_hat, self.klt_matrix, self.klt_mean)
        else:
            x_hat = x_klt_hat

        return x_hat, y, mu, sigma

    def build_encoder(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.channels, 5, 2,
                                   'same', activation='relu'), GDN(),
            tf.keras.layers.Conv2D(32*(self.channels+1),
                                   5, 2, 'same', activation='relu'), GDN(),
            tf.keras.layers.Conv2D(32*(self.channels+1),
                                   5, 2, 'same', activation='relu'), GDN(),
            tf.keras.layers.Conv2D(LATENT_CHANNELS, 5, 2, 'same'),
        ])

    def build_decoder(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                32*(self.channels+1), 5, 2, 'same'), IGDN(),
            tf.keras.layers.Conv2DTranspose(
                32*(self.channels+1), 5, 2, 'same'), IGDN(),
            tf.keras.layers.Conv2DTranspose(
                32*(self.channels+1), 5, 2, 'same'), IGDN(),
            tf.keras.layers.Conv2DTranspose(self.channels, 5, 2, 'same'),
        ])
# ======================================================
# Loss
# ======================================================


@tf.function
def rate_distortion_loss(x, x_hat, y, mu, sigma):
    distortion = tf.reduce_mean(tf.square(x - x_hat))
    likelihood = gaussian_likelihood(y, mu, sigma)
    rate = tf.reduce_mean(-tf.math.log(likelihood + 1e-9)) / math.log(2)
    return distortion + LD * rate, distortion, rate


def load_klt_from_file(filepath):
    """
    Load KLT matrix and mean vector from a saved .npz file.

    Args:
        filepath: Path to the .npz file containing 'klt_matrix' and 'mean_vec'

    Returns:
        klt_matrix: Principal components [n_components, n_bands]
        mean_vec: Mean vector [n_bands]
    """
    data = np.load(filepath)
    return data['klt_matrix'], data['mean_vec']


@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        x_hat, y, mu, sigma = model(x, training=True)
        loss, d, r = rate_distortion_loss(x, x_hat, y, mu, sigma)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, d, r


if __name__ == "__main__":
    def fake_dataset(n=100):
        for _ in range(n):
            yield np.random.rand(IMAGE_SIZE, IMAGE_SIZE, CHANNELS).astype(np.float32)

    dataset = tf.data.Dataset.from_generator(
        fake_dataset,
        output_signature=tf.TensorSpec(
            shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            dtype=tf.float32
        )
    ).batch(BATCH_SIZE)
    print("✅ Fake dataset created.")

    # Example 1: Without KLT (original behavior)
    print("\n--- Example 1: Without KLT ---")
    model = FullCompressor()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    print("✅ Full hyperprior + context model initialized (no KLT).")
    dummy_input = tf.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    _ = model(dummy_input, training=False)

    # Example 2: With KLT preprocessing
    print("\n--- Example 2: With KLT preprocessing ---")
    # Create fake KLT matrix (in practice, train this using train_klt from KLT.py)
    n_klt_components = 24
    klt_matrix = np.random.randn(n_klt_components, CHANNELS).astype(np.float32)
    klt_mean = np.random.randn(CHANNELS).astype(np.float32)

    model_with_klt = FullCompressor(
        bands=CHANNELS,
        klt_matrix=klt_matrix,
        klt_mean=klt_mean,
        n_klt_components=n_klt_components
    )
    optimizer = tf.keras.optimizers.Adam(1e-4)
    print("✅ Full hyperprior + context model initialized with KLT.")

    # Build the model by calling it with a dummy input
    dummy_input = tf.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    _ = model_with_klt(dummy_input, training=False)

    print("🚀 Starting Full hyperprior + context model training (with KLT)...")
    for epoch in range(EPOCHS):
        for batch in dataset:
            with tf.GradientTape() as tape:
                x_hat, y, mu, sigma = model_with_klt(batch, training=True)
                loss, d, r = rate_distortion_loss(batch, x_hat, y, mu, sigma)
            grads = tape.gradient(loss, model_with_klt.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model_with_klt.trainable_variables))
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, D={d:.4f}, R={r:.4f}")

    print("✅ Full hyperprior + context model training complete.")
