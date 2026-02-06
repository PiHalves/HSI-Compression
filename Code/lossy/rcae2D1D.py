import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class SpatialReshape(layers.Layer):
    """Custom layer to reshape between spatial and flattened representations while tracking dimensions."""
    def __init__(self, channels=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
    
    def call(self, inputs):
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config

@keras.utils.register_keras_serializable()
class rcae2D1D(Model):
    def __init__(
        self, 
        src_channels: int = 202, 
        latent_channels: int = 32,
        name: str = "cae2D1D",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.src_channels = src_channels
        self.latent_channels = latent_channels
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _spatial_residual_block(self, filters, kernel_size=(3, 3)):
        def block(x):
            residual = x
            
            # 2D spatial convolutions
            x = layers.Conv2D(filters, kernel_size, padding='same', 
                            activation='leaky_relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            if residual.shape[-1] != filters:
                residual = layers.Conv2D(filters, 1, padding='same')(residual)
            
            x = layers.Add()([x, residual])
            return layers.LeakyReLU()(x) 
        return block
    
    def _spectral_residual_block(self, filters, kernel_size=3):
        def block(x):
            residual = x
            
            # 1D spectral convolutions (treat spectral dimension as sequence)
            x = layers.Conv1D(filters, kernel_size, padding='same',
                            activation='leaky_relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            if residual.shape[-1] != filters:
                residual = layers.Conv1D(filters, 1, padding='same')(residual)
            x = layers.Add()([x, residual])
            return layers.LeakyReLU()(x) 
        return block

    def get_config(self):
        config = {
            'src_channels': self.src_channels,
            'latent_channels': self.latent_channels,
            'name': self.name
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def _build_encoder(self):
        inputs = layers.Input(shape=(None, None, self.src_channels))

        x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._spatial_residual_block(128)(x)

        x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._spatial_residual_block(64)(x)

        h = layers.Lambda(lambda x: tf.fill([tf.shape(x)[0]], tf.shape(x)[1]))(x)
        w = layers.Lambda(lambda x: tf.fill([tf.shape(x)[0]], tf.shape(x)[2]))(x)

        x = layers.Reshape((-1, 64))(x)

        x = layers.Conv1D(64, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._spectral_residual_block(64)(x)

        x = layers.Conv1D(self.latent_channels, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        latent = layers.Activation('sigmoid')(x)

        return Model(inputs, [latent, h, w], name="encoder")

    
    def _build_decoder(self):
        latent_input = layers.Input(shape=(None, self.latent_channels))
        h_input = layers.Input(shape=(), dtype=tf.int32)
        w_input = layers.Input(shape=(), dtype=tf.int32)

        # ----- Spectral decoder -----
        x = layers.Conv1DTranspose(64, 3, strides=2, padding='same', use_bias=False)(latent_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._spectral_residual_block(64)(x)

        # ----- Sequence → spatial -----
        def reshape_back(args):
            x, h, w = args
            batch = tf.shape(x)[0]
            return tf.reshape(x, [batch, h, w, 64])

        x = SequenceToSpatial(64)([x, h_input, w_input])


        # ----- Spatial decoder -----
        x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._spatial_residual_block(128)(x)

        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        outputs = layers.Conv2D(self.src_channels, 3, padding='same', activation='sigmoid')(x)

        return Model([latent_input, h_input, w_input], outputs, name="decoder")

    
    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)
    
    def decode(self, latent, h, w, training=False):
        return self.decoder([latent, h, w], training=training)

    
    def call(self, inputs, training=False):
        latent, h, w = self.encoder(inputs, training=training)
        reconstructed = self.decoder([latent, h, w], training=training)
        return reconstructed , latent

@keras.utils.register_keras_serializable()
class SequenceToSpatial(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def call(self, inputs):
        x, h, w = inputs
        batch = tf.shape(x)[0]
        return tf.reshape(x, [batch, h[0], w[0], self.channels])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], None, None, self.channels)

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config




if __name__ == "__main__":
    model = cae2D1D(src_channels=202, latent_channels=32)
    
    
    test_input = tf.random.normal([4, 128, 128, 202])  
    
    _ = model(test_input, training=False)  # Build the model
    
    reconstructed, latent = model(test_input, training=False)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"\nModel summary:")
    model.summary()