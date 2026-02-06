import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

@keras.utils.register_keras_serializable()
class ResidualConv3DAutoencoder(Model):
    def __init__(
        self, 
        src_channels: int = 202, 
        latent_channels: int = 32,
        name: str = "res_cae3D",
        **kwargs
    ):
        super().__init__(name=name,**kwargs)
        self.src_channels = src_channels
        self.latent_channels = latent_channels
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _residual_block(self, filters, kernel_size=(3, 3, 3)):
        def block(x):
            residual = x
            
            if residual.shape[-1] != filters:
                residual = layers.Conv3D(filters, 1, padding='same')(residual)
                residual = layers.BatchNormalization()(residual)

            x = layers.Conv3D(filters, kernel_size, padding='same', 
                            activation='leaky_relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)

            x = layers.Conv3D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            
            x = layers.Add()([x, residual])
            return layers.LeakyReLU()(x) 
        return block

    def get_config(self):
        config=({
            'src_channels': self.src_channels,
            'latent_channels': self.latent_channels,
            'name': self.name
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def _build_encoder(self):
        inputs = layers.Input(shape=(None, None, None, self.src_channels))

        x = layers.Conv3D(128, (3, 3, 1), strides=(2, 2, 1), padding='same',
                          use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = self._residual_block(128, kernel_size=(3, 3, 1))(x)
        
        x = layers.Conv3D(64, (3, 3, 1), strides=(2, 2, 1), padding='same',
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = self._residual_block(64, kernel_size=(3, 3, 1))(x)
        

        latent = layers.Conv3D(self.latent_channels, (3, 3, 1), strides=(1, 1, 1),
                              padding='same', activation='sigmoid')(x)
        
        return Model(inputs, latent, name='encoder')
    
    def _build_decoder(self):
        inputs = layers.Input(shape=(None, None, None, self.latent_channels))

        x = layers.Conv3DTranspose(64, (3, 3, 1), strides=(2, 2, 1), padding='same',
                                  use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = self._residual_block(64, kernel_size=(3, 3, 1))(x)
        
        x = layers.Conv3DTranspose(128, (3, 3, 1), strides=(1, 1, 1), padding='same',
                                  use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = self._residual_block(128, kernel_size=(3, 3, 1))(x)
        

        x = layers.Conv3DTranspose(self.src_channels, (3, 3, 1), strides=(2, 2, 1), padding='same',
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        outputs = layers.Conv3DTranspose(self.src_channels, (3, 3, 1), strides=(1, 1, 1),
                                        padding='same', activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='decoder')
    
    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)
    
    def decode(self, latent, training=False):
        return self.decoder(latent, training=training)
    
    def call(self, inputs, training=False):
        latent = self.encoder(inputs, training=training)
        reconstructed = self.decoder(latent, training=training)
        return reconstructed, latent
    
if __name__ == "__main__":
    model = ResidualConv3DAutoencoder(src_channels=202, latent_channels=32)

    test_input = tf.random.normal([4, 128, 128, 1, 202])  # (batch, height, width, depth=1, channels=202)

    _ = model(test_input, training=False) 

    reconstructed, latent = model(test_input, training=False)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
