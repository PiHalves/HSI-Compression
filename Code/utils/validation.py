import tensorflow as tf
import numpy as np
from metrics.metrics import *


def validate(model, val_dataset, criterion):

    loss_meter = tf.keras.metrics.Mean()
    psnr_meter = tf.keras.metrics.Mean()
    ssim_meter = tf.keras.metrics.Mean()
    sa_meter = tf.keras.metrics.Mean()
    mse_meter = tf.keras.metrics.Mean()
    
    from metrics.metrics import PeakSignalToNoiseRatio, StructualSimilarity, SpectralAngle
    psnr_metric = PeakSignalToNoiseRatio(max_val=1.0)
    ssim_metric = StructualSimilarity(data_range=1.0)
    sa_metric = SpectralAngle()

    for org in val_dataset:
        rec, latent = model(org, training=False)
        loss = criterion(org, rec)

        loss_meter.update_state(loss)
        mse_meter.update_state(tf.reduce_mean(tf.square(org - rec)))
        psnr_meter.update_state(psnr_metric(org, rec))
        ssim_meter.update_state(ssim_metric(org, rec))
        sa_meter.update_state(sa_metric(org, rec))

    
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    metrics_dict = {
        'val_mse': mse_meter.result().numpy(),
        'val_loss': loss_meter.result().numpy(),
        'val_psnr': psnr_meter.result().numpy(),
        'val_ssim': ssim_meter.result().numpy(),
        'val_sa': sa_meter.result().numpy()
    }
    
    print(f"Validation Loss: {metrics_dict['val_loss']:.6f}")
    print(f"Validation PSNR: {metrics_dict['val_psnr']:.2f} dB")
    print(f"Validation SSIM: {metrics_dict['val_ssim']:.4f}")
    print(f"Validation Spectral Angle: {metrics_dict['val_sa']:.2f}°")
    
    return metrics_dict