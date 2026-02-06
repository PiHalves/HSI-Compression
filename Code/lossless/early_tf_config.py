"""
Early TensorFlow configuration - import this FIRST in your training scripts
before importing any TensorFlow models or operations.

Usage:
    from Code.lossless.early_tf_config import setup_tensorflow
    setup_tensorflow()
    
    # Now import your models and other TensorFlow code
    from Code.lossless.lineRWKV_trainer import create_line_rwkv_trainer
"""

import os


def setup_tensorflow(num_threads=None, enable_xla=True, reduce_logging=True):
    """
    Configure TensorFlow settings that must be set before first TF operation.

    Args:
        num_threads: Number of threads to use (None = use all cores)
        enable_xla: Whether to enable XLA compilation
        reduce_logging: Whether to reduce TensorFlow logging output
    """

    print("Setting up TensorFlow configuration...")

    # Set environment variables first (before importing TF)
    if reduce_logging:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if enable_xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    # Now import TensorFlow
    import tensorflow as tf

    # Configure threading before any TF operations
    if num_threads is None:
        num_threads = 0  # Use all available cores

    try:
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        print(
            f"Threading configured: using {'all cores' if num_threads == 0 else f'{num_threads} threads'}")
    except RuntimeError as e:
        print(f"Threading configuration failed: {e}")
        print("TensorFlow may already be initialized")

    # Configure XLA
    if enable_xla:
        try:
            tf.config.optimizer.set_jit(True)
            print("XLA compilation enabled")
        except Exception as e:
            print(f"XLA configuration failed: {e}")

    # Configure GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            print("No GPU detected - using CPU optimization")
    except Exception as e:
        print(f"GPU configuration failed: {e}")

    # Disable op determinism for better performance
    try:
        tf.config.experimental.enable_op_determinism(False)
        print("Non-deterministic operations enabled for performance")
    except Exception as e:
        print(f"Determinism configuration failed: {e}")

    print("TensorFlow configuration complete\n")
    return tf


# Auto-setup if imported directly
if __name__ == "__main__":
    setup_tensorflow()
