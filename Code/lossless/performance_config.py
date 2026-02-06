"""
Performance optimization configuration for LineRWKV
"""
import tensorflow as tf
import os


def is_tensorflow_initialized():
    """Check if TensorFlow has already been initialized"""
    try:
        # Try to access the current context
        context = tf.distribute.get_strategy()
        return True
    except:
        try:
            # Alternative check - see if any ops have been created
            tf.constant(1)
            return True
        except:
            return False


def configure_tensorflow_performance(enable_xla=True,
                                     enable_mixed_precision=False,
                                     inter_op_parallelism=0,  # 0 = use all available cores
                                     intra_op_parallelism=0,  # 0 = use all available cores
                                     allow_memory_growth=True,
                                     warn_on_late_config=True):
    """
    Configure TensorFlow for optimal performance

    Args:
        enable_xla: Enable XLA (Accelerated Linear Algebra) compilation
        enable_mixed_precision: Enable mixed precision training (float16/float32)
        inter_op_parallelism: Number of threads for inter-op parallelism (0 = auto)
        intra_op_parallelism: Number of threads for intra-op parallelism (0 = auto)
        allow_memory_growth: Allow GPU memory to grow as needed
        warn_on_late_config: Whether to warn if TensorFlow is already initialized
    """

    config_applied = {'threading': False, 'xla': False,
                      'gpu': False, 'mixed_precision': False}

    # Configure CPU parallelism (only if TensorFlow not initialized)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(
            inter_op_parallelism)
        tf.config.threading.set_intra_op_parallelism_threads(
            intra_op_parallelism)
        config_applied['threading'] = True
    except RuntimeError as e:
        if warn_on_late_config and "cannot be modified after initialization" in str(e):
            print(
                "WARNING: TensorFlow already initialized - threading configuration skipped")
            print(
                f"Current inter_op threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
            print(
                f"Current intra_op threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
        else:
            print(f"Threading configuration failed: {e}")

    # Enable XLA compilation for faster execution
    try:
        if enable_xla:
            tf.config.optimizer.set_jit(True)
            config_applied['xla'] = True
    except Exception as e:
        print(f"XLA configuration failed: {e}")

    # Configure GPU settings if available
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Found {len(physical_devices)} GPU(s)")
            for gpu in physical_devices:
                if allow_memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Configured GPU: {gpu}")
            config_applied['gpu'] = True
        else:
            print("No GPU found - using CPU with optimized threading")
            # CPU-specific optimizations
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
    except Exception as e:
        print(f"GPU configuration failed: {e}")

    # Mixed precision setup
    try:
        if enable_mixed_precision:
            if tf.config.list_physical_devices('GPU'):  # Only enable for GPU
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision enabled (float16)")
                config_applied['mixed_precision'] = True
            else:
                print("Mixed precision requires GPU - skipping")
    except Exception as e:
        print(f"Mixed precision configuration failed: {e}")

    # Additional optimizations
    try:
        tf.config.experimental.enable_op_determinism(
            False)  # Allow non-deterministic ops for speed
    except Exception as e:
        print(f"Op determinism configuration failed: {e}")

    print("TensorFlow performance configuration complete")
    return config_applied


def create_optimized_dataset(dataset, prefetch_size=tf.data.AUTOTUNE,
                             num_parallel_calls=tf.data.AUTOTUNE):
    """
    Apply performance optimizations to tf.data dataset

    Args:
        dataset: Input tf.data.Dataset
        prefetch_size: Number of elements to prefetch
        num_parallel_calls: Number of parallel calls for map operations
    """
    return (dataset
            .prefetch(prefetch_size)
            .cache()  # Cache dataset in memory if it fits
            )


def get_cpu_count():
    """Get number of CPU cores"""
    return os.cpu_count()


def print_performance_info():
    """Print current performance configuration"""
    print("=== Performance Configuration ===")
    print(f"CPU cores: {get_cpu_count()}")
    print(
        f"Inter-op parallelism: {tf.config.threading.get_inter_op_parallelism_threads()}")
    print(
        f"Intra-op parallelism: {tf.config.threading.get_intra_op_parallelism_threads()}")
    print(f"XLA enabled: {tf.config.optimizer.get_jit()}")
    print(f"Mixed precision: {tf.keras.mixed_precision.global_policy().name}")

    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    print("=================================")


# Default configuration for immediate use
def quick_setup():
    """Quick performance setup with sensible defaults"""
    config_applied = configure_tensorflow_performance(
        enable_xla=True,
        enable_mixed_precision=False,  # Conservative default
        inter_op_parallelism=0,  # Use all cores
        intra_op_parallelism=0,  # Use all cores
        allow_memory_growth=True
    )
    print_performance_info()
    return config_applied


def setup_early_tensorflow_config():
    """
    Configure TensorFlow settings that MUST be set before any TF operations.
    Call this at the very beginning of your script, before importing models.
    """
    import os

    # Set environment variables before TensorFlow initialization
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging

    # Import tensorflow after setting env vars
    import tensorflow as tf

    # Configure threading (this must be done before any TF operations)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(
            0)  # Use all cores
        tf.config.threading.set_intra_op_parallelism_threads(
            0)  # Use all cores
        print("Early TensorFlow threading configuration applied successfully")
        return True
    except Exception as e:
        print(f"Early configuration failed: {e}")
        return False


def apply_late_optimizations(enable_xla=True, enable_mixed_precision=False, allow_memory_growth=True):
    """
    Apply optimizations that can be set after TensorFlow initialization
    """
    config_applied = {'xla': False, 'gpu': False, 'mixed_precision': False}

    # Enable XLA compilation
    try:
        if enable_xla:
            tf.config.optimizer.set_jit(True)
            config_applied['xla'] = True
            print("XLA optimization enabled")
    except Exception as e:
        print(f"XLA configuration failed: {e}")

    # Configure GPU settings
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Found {len(physical_devices)} GPU(s)")
            for gpu in physical_devices:
                if allow_memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Configured GPU: {gpu}")
            config_applied['gpu'] = True
        else:
            print("No GPU found - using CPU")
    except Exception as e:
        print(f"GPU configuration failed: {e}")

    # Mixed precision setup
    try:
        if enable_mixed_precision and tf.config.list_physical_devices('GPU'):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled (float16)")
            config_applied['mixed_precision'] = True
        elif enable_mixed_precision:
            print("Mixed precision requires GPU - skipping")
    except Exception as e:
        print(f"Mixed precision configuration failed: {e}")

    return config_applied
