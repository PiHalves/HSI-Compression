import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA, IncrementalPCA
import time


def train_klt(hsi_dataset, n_components=24):
    """
    hsi_dataset: list of HSI cubes [H,W,B]
    """
    pixels = []
    for hsi in hsi_dataset:
        H, W, B = hsi.shape
        pixels.append(hsi.reshape(-1, B))
    pixels = np.vstack(pixels)

    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(pixels)

    return pca.components_, pca.mean_


def train_klt_incremental(hsi_generator, n_components=24, batch_size=10000):
    """
    Train KLT incrementally without loading entire dataset into memory.

    Args:
        hsi_generator: Generator or iterable yielding HSI cubes [H,W,B]
        n_components: Number of principal components to keep
        batch_size: Number of pixels per batch for incremental fitting

    Returns:
        klt_matrix: Principal components [n_components, n_bands]
        mean_vec: Mean vector [n_bands]
    """
    ipca = IncrementalPCA(n_components=n_components)

    pixel_buffer = []
    buffer_size = 0
    start = time.time()
    for idx, hsi in enumerate(hsi_generator):
        if idx % 32 == 0:
            elapsed = time.time() - start
            to_go = (len(hsi_generator) - idx) * (elapsed / (idx + 1))
            print(
                f"Processing HSI cube {idx}/{len(hsi_generator)}, elapsed time: {elapsed:.2f}s to go: {to_go:.2f}s...")
        hsi = hsi.numpy()  # Convert from Tensor to NumPy array if needed
        hsi = hsi[0]
        # print("processing hsi shape: ", hsi.shape)
        H, W, B = hsi.shape
        pixels = hsi.reshape(-1, B)
        pixel_buffer.append(pixels)
        buffer_size += pixels.shape[0]

        # When buffer is large enough, fit a batch
        while buffer_size >= batch_size:
            print("Fitting batch of size:", batch_size)
            all_pixels = np.vstack(pixel_buffer)
            batch = all_pixels[:batch_size]
            remaining = all_pixels[batch_size:]

            ipca.partial_fit(batch)

            if remaining.shape[0] > 0:
                pixel_buffer = [remaining]
                buffer_size = remaining.shape[0]
            else:
                pixel_buffer = []
                buffer_size = 0

    # Fit any remaining pixels
    if buffer_size > 0:
        remaining_pixels = np.vstack(pixel_buffer)
        # IncrementalPCA requires at least n_components samples
        if remaining_pixels.shape[0] >= n_components:
            ipca.partial_fit(remaining_pixels)

    return ipca.components_, ipca.mean_


def train_klt_from_files(dataloader, load_fn, n_components=24, batch_size=10000):
    """
    Train KLT from file paths without loading all files at once.

    Args:
        dataloader: Iterable or generator yielding HSI cubes [H,W,B]
        load_fn: Function that takes a path and returns an HSI cube [H,W,B]
        n_components: Number of principal components
        batch_size: Pixels per batch

    Returns:
        klt_matrix, mean_vec
    """

    return train_klt_incremental(dataloader, n_components, batch_size)


def apply_klt_tf(hsi, klt_matrix, mean_vec):
    """
    Apply KLT transform to hyperspectral image.

    Args:
        hsi: Input tensor of shape [H, W, B] or [batch, H, W, B]
        klt_matrix: KLT transformation matrix [n_components, n_bands]
        mean_vec: Mean vector [n_bands]

    Returns:
        Transformed tensor with last dimension = n_components
    """
    dtype = hsi.dtype
    klt_matrix = tf.cast(klt_matrix, dtype)
    mean_vec = tf.cast(mean_vec, dtype)
    hsi = hsi - mean_vec
    # Use axis -1 (last axis) for bands, works for both 3D and 4D tensors
    # klt_matrix is [n_components, n_bands], we contract on bands (axis 1 of klt_matrix)
    return tf.tensordot(hsi, tf.transpose(klt_matrix), axes=[[-1], [0]])


def inverse_klt(hsi_k, klt_matrix, mean_vec):
    """
    Apply inverse KLT transform to recover original spectral representation.

    Args:
        hsi_k: KLT-transformed tensor of shape [H, W, K] or [batch, H, W, K]
        klt_matrix: KLT transformation matrix [n_components, n_bands]
        mean_vec: Mean vector [n_bands]

    Returns:
        Reconstructed tensor with last dimension = n_bands
    """
    dtype = hsi_k.dtype
    klt_matrix = tf.cast(klt_matrix, dtype)
    mean_vec = tf.cast(mean_vec, dtype)
    # klt_matrix is [n_components, n_bands], we contract on components (axis 0)
    hsi_recon = tf.tensordot(hsi_k, klt_matrix, axes=[[-1], [0]])
    hsi_recon = hsi_recon + mean_vec
    return hsi_recon


def cluster_klt_bands(hsi_k, cluster_size=3):
    clusters = []
    for i in range(0, hsi_k.shape[-1], cluster_size):
        clusters.append(hsi_k[..., i:i+cluster_size])
    return clusters


def inverse_cluster_klt(clusters):
    return tf.concat(clusters, axis=-1)


def save_klt_matrix(klt_matrix, mean_vec, filepath):
    np.savez(filepath, klt_matrix=klt_matrix, mean_vec=mean_vec)


if __name__ == "__main__":
    # Example usage
    hsi_example = np.random.rand(128, 128, 202).astype(np.float32)
    hsi_dataset = [hsi_example for _ in range(10)]

    klt_matrix, mean_vec = train_klt(hsi_dataset, n_components=24)

    hsi_tf = tf.convert_to_tensor(hsi_example)
    hsi_k = apply_klt_tf(hsi_tf, klt_matrix, mean_vec)

    clusters = cluster_klt_bands(hsi_k, cluster_size=3)
    hsi_k_recon = inverse_cluster_klt(clusters)

    hsi_recon = inverse_klt(hsi_k_recon, klt_matrix, mean_vec)

    print("Original HSI shape:", hsi_example.shape)
    print("KLT transformed shape:", hsi_k.shape)
    print("Reconstructed HSI shape:", hsi_recon.shape)
