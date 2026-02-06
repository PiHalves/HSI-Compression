import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf


class StructualSimilarity(tf.keras.layers.Layer):
    def __init__(self, data_range=1.0, **kwargs):
        super(StructualSimilarity, self).__init__()
        self.data_range = data_range

    def call(self, y_true, y_pred):
        if len(y_true.shape) == 5:
            y_true = tf.squeeze(y_true, axis=3)  # (B, H, W, C)
            y_pred = tf.squeeze(y_pred, axis=3)

        ssim_per_channel = tf.image.ssim(
            y_true,
            y_pred,
            max_val=self.data_range
        )

        return tf.reduce_mean(ssim_per_channel)


class structuralSimilarityLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(structuralSimilarityLoss, self).__init__()
        self.metric = StructualSimilarity()

    def call(self, y_true, y_pred):
        return 1.0 - self.metric(y_true, y_pred)


def hsi_to_rgb(img, rgb_bands=(43, 28, 10)):
    if img.ndim == 4:
        img = img[0]
    bands = [min(b, img.shape[-1]-1) for b in rgb_bands]
    rgb = img[..., bands]
    rgb_min = rgb.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb.max(axis=(0, 1), keepdims=True)
    return (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)


def plot_reconstruction_error_histogram(
    model,
    dataset,
    max_samples=None,
    bins=50,
    clip_percentile=99.9,
    error_type=None,
    title="Reconstruction Error per Image (1-SSIM)",
    show=True,
    save_path=None,
    visualize_examples=True,
    rgb_bands=(43, 28, 10),
    data_range=1.0
):

    # ===== OUTPUT DIRS =====
    if save_path:
        base_dir = os.path.dirname(save_path)
        images_dir = os.path.join(base_dir, "images")
        plots_dir = os.path.join(base_dir, "plots")
    else:
        images_dir = "outputs/images"
        plots_dir = "outputs/plots"

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    image_errors = []
    stored_samples = []

    ssim_metric = StructualSimilarity(data_range=data_range)

    # ===== PROCESS DATA =====
    for i, x in enumerate(dataset):
        print(f"Processing sample {i+1}/{len(dataset)}...", end='\r')

        if max_samples is not None and i >= max_samples:
            break

        rec = model(x, training=False)
        x_hat = rec[0] if isinstance(rec, tuple) else rec

        print("x shape:", x.shape, "x_hat shape:", x_hat.shape)
        
        if len(x.shape) == 5:
            x = tf.squeeze(x, axis=3)
        if len(x_hat.shape) == 5:
            x_hat = tf.squeeze(x_hat, axis=3)
        
        x_true_f = tf.cast(x, tf.float32)
        x_hat_f = tf.cast(x_hat, tf.float32)

        # Handle 5D tensors (B, H, W, 1, C) -> squeeze to (B, H, W, C)
        

        img = x_true_f[0] if x_true_f.ndim == 4 else x_true_f
        img_hat = x_hat_f[0] if x_hat_f.ndim == 4 else x_hat_f

        ssim_val = ssim_metric(img, img_hat).numpy()
        error = 1.0 - ssim_val

        image_errors.append(error)
        stored_samples.append(
            (x.numpy(), x_hat.numpy(), x_hat.numpy() - x.numpy())
        )

    if not image_errors:
        print("No data processed.")
        return None, None, None

    image_errors = np.array(image_errors)

    # ===== GLOBAL HISTOGRAM (TXT) =====
    counts, bin_edges = np.histogram(image_errors, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    np.savetxt(
        os.path.join(plots_dir, "histogram_1ssim.txt"),
        np.column_stack((bin_centers, counts)),
        header="Bin_center\tDensity",
        fmt="%.6e\t%.6e",
        delimiter="\t",
        comments=""
    )

    # ===== FIGURE (OPTIONAL) =====
    plt.figure(figsize=(10, 6))
    plt.hist(image_errors, bins=bins, density=True,
             alpha=0.7, edgecolor="black")
    plt.xlabel("1 - SSIM (per image)")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        histogram_save_path = os.path.join(plots_dir, "histogram_1ssim.png")
        plt.savefig(histogram_save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    # ===== BEST / MEDIAN / WORST =====
    if visualize_examples:
        sorted_idx = np.argsort(image_errors)
        indices = {
            "best": sorted_idx[0],
            "median": sorted_idx[len(sorted_idx)//2],
            "worst": sorted_idx[-1]
        }

        for label, idx in indices.items():
            x, x_hat, err = stored_samples[idx]

            # ===== RGB IMAGES =====
            x_rgb = hsi_to_rgb(x, rgb_bands=rgb_bands)
            xhat_rgb = hsi_to_rgb(x_hat, rgb_bands=rgb_bands)

            plt.imsave(os.path.join(
                images_dir, f"{label}_original_{idx}.png"), x_rgb)
            plt.imsave(
                os.path.join(images_dir, f"{label}_reconstructed_{idx}.png"), xhat_rgb)

            # ===== ERROR MAP PNG =====
            if err.ndim == 4:
                # Remove abs() to preserve sign
                err_map = np.mean(err[0], axis=-1)
            else:
                # Remove abs() to preserve sign
                err_map = np.mean(err, axis=-1)

            # For diverging colormap, center the normalization around 0
            err_abs_max = np.max(np.abs(err_map))
            err_min, err_max = err_map.min(), err_map.max()

            # Normalize symmetrically around zero for better color mapping
            err_map_norm = err_map / (err_abs_max + 1e-8)

            # Create custom colormap with black background (blue-black-red)
            colors = ['blue', 'black', 'red']
            n_bins = 256
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                'custom_rbk', colors, N=n_bins)

            plt.imsave(
                os.path.join(
                    images_dir, f"{label}_error_map_{idx}_min_{err_min:.6e}_max_{err_max:.6e}.png"),
                err_map_norm,
                cmap=custom_cmap,  # Custom blue-black-red colormap
                vmin=-1, vmax=1  # Ensure symmetric color range
            )

            # ===== PIXEL ERROR HISTOGRAM (TXT) =====
            pixel_err = err.reshape(-1)
            if clip_percentile is not None:
                p = np.percentile(np.abs(pixel_err), clip_percentile)
                pixel_err = np.clip(pixel_err, -p, p)

            hist_pe, bins_pe = np.histogram(pixel_err, bins=100, density=True)
            centers_pe = (bins_pe[:-1] + bins_pe[1:]) / 2

            np.savetxt(
                os.path.join(plots_dir, f"pixel_error_hist_{label}_{idx}.txt"),
                np.column_stack((centers_pe, hist_pe)),
                header="Pixel_error\tDensity",
                fmt="%.6e\t%.6e",
                delimiter="\t",
                comments=""
            )

            # ===== SPECTRAL COVERAGE (TXT) =====
            n_bands = x.shape[-1]
            avg_orig = [np.mean(x[..., b]) for b in range(n_bands)]
            avg_rec = [np.mean(x_hat[..., b]) for b in range(n_bands)]

            np.savetxt(
                os.path.join(plots_dir, f"spectral_{label}_{idx}.txt"),
                np.column_stack((range(n_bands), avg_orig, avg_rec)),
                header="Band\tOriginal\tReconstructed",
                fmt="%d\t%.6e\t%.6e",
                delimiter="\t",
                comments=""
            )

    return image_errors, bin_edges, counts


# def plot_reconstruction_error_histogram(
#     model,
#     dataset,
#     max_samples=None,
#     bins=200,
#     clip_percentile=99.9,
#     error_type="signed",
#     title="Reconstruction Error Histogram",
#     show=True,
#     save_path=None,
#     save_hist_data_path=None
# ):
#     """
#     Plot reconstruction error histogram and save data for LaTeX.
#     """

#     all_errors = []

#     for i, x in enumerate(dataset):
#         if max_samples is not None and i >= max_samples:
#             break

#         rec = model(x, training=False)
#         if isinstance(rec, (tuple)):
#             x_hat = rec[0]
#         else:
#             x_hat = rec
#         error = x_hat - x

#         if error_type == "absolute":
#             error = tf.abs(error)
#         elif error_type == "squared":
#             error = tf.square(error)

#         all_errors.append(error.numpy().reshape(-1))

#     if not all_errors:
#         print("No data processed.")
#         return None, None, None

#     all_errors = np.concatenate(all_errors, axis=0)

#     if len(all_errors) == 0:
#         print("Error: No errors computed.")
#         return None, None, None

#     # Clipping
#     if clip_percentile is not None and len(all_errors) > 0:
#         p = np.percentile(np.abs(all_errors), clip_percentile)
#         all_errors = np.clip(all_errors, -p, p)

#     # Compute histogram
#     counts, bin_edges = np.histogram(all_errors, bins=bins, density=True)

#     # ============ KLUCZOWA CZĘŚĆ: ZAPIS DLA LaTeX ============
#     if save_hist_data_path:
#         # Upewnij się, że ścieżka ma odpowiednie rozszerzenie
#         if not save_hist_data_path.endswith('.txt') and not save_hist_data_path.endswith('.dat'):
#             save_hist_data_path += '.dat'

#         # Oblicz środki binów
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#         # Zapisz dane w formacie przyjaznym dla LaTeX/PGFPlots
#         hist_data = np.column_stack((bin_centers, counts))

#         # Nagłówek z informacjami o danych
#         header = (f"# Histogram data for LaTeX/PGFPlots\n"
#                  f"# Generated from: {title}\n"
#                  f"# Total samples: {len(all_errors)}\n"
#                  f"# Mean: {np.mean(all_errors):.6e}, Std: {np.std(all_errors):.6e}\n"
#                  f"# Bin_center Density")

#         np.savetxt(save_hist_data_path, hist_data,
#                   fmt="%.6e %.6e",  # dwa formaty dla dwóch kolumn
#                   delimiter='\t',    # tabulator lepszy dla LaTeX
#                   header=header,
#                   comments='')

#         print(f"Histogram data saved to: {save_hist_data_path}")
#         print(f"  Bins: {bins}, Samples: {len(all_errors)}")

#     # ============ WIZUALIZACJA ============
#     plt.figure(figsize=(10, 6))
#     plt.hist(all_errors, bins=bins, density=True, alpha=0.7, edgecolor='black')
#     plt.xlabel('Reconstruction Error')
#     plt.ylabel('Density')
#     plt.title(title)
#     plt.grid(True, alpha=0.3)

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")

#     if show:
#         plt.show()
#     else:
#         plt.close()

#     return all_errors, bin_edges, counts
