import tensorflow as tf
import csv
import numpy as np
from PIL import Image


class TFHySpecNetLoader:
    def __init__(self, root_dir, mode="easy", split="train", transform=None,
                 num_parallel_calls=32, prefetch_buffer=32, data_mode=1, batch_size=12):
        self.root_dir = root_dir
        self.transform = transform
        self.num_parallel_calls = num_parallel_calls
        self.data_mode = data_mode
        self.batch_size = batch_size
        self.split = split
        print(self.data_mode)

        # Load file paths
        import csv
        import os
        csv_path = os.path.join(self.root_dir, "splits", mode, f"{split}.csv")
        with open(csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            npy_paths = sum(csv_data, [])
        self.npy_paths = [os.path.join(
            self.root_dir, "patches", x) for x in npy_paths]
        self.dataset = self._create_dataset(prefetch_buffer)

    def _load_and_preprocess(self, file_path):
        if self.data_mode == 3:
            # Load both NPY (hyperspectral) and TIF (segmentation mask) for U-Net
            def load_image_and_mask(path):
                path_str = path.decode('utf-8')
                # Load hyperspectral data
                img = np.load(path_str)
                img = np.transpose(img, [1, 2, 0])  # [C,H,W] -> [H,W,C]
                # [H,W,C] -> [H,W,1,C] for 3D U-Net
                img = np.expand_dims(img, axis=2)
                # Swap to get [H,W,C,1] for proper 3D conv input
                img = np.transpose(img, [0, 1, 3, 2])  # [H,W,1,C] -> [H,W,C,1]

                # Load segmentation mask
                tif_path = path_str.replace(
                    '-DATA.npy', '-QL_QUALITY_CLASSES.TIF')
                mask = np.array(Image.open(tif_path)).astype(np.int32)
                return img.astype(np.float32), mask

            img, mask = tf.numpy_function(
                load_image_and_mask,
                [file_path],
                [tf.float32, tf.int32]
            )
            img.set_shape([128, 128, 202, 1])
            mask.set_shape([128, 128])
            return img, mask
        else:
            # Original NPY loading
            img = tf.numpy_function(
                lambda x: np.load(x.decode('utf-8')),
                [file_path],
                tf.float32
            )

        if self.data_mode == 1:
            img = tf.transpose(img, [1, 2, 0])
            img = tf.expand_dims(img, axis=2)
        elif self.data_mode == 2:
            img = tf.transpose(img, [1, 2, 0])

        if self.transform:
            img = self.transform(img)

        return img

    def _create_dataset(self, prefetch_buffer):
        dataset = tf.data.Dataset.from_tensor_slices(self.npy_paths)
        dataset = dataset.map(
            self._load_and_preprocess,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=False
        )
        if self.split == "train":
            dataset = dataset.shuffle(
                buffer_size=1000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        print(len(dataset))
        dataset = dataset.prefetch(buffer_size=prefetch_buffer)

        return dataset
