import numpy as np
import tensorflow as tf


class CompressionMetrics:
    @staticmethod
    def mse(a, b):
        return np.mean((a - b) ** 2)

    @staticmethod
    def rmse(a, b):
        return np.sqrt(CompressionMetrics.mse(a, b))

    @staticmethod
    def psnr(a, b, max_val=1.0):
        if np.array_equal(a, b):
            return float('inf')
        mse = CompressionMetrics.mse(a, b)
        return 20 * np.log10(max_val / np.sqrt(mse + 1e-12))

    @staticmethod
    def snr(signal, noise):
        if np.array_equal(signal, signal + noise):
            return float('inf')
        return 10 * np.log10(
            np.sum(signal ** 2) / (np.sum(noise ** 2))
        )

    @staticmethod
    def sam(a, b):
        """
        Spectral Angle Mapper: a and b are (..., bands)
        """
        dot = np.sum(a * b, axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cos = np.clip(dot / (na * nb + 1e-12), -1, 1)
        return np.mean(np.arccos(cos))

    @staticmethod
    def compression_ratio(original_bits, compressed_bits):
        return original_bits / max(compressed_bits, 1)

    @staticmethod
    def bpp(compressed_bits, H, W):
        return compressed_bits / (H * W)

    @staticmethod
    def bpppb(compressed_bits, H, W, B):
        return compressed_bits / (H * W * B)

    @staticmethod
    def ssim(a, b, data_range=1.0):
        a = tf.convert_to_tensor(a)
        b = tf.convert_to_tensor(b)
        ssim_per_channel = tf.image.ssim(
            a,
            b,
            max_val=data_range
        )
        return tf.reduce_mean(ssim_per_channel)

    @staticmethod
    def mae(a, b):
        return np.mean(np.abs(a - b))

    @staticmethod
    def maximum_absolute_error(a, b):
        max_error = np.max(np.abs(a - b))
        # Find the first occurrence of the maximum absolute error
        max_error_index = np.unravel_index(np.argmax(np.abs(a - b)), a.shape)
        return max_error, max_error_index

    @staticmethod
    def error_histogram(a, b, bins=50):
        errors = np.abs(a - b).flatten()
        histogram, bin_edges = np.histogram(errors, bins=bins)
        return histogram, bin_edges


class MeanSquaredError(tf.keras.layers.Layer):
    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))


class PeakSignalToNoiseRatio(tf.keras.layers.Layer):
    def __init__(self, max_val=1.0):
        super(PeakSignalToNoiseRatio, self).__init__()
        self.max_val = max_val

    def call(self, y_true, y_pred):
        mse = MeanSquaredError()(y_true, y_pred)
        mse = tf.maximum(mse, 1e-15)
        return 20 * tf.math.log(self.max_val) / tf.math.log(10.0) \
            - 10 * tf.math.log(mse) / tf.math.log(10.0)


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


class SpectralAngle(tf.keras.layers.Layer):

    """
        dot = np.sum(a * b, axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cos = np.clip(dot / (na * nb + 1e-12), -1, 1)
        return np.mean(np.arccos(cos))
    """

    def __init__(self, epsilon=1e-12):
        super(SpectralAngle, self).__init__()
        self.epsilon = epsilon

    def call(self, y_true, y_pred):

        dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
        norm_true = tf.norm(y_true, axis=-1)
        norm_pred = tf.norm(y_pred, axis=-1)
        cos = tf.clip_by_value(
            dot_product / (norm_true * norm_pred + self.epsilon), -1.0, 1.0)
        sa_rad = tf.acos(cos)
        sa_deg = sa_rad * (180.0 / np.pi)

        return tf.reduce_mean(sa_deg)


class MeanSquaredErrorLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(MeanSquaredErrorLoss, self).__init__()
        self.metric = MeanSquaredError()

    def call(self, y_true, y_pred):
        return self.metric(y_true, y_pred)


class SpectralAngleLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(SpectralAngleLoss, self).__init__()
        self.metric = SpectralAngle()

    def call(self, y_true, y_pred):
        return self.metric(y_true, y_pred)


class structuralSimilarityLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(structuralSimilarityLoss, self).__init__()
        self.metric = StructualSimilarity()

    def call(self, y_true, y_pred):
        return 1.0 - self.metric(y_true, y_pred)


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, mse_weight=0.5, ssim_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_loss = MeanSquaredErrorLoss()
        self.ssim_loss = structuralSimilarityLoss()

    def call(self, y_true, y_pred):
        mse = self.mse_loss(y_true, y_pred)
        ssim = self.ssim_loss(y_true, y_pred)
        ssim_weighted = ssim * self.ssim_weight
        mse_weighted = mse * self.mse_weight
        loss = mse_weighted + ssim_weighted

        return loss


class SegmentationMetrics:
    """Class for segmentation metrics calculation."""

    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes=None):
        """
        Compute confusion matrix.

        Args:
            y_true: Ground truth labels (flattened or 2D array)
            y_pred: Predicted labels (flattened or 2D array)
            num_classes: Number of classes. If None, inferred from data.

        Returns:
            Confusion matrix of shape (num_classes, num_classes)
            where rows are true classes and columns are predicted classes.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if num_classes is None:
            num_classes = max(np.max(y_true), np.max(y_pred)) + 1

        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    @staticmethod
    def _get_tp_tn_fp_fn(y_true, y_pred, class_idx=None):
        """
        Calculate TP, TN, FP, FN for binary or multiclass classification.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_idx: If provided, calculate for specific class (one-vs-rest)

        Returns:
            Tuple of (TP, TN, FP, FN)
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if class_idx is not None:
            # One-vs-rest for multiclass
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
        else:
            y_true_binary = y_true
            y_pred_binary = y_pred

        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        return tp, tn, fp, fn

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculate overall accuracy.

        Accuracy = (TP + TN) / (TP + TN + FP + FN)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Accuracy score between 0 and 1
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        return np.mean(y_true == y_pred)

    @staticmethod
    def ppv(y_true, y_pred, class_idx=None, average='macro'):
        """
        Calculate Positive Predictive Value (Precision).

        PPV = TP / (TP + FP)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_idx: Specific class index for binary calculation
            average: 'macro', 'micro', or 'weighted' for multiclass

        Returns:
            PPV score
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if class_idx is not None:
            tp, _, fp, _ = SegmentationMetrics._get_tp_tn_fp_fn(
                y_true, y_pred, class_idx)
            return tp / (tp + fp + 1e-12)

        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

        if average == 'micro':
            tp_total = fp_total = 0
            for c in range(num_classes):
                tp, _, fp, _ = SegmentationMetrics._get_tp_tn_fp_fn(
                    y_true, y_pred, c)
                tp_total += tp
                fp_total += fp
            return tp_total / (tp_total + fp_total + 1e-12)

        ppv_per_class = []
        class_counts = []
        for c in range(num_classes):
            tp, _, fp, _ = SegmentationMetrics._get_tp_tn_fp_fn(
                y_true, y_pred, c)
            ppv_per_class.append(tp / (tp + fp + 1e-12))
            class_counts.append(np.sum(y_true == c))

        if average == 'macro':
            return np.mean(ppv_per_class)
        elif average == 'weighted':
            weights = np.array(class_counts) / np.sum(class_counts)
            return np.sum(np.array(ppv_per_class) * weights)

        return np.array(ppv_per_class)

    @staticmethod
    def recall(y_true, y_pred, class_idx=None, average='macro'):
        """
        Calculate Recall (Sensitivity, True Positive Rate).

        Recall = TP / (TP + FN)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_idx: Specific class index for binary calculation
            average: 'macro', 'micro', or 'weighted' for multiclass

        Returns:
            Recall score
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if class_idx is not None:
            tp, _, _, fn = SegmentationMetrics._get_tp_tn_fp_fn(
                y_true, y_pred, class_idx)
            return tp / (tp + fn + 1e-12)

        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

        if average == 'micro':
            tp_total = fn_total = 0
            for c in range(num_classes):
                tp, _, _, fn = SegmentationMetrics._get_tp_tn_fp_fn(
                    y_true, y_pred, c)
                tp_total += tp
                fn_total += fn
            return tp_total / (tp_total + fn_total + 1e-12)

        recall_per_class = []
        class_counts = []
        for c in range(num_classes):
            tp, _, _, fn = SegmentationMetrics._get_tp_tn_fp_fn(
                y_true, y_pred, c)
            recall_per_class.append(tp / (tp + fn + 1e-12))
            class_counts.append(np.sum(y_true == c))

        if average == 'macro':
            return np.mean(recall_per_class)
        elif average == 'weighted':
            weights = np.array(class_counts) / np.sum(class_counts)
            return np.sum(np.array(recall_per_class) * weights)

        return np.array(recall_per_class)

    @staticmethod
    def f1_score(y_true, y_pred, class_idx=None, average='macro'):
        """
        Calculate F1-score (harmonic mean of PPV and Recall).

        F1 = 2 * (PPV * Recall) / (PPV + Recall)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_idx: Specific class index for binary calculation
            average: 'macro', 'micro', or 'weighted' for multiclass

        Returns:
            F1-score
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if class_idx is not None:
            p = SegmentationMetrics.ppv(y_true, y_pred, class_idx)
            r = SegmentationMetrics.recall(y_true, y_pred, class_idx)
            return 2 * (p * r) / (p + r + 1e-12)

        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

        if average == 'micro':
            p = SegmentationMetrics.ppv(y_true, y_pred, average='micro')
            r = SegmentationMetrics.recall(y_true, y_pred, average='micro')
            return 2 * (p * r) / (p + r + 1e-12)

        f1_per_class = []
        class_counts = []
        for c in range(num_classes):
            p = SegmentationMetrics.ppv(y_true, y_pred, c)
            r = SegmentationMetrics.recall(y_true, y_pred, c)
            f1_per_class.append(2 * (p * r) / (p + r + 1e-12))
            class_counts.append(np.sum(y_true == c))

        if average == 'macro':
            return np.mean(f1_per_class)
        elif average == 'weighted':
            weights = np.array(class_counts) / np.sum(class_counts)
            return np.sum(np.array(f1_per_class) * weights)

        return np.array(f1_per_class)

    @staticmethod
    def iou(y_true, y_pred, class_idx=None, average='macro'):
        """
        Calculate Intersection over Union (Jaccard Index).

        IoU = TP / (TP + FP + FN)

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_idx: Specific class index for binary calculation
            average: 'macro', 'micro', or 'weighted' for multiclass

        Returns:
            IoU score
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if class_idx is not None:
            tp, _, fp, fn = SegmentationMetrics._get_tp_tn_fp_fn(
                y_true, y_pred, class_idx)
            return tp / (tp + fp + fn + 1e-12)

        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

        if average == 'micro':
            tp_total = fp_total = fn_total = 0
            for c in range(num_classes):
                tp, _, fp, fn = SegmentationMetrics._get_tp_tn_fp_fn(
                    y_true, y_pred, c)
                tp_total += tp
                fp_total += fp
                fn_total += fn
            return tp_total / (tp_total + fp_total + fn_total + 1e-12)

        iou_per_class = []
        class_counts = []
        for c in range(num_classes):
            tp, _, fp, fn = SegmentationMetrics._get_tp_tn_fp_fn(
                y_true, y_pred, c)
            iou_per_class.append(tp / (tp + fp + fn + 1e-12))
            class_counts.append(np.sum(y_true == c))

        if average == 'macro':
            return np.mean(iou_per_class)
        elif average == 'weighted':
            weights = np.array(class_counts) / np.sum(class_counts)
            return np.sum(np.array(iou_per_class) * weights)

        return np.array(iou_per_class)

    @staticmethod
    def mean_iou(y_true, y_pred, num_classes=None):
        """
        Calculate Mean Intersection over Union (mIoU).

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            num_classes: Number of classes

        Returns:
            Mean IoU across all classes
        """
        return SegmentationMetrics.iou(y_true, y_pred, average='macro')

    @staticmethod
    def auc_roc(y_true, y_scores, num_classes=None):
        """
        Calculate Area Under the ROC Curve.

        AUC = integral of TPR over FPR

        Uses trapezoidal rule for numerical integration.

        Args:
            y_true: Ground truth labels (integers for multiclass)
            y_scores: Prediction scores/probabilities 
                      Shape: (n_samples,) for binary or 
                             (n_samples, n_classes) for multiclass
            num_classes: Number of classes (inferred if None)

        Returns:
            AUC score (macro-averaged for multiclass)
        """
        y_true = np.asarray(y_true).flatten()
        y_scores = np.asarray(y_scores)

        # Binary classification
        if y_scores.ndim == 1 or y_scores.shape[1] == 1:
            if y_scores.ndim == 2:
                y_scores = y_scores.flatten()
            return SegmentationMetrics._binary_auc(y_true, y_scores)

        # Multiclass - compute macro-averaged AUC
        if num_classes is None:
            num_classes = y_scores.shape[1]

        auc_per_class = []
        for c in range(num_classes):
            y_true_binary = (y_true == c).astype(int)
            y_scores_c = y_scores[:, c]
            auc_c = SegmentationMetrics._binary_auc(y_true_binary, y_scores_c)
            auc_per_class.append(auc_c)

        return np.mean(auc_per_class)

    @staticmethod
    def _binary_auc(y_true, y_scores):
        """
        Calculate binary AUC using trapezoidal rule.

        Args:
            y_true: Binary ground truth labels (0 or 1)
            y_scores: Prediction scores

        Returns:
            AUC score
        """
        # Sort by scores descending
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        y_scores_sorted = y_scores[desc_score_indices]

        # Get unique thresholds
        distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
        threshold_idxs = np.concatenate(
            ([0], distinct_value_indices + 1, [len(y_scores_sorted)]))

        # Calculate TPR and FPR at each threshold
        tps = np.cumsum(y_true_sorted)[threshold_idxs[:-1]]
        fps = np.cumsum(1 - y_true_sorted)[threshold_idxs[:-1]]

        total_positives = np.sum(y_true)
        total_negatives = len(y_true) - total_positives

        if total_positives == 0 or total_negatives == 0:
            return 0.5  # Undefined, return random classifier performance

        tpr = tps / total_positives
        fpr = fps / total_negatives

        # Add (0, 0) and (1, 1) endpoints
        tpr = np.concatenate([[0], tpr, [1]])
        fpr = np.concatenate([[0], fpr, [1]])

        # Trapezoidal integration
        auc = np.trapezoid(tpr, fpr)

        return auc

    @staticmethod
    def dice_coefficient(y_true, y_pred, class_idx=None, average='macro'):
        """
        Calculate Dice Coefficient (F1-score equivalent for segmentation).

        Dice = 2 * TP / (2 * TP + FP + FN)

        This is equivalent to F1-score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_idx: Specific class index
            average: 'macro', 'micro', or 'weighted'

        Returns:
            Dice coefficient
        """
        return SegmentationMetrics.f1_score(y_true, y_pred, class_idx, average)

    @staticmethod
    def per_class_metrics(y_true, y_pred, num_classes=None):
        """
        Calculate all metrics per class.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            num_classes: Number of classes

        Returns:
            Dictionary with per-class metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if num_classes is None:
            num_classes = max(np.max(y_true), np.max(y_pred)) + 1

        results = {
            'ppv': [],
            'recall': [],
            'f1_score': [],
            'iou': [],
            'support': []
        }

        for c in range(num_classes):
            results['ppv'].append(
                SegmentationMetrics.ppv(y_true, y_pred, c))
            results['recall'].append(
                SegmentationMetrics.recall(y_true, y_pred, c))
            results['f1_score'].append(
                SegmentationMetrics.f1_score(y_true, y_pred, c))
            results['iou'].append(
                SegmentationMetrics.iou(y_true, y_pred, c))
            results['support'].append(np.sum(y_true == c))

        return results
