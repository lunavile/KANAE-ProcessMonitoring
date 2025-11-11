import numpy as np

def compute_detection_time(values, threshold, time_step=3):
    """
    Computes the first time the values cross the detection threshold.
    Returns the time (in minutes) or None if not detected.
    """
    for i, v in enumerate(values):
        if v > threshold:
            return i * time_step
    return None

def compute_metrics(values, threshold, fault_start=480, fault_end=2880, time_step=3):
    """
    Computes performance metrics based on a given threshold and SPE profile.

    Returns: FNR, FPR, Precision, Recall, F1, FDR
    """
    values = np.array(values)
    times = np.arange(len(values)) * time_step
    detected = values > threshold

    # Time windows
    fault_mask = (times >= fault_start) & (times <= fault_end)
    normal_mask = (times < fault_start)

    TP = np.sum(detected & fault_mask)
    FN = np.sum(~detected & fault_mask)
    FP = np.sum(detected & normal_mask)
    TN = np.sum(~detected & normal_mask)

    # Avoid div-by-zero
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    FDR = TP / (TP + FN) if (TP + FN) > 0 else 0

    return {
        "FNR": FNR,
        "FPR": FPR,
        "Precision": Precision,
        "Recall": Recall,
        "F1": F1,
        "FDR": FDR
    }
