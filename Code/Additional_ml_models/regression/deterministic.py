import pandas as pd
import numpy as np

def get_white_pixel_percentage(mask, col_position):
    """
    Computes the percentage of white pixels in a vertical strip of the binary mask.

    Parameters:
        mask_path (str): Path to the mask image.
        col_position (float): Normalized position (0 to 1) of the strip within the cropped mask.
        threshold (int): Threshold to binarize the mask (default: 128).

    Returns:
        float: White pixel percentage.
    """
    # Binarize the mask (convert to 0 and 1)
    binary_mask = mask.squeeze().cpu().numpy()  # Shape (3, 224, 224)
    binary_mask = binary_mask[0]
    binary_mask = (binary_mask > 0).astype(np.uint8)

    h, w = binary_mask.shape

    non_zero_coords = np.where(binary_mask > 0)[1]
    x_min = np.min(non_zero_coords)
    x_max = np.max(non_zero_coords)

    h, w = binary_mask.shape

    trim_check_threshold = int(0.25 * w)

    default_trim_threshold = int(0.08 * w)

    trim_left = x_min < trim_check_threshold
    trim_right = x_max > (w - trim_check_threshold)

    if trim_left and trim_right:
        left_crop = x_min
        right_crop = x_max
    elif trim_left and not trim_right:
        left_crop = x_min
        right_crop = w - x_min
    elif trim_right and not trim_left:
        right_crop = x_max
        left_crop = w - x_max
    else:
        left_crop = default_trim_threshold
        right_crop = w - default_trim_threshold

    cropped_mask = binary_mask[:, left_crop:right_crop]

    # Compute stripe position in the cropped mask
    new_w = cropped_mask.shape[1]
    col_position = np.clip(int(col_position * new_w), 0, new_w - 1)

    # Extract vertical stripe
    stripe = cropped_mask[:, col_position]
    # Calculate white pixel percentage
    return np.sum(stripe)


def get_scaled_jsw_results_df(
    test_dataset,
    medial_columns,
    lateral_columns,
    label,
):
    """
    JSW estimation using physical scaling:
    white_pixel_percentage * IMPIXSZ / 224 * ORIG_H
    """

    results = {"Sample_ID": []}
    medial_gt, medial_pred = [], []
    lateral_gt, lateral_pred = [], []

    for col in medial_columns + lateral_columns:
        results[f"{col}_Ground_Truth"] = []
        results[f"{col}_Prediction"] = []

    for i in range(len(test_dataset)):
        data = test_dataset[i]
        img = data["image"].unsqueeze(0)

        results["Sample_ID"].append(i)

        for idx, col in enumerate(medial_columns):
            pos = int(col[-3:]) / 1000
            if label == "Right":
                pos = 1 - pos

            wp = get_white_pixel_percentage(img, pos)
            jsw = wp * data["IMPIXSZ"] / img.shape[2] * data["ORIG_H"]

            gt = data["medial_target"][idx]
            results[f"{col}_Ground_Truth"].append(gt)
            results[f"{col}_Prediction"].append(jsw)

            if not np.isnan(gt):
                medial_gt.append(gt)
                medial_pred.append(jsw)

        for idx, col in enumerate(lateral_columns):
            pos = int(col[-3:]) / 1000
            if label == "Right":
                pos = 1 - pos

            wp = get_white_pixel_percentage(img, pos)
            jsw = wp * data["IMPIXSZ"] / img.shape[2] * data["ORIG_H"]

            gt = data["lateral_target"][idx]
            results[f"{col}_Ground_Truth"].append(gt)
            results[f"{col}_Prediction"].append(jsw)

            if not np.isnan(gt):
                lateral_gt.append(gt)
                lateral_pred.append(jsw)

    return pd.DataFrame(results)

def get_deterministic_jsw_results_df(
    test_dataset,
    medial_columns,
    lateral_columns,
    label,
    coef=77.88208682895399,
    intercept=1.3779068446995613,
):
    results = {"Sample_ID": []}
    medial_gt, medial_pred = [], []
    lateral_gt, lateral_pred = [], []

    for col in medial_columns + lateral_columns:
        results[f"{col}_Ground_Truth"] = []
        results[f"{col}_Prediction"] = []

    for i in range(len(test_dataset)):
        data = test_dataset[i]
        img = data["image"].unsqueeze(0)

        results["Sample_ID"].append(i)

        for idx, col in enumerate(medial_columns):
            pos = int(col[-3:]) / 1000
            if label == "Right":
                pos = 1 - pos

            wp = get_white_pixel_percentage(img, pos)
            jsw = wp / img.shape[2] * coef + intercept

            gt = data["medial_target"][idx]
            results[f"{col}_Ground_Truth"].append(gt)
            results[f"{col}_Prediction"].append(jsw)

            if not np.isnan(gt):
                medial_gt.append(gt)
                medial_pred.append(jsw)

        for idx, col in enumerate(lateral_columns):
            pos = int(col[-3:]) / 1000
            if label == "Right":
                pos = 1 - pos

            wp = get_white_pixel_percentage(img, pos)
            jsw = wp / img.shape[2] * coef + intercept

            gt = data["lateral_target"][idx]
            results[f"{col}_Ground_Truth"].append(gt)
            results[f"{col}_Prediction"].append(jsw)

            if not np.isnan(gt):
                lateral_gt.append(gt)
                lateral_pred.append(jsw)

    return pd.DataFrame(results)