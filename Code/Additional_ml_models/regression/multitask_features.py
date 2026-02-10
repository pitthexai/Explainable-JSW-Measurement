import torch
import numpy as np
from deterministic import get_white_pixel_percentage

MEDIAL_COLUMNS = [
    "V00JSW150", "V00JSW175", "V00JSW200",
    "V00JSW225", "V00JSW250", "V00JSW275", "V00JSW300"
]

LATERAL_COLUMNS = [
    "V00LJSW700", "V00LJSW725", "V00LJSW750",
    "V00LJSW775", "V00LJSW800", "V00LJSW825",
    "V00LJSW850", "V00LJSW875", "V00LJSW900"
]

ALL_COLUMNS = MEDIAL_COLUMNS + LATERAL_COLUMNS


def extract_multitask_features(image, side_label):
    """
    image: Tensor [1, H, W]
    side_label: "Left" or "Right"
    """
    features = []

    for col in ALL_COLUMNS:
        pos = int(col[-3:]) / 1000
        if side_label == "Right":
            pos = 1 - pos

        wp = get_white_pixel_percentage(image, pos)
        features.append(wp)

    # SIDE feature
    side_feat = 1 if side_label == "Right" else 0
    features.append(side_feat)

    return np.array(features, dtype=np.float32)
