import math
import numpy as np
from scipy.ndimage import label

import torch

from lib.data import LABELS

SCALE_FACTORS = {
    'lv_cav': {
        'slope': 0.14851364940541464,
        'intercept': -1.5047371493498822
    },
    'rv_cav': {
        'slope': 0.12255735679080532,
        'intercept': 0.7310369657588982
    },
    'lv_wall': {
        'slope': 0.13741395713194718,
        'intercept': -0.03419160910565466
    },
    'aorta': {
        'slope': 0.29564912866813053,
        'intercept': 1.3448943584231348
    }
}


def rescale_scaled_volume_to_ml(volumes):
    lv_cav = volumes['lv_cav'] / SCALE_FACTORS['lv_cav']['slope'] - SCALE_FACTORS['lv_cav']['intercept']
    rv_cav = volumes['rv_cav'] / SCALE_FACTORS['rv_cav']['slope'] - SCALE_FACTORS['rv_cav']['intercept']
    lv_wall = volumes['lv_wall'] / SCALE_FACTORS['lv_wall']['slope'] - SCALE_FACTORS['lv_wall']['intercept']
    return lv_cav, rv_cav, lv_wall


def get_aortic_diameter_from_volumes_pytorch(volume):
    try:
        with torch.no_grad():
            stack_out = torch.argmax(volume[0], dim=0).cpu().numpy()
        pa_boolean_stack = stack_out == LABELS['pa']
        pa_z = np.any(pa_boolean_stack, axis=(1, 2))
        mid_pa_z = int(np.median(np.argwhere(pa_z)))
        aorta_boolean_slice = stack_out[mid_pa_z] == LABELS['aorta']
        # https://stackoverflow.com/questions/9440921/identify-contiguous-regions-in-2d-numpy-array
        labels, numL = label(aorta_boolean_slice)
        aorta_indices = [(labels == i).nonzero() for i in range(1, numL + 1)]
        if numL == 2:
            aorta_pred = math.sqrt(len(aorta_indices[0][0]))
            aorta_pred = aorta_pred / SCALE_FACTORS['aorta']['slope'] - SCALE_FACTORS['aorta']['intercept']
            return aorta_pred
        elif numL < 1:
            return False
        else:
            return False
    except ValueError as e:
        print(f"Failed to get aortic diameter: {e}")
        return False

def get_aortic_diameter_from_volumes_numpy(volume):
    try:
        stack_out = np.argmax(volume[0], axis=0)
        pa_boolean_stack = stack_out == LABELS['pa']
        pa_z = np.any(pa_boolean_stack, axis=(1, 2))
        mid_pa_z = int(np.median(np.argwhere(pa_z)))
        aorta_boolean_slice = stack_out[mid_pa_z] == LABELS['aorta']
        # https://stackoverflow.com/questions/9440921/identify-contiguous-regions-in-2d-numpy-array
        labels, numL = label(aorta_boolean_slice)
        aorta_indices = [(labels == i).nonzero() for i in range(1, numL + 1)]
        if numL == 2:
            aorta_pred = math.sqrt(len(aorta_indices[0][0]))
            aorta_pred = aorta_pred / SCALE_FACTORS['aorta']['slope'] - SCALE_FACTORS['aorta']['intercept']
            return aorta_pred
        elif numL < 1:
            return False
        else:
            return False
    except ValueError as e:
        print(f"Failed to get aortic diameter: {e}")
        return False