import os
import pydicom
import numpy as np
from glob import glob
from skimage.transform import resize


LABELS = {label: i for i, label in enumerate([
    'background',  # Background is LAST channel therefore
    'aorta',
    'la_cav',
    'lv_wall',
    'pa',
    'pleuraleffusion',
    'ra_cav',
    'rv_cav',
    'rv_wall',
    'lv_cav'])}  # LATER CHANNELS CAN OVER-WRITE EARLIER, IMPORTANT FOR LV CAVITY!


def pad_numpy_to_square(numpy):
    orig_h, orig_w = numpy.shape[:2]
    new_hw = max(orig_h, orig_w)
    row_from = (orig_w - orig_h) // 2 if orig_h < orig_w else 0
    col_from = (orig_h - orig_w) // 2 if orig_w < orig_h else 0
    if len(numpy.shape) == 3:
        new = np.zeros((new_hw, new_hw, numpy.shape[2]))
    else:
        new = np.zeros((new_hw, new_hw))
    new[row_from:row_from + orig_h, col_from:col_from + orig_w] = numpy
    return new


def numpy_img_from_dicom_path(dicom_path, size):
    dicom = pydicom.dcmread(dicom_path)
    frame = dicom.pixel_array
    window_center = dicom.WindowCenter
    window_width = dicom.WindowWidth
    window_minimum = max(0, window_center - window_width)
    frame = frame - window_minimum
    frame = frame / window_width
    frame = np.clip(frame, 0, 1)
    frame = resize(frame, size, order=1, anti_aliasing=True)
    frame = (frame * 255).astype(np.uint8)
    return frame


def get_fov_width_and_slice_spacing_from_dicom(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    pixel_spacing = float(dcm.PixelSpacing[0])
    width_px = int(dcm.Columns)
    fov_width_mm = pixel_spacing * width_px
    slice_spacing_mm = dcm.SpacingBetweenSlices
    return fov_width_mm, slice_spacing_mm


def load_dir_of_dicoms_as_numpy(dicom_dir, slice_dimensions):
    dicom_paths = sorted(glob(os.path.join(dicom_dir, "*.dcm")))

    # Load DICOM
    n_slices = len(dicom_paths)
    stack = np.zeros((n_slices, 1, slice_dimensions[0], slice_dimensions[1]), dtype=np.uint8)
    for i_dicom, dicom_path in enumerate(dicom_paths):
        stack[i_dicom, 0] = numpy_img_from_dicom_path(dicom_path=dicom_path, size=slice_dimensions)

    # Get pixel size and slice thickness
    fov_width_mm, slice_spacing_mm = get_fov_width_and_slice_spacing_from_dicom(dicom_paths[0])

    return stack / 255, fov_width_mm, slice_spacing_mm


def volumes_from_volume_and_slice_width(volume, fov_width_mm):
    volumes_ml = {}
    volumes_px = {}
    try:
        volume = volume.detach().cpu().numpy()
    except AttributeError:
        pass
    volmax = np.argmax(volume[0], axis=0)
    for label_name, label_id in LABELS.items():
        px = np.sum(volmax == label_id)
        volumes_px[label_name] = px
        volumes_ml[label_name] = px * (fov_width_mm / volume.shape[-1]) / 1000  # /1000 as ml not mm3
    return volumes_ml, volumes_px
