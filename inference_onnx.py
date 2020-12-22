import os
import numpy as np
import onnxruntime as ort

from scipy.ndimage import zoom

from lib.data import load_dir_of_dicoms_as_numpy, LABELS, volumes_from_volume_and_slice_width
from lib.calculations import rescale_scaled_volume_to_ml, get_aortic_diameter_from_volumes_numpy

N_CLASSES_HRNET = len(LABELS)
MAX_VIEWPORT = 350
SLICE_DIMENSIONS_PX = (560, 560)

MODEL_PATH = "./models/exported_apayn_hrnet_model.onnx"

STUDY_PATHS = [(r"E:\Data\APAYN_Examples\anon\RYJ10901891", 1.565217391)]

VIS = True

# LOAD MODEL
sess = ort.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

for study_path, bsa in STUDY_PATHS:
    # Load dicom
    stack, fov_width_mm, slice_spacing_mm = load_dir_of_dicoms_as_numpy(study_path, SLICE_DIMENSIONS_PX)
    x = stack.astype(np.float32)  # 40x1x560x560

    # Forward pass
    hrnet_out = sess.run([output_name], {input_name: x})[0]

    # Scale 3D
    n_slices, colourchannels, dim_y, dim_x = x.shape
    width_to_vheight_ratio = fov_width_mm / slice_spacing_mm
    z_out = round((dim_x / width_to_vheight_ratio) * n_slices // 4)
    hrnet_out = np.swapaxes(hrnet_out, 0, 1)  # Move channel dim before slice dim; e.g. 10*40*140*140
    volume = np.expand_dims(zoom(hrnet_out, (1, z_out/n_slices, 1, 1)), 0)

    volumes_scaled, volumes_px = volumes_from_volume_and_slice_width(volume, fov_width_mm)

    lvd, rvd, lvm = rescale_scaled_volume_to_ml(volumes_scaled)
    aorta = get_aortic_diameter_from_volumes_numpy(volume)
    lvd_i, rvd_i, lvm_i, aorta_i = lvd / bsa, rvd / bsa, lvm / bsa, aorta / bsa

    print(f"{os.path.basename(study_path)}\n"
          f"LV diastolic volume:\t{lvd:.1f}ml\t({lvd_i:.1f}ml/m2)\n"
          f"RV diastolic volume:\t{rvd:.1f}ml\t({rvd_i:.1f}ml/m2)\n"
          f"LV mass:\t\t\t\t{lvm:.1f}ml\t({lvm_i:.1f}ml/m2)\n"
          f"Asc. aorta diameter:\t{aorta:.1f}mm\t({aorta_i:.1f}mm/m2)\n")

    # VIS
    if VIS:
        import numpy as np
        import sys, traceback
        from lib.vis import visualise_label_or_pred

        stack_out = np.argmax(volume[0], axis=0)
        stack_vis = []
        for slice in stack_out:
            stack_vis.append(visualise_label_or_pred(slice, plot=False, upsample_seg=0))
        stack_vis = np.stack(stack_vis)
        alpha = np.zeros(stack_vis.shape[:-1])[:, :, :, np.newaxis]
        alpha[np.any(stack_vis, axis=-1)] = 0.5 / 1.5
        stack_vis_alpha = np.concatenate((stack_vis, alpha), axis=-1)

        from pyqtgraph.Qt import QtCore, QtGui
        import pyqtgraph.opengl as gl

        if QtCore.QT_VERSION >= 0x50501:
            def excepthook(type_, value, traceback_):
                traceback.print_exception(type_, value, traceback_)
                QtCore.qFatal('')
        sys.excepthook = excepthook

        app = QtGui.QApplication([])
        w = gl.GLViewWidget()
        w.opts['distance'] = 200
        w.show()
        w.setWindowTitle('pyqtgraph example: GLVolumeItem')

        v = gl.GLVolumeItem(stack_vis_alpha.transpose((1, 2, 0, 3)) * 255)
        v.translate(-50, -50, 0)

        w.addItem(v)

        ax = gl.GLAxisItem()
        w.addItem(ax)

        if __name__ == '__main__':
            import sys

            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtGui.QApplication.instance().exec_()
