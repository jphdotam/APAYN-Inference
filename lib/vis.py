import sys
import numpy as np
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt

from skimage.color import label2rgb
from skimage.transform import rescale
from pyqtgraph.Qt import QtCore, QtGui

import torch

from lib.data import LABELS

N_LABELS = len(LABELS)

def get_colourmap(offset_by_1=False):
    """Adapted from & thanks to https://github.com/wkentaro/labelme/blob/master/labelme/utils/draw.py"""
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N_LABELS, 3))
    for i in range(0, N_LABELS):
        id = i
        if offset_by_1:
            id+=1
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

def visualise_label_or_pred(y, x=None, upsample_seg=4, plot=True, title=None):
    if type(x) == torch.Tensor:
        x = x.numpy()
    if type(y) == torch.Tensor:
        y = y.numpy()
    if len(y.shape) == 3:
        if upsample_seg and upsample_seg != 1:
            y = y.transpose((1,2,0))
            # try:
            #     y = rescale(y, (upsample_seg, upsample_seg, y.shape[-1]))
            # except:
            y = rescale(y, (upsample_seg, upsample_seg), multichannel=True)
            y = y.transpose((2,0,1))
            y = np.argmax(y, axis=0)
    elif len(y.shape) == 2 and upsample_seg and upsample_seg != 1:
        y = rescale(y.astype(np.float32), (upsample_seg, upsample_seg), order=0, anti_aliasing=False).astype(np.uint8)
    if x is not None:
        cmap = get_colourmap()
        rgb = label2rgb(y, image=x, colors=cmap)  # Have to use the scikit version if overlaying
    else:
        cmap = get_colourmap()
        rgb = cmap[y.ravel()].reshape(y.shape + (3,))  # This is better as colours standardised
    if plot:
        plt.imshow(rgb)
        if title:
            plt.title(title)
        plt.show()
        return rgb
    else:
        return rgb

def draw_volume(volume):
    stack_out = np.argmax(volume[0].detach().cpu().numpy(), axis=0).astype(np.uint8)
    stack_vis = []
    for slice in stack_out:
        stack_vis.append(visualise_label_or_pred(slice, plot=False, upsample_seg=0))
    stack_vis = np.stack(stack_vis)
    alpha = np.zeros(stack_vis.shape[:-1])[:,:,:,np.newaxis]
    alpha[np.any(stack_vis,axis=-1)] = 0.5/2.5
    stack_vis_alpha = np.concatenate((stack_vis, alpha), axis=-1)

    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 200
    w.show()
    w.setWindowTitle('pyqtgraph example: GLVolumeItem')

    # b = gl.GLBoxItem()
    # w.addItem(b)
    g = gl.GLGridItem()
    g.scale(10, 10, 1)
    w.addItem(g)

    v = gl.GLVolumeItem(stack_vis_alpha.transpose((1,2,0,3))*255)
    v.translate(-50, -50, -stack_vis_alpha.shape[0]/3)
    w.addItem(v)

    ax = gl.GLAxisItem()
    w.addItem(ax)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

