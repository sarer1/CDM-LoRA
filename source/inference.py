"""Hanning-weighted sliding-window inference for paired imagery."""
import numpy as np
import torch

import utils as U
from utils import N_CLASSES, WINDOW_SIZE


def infer_tile(net, mods, stride, batch_size, window_size=WINDOW_SIZE):
    """Hanning-weighted sliding-window inference over one tile → argmax map."""
    img = mods[0]
    pred = np.zeros(img.shape[:2] + (N_CLASSES,), dtype=np.float32)
    pw, ph = window_size
    wy, wx = np.hanning(ph), np.hanning(pw)
    weight = (wy[:, None] * wx[None, :]).astype(np.float32)
    with torch.no_grad():
        for coords in U.grouper(batch_size, U.sliding_window(img, step=stride, window_size=window_size)):
            mod_patches_list = []
            for mod_data in mods:
                if mod_data.ndim == 3:
                    patches = [np.copy(mod_data[x:x + w, y:y + h]).transpose((2, 0, 1))
                               for x, y, w, h in coords]
                else:
                    patches = [np.copy(mod_data[x:x + w, y:y + h]) for x, y, w, h in coords]
                mod_patches_list.append(torch.from_numpy(np.asarray(patches)).cuda())
            outs = net(mod_patches_list, mode='Test').cpu().numpy()
            for out, (x, y, w_n, h_n) in zip(outs, coords):
                pred[x:x + w_n, y:y + h_n] += out.transpose((1, 2, 0)) * weight[:, :, None]
    return np.argmax(pred, axis=-1)
