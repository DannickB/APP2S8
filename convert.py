from tkinter import Image
from PIL import Image
import numpy as np


def convert(I_source):
    I_source = np.array(I_source)
    l, c, _ = I_source.shape
    img_affichee = np.zeros((l,c))
    for id_l in range(l):
        for id_c in range(c):
            pixel = I_source[id_l][id_c]
            img_affichee[id_l][id_c] = 0.2126*pixel[0] + 0.7152*pixel[1] + 0.0722*pixel[2]

    return img_affichee
