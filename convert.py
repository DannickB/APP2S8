from tkinter import Image
from PIL import Image
import numpy as np


def convert(I_source):
    I_source = np.array(I_source)
    img_affichee = Image.fromarray(I_source.astype('uint8'))
    return img_affichee.convert('L')
