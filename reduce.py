import numpy as np


def reduce(I_source, LIGNES, COLONNES):
    I_source = np.array(I_source)
    source_L, source_c = I_source.shape
    I_reduced = np.zeros((LIGNES, COLONNES), dtype=np.float32)
    for L in range(0, LIGNES):
        for C in range(0, COLONNES):
            src_l = L * source_L / LIGNES
            src_c = C * source_c / COLONNES

            l0 = int(np.floor(src_l))
            c0 = int(np.floor(src_c))
            l1 = min(l0 + 1, source_L - 1)
            c1 = min(c0 + 1, source_c - 1)

            dl = src_l - l0
            dc = src_c - c0

            HG = I_source[l0, c0]
            HD = I_source[l0, c1]
            BG = I_source[l1, c0]
            BD = I_source[l1, c1]

            haut = (1 - dc) * HG + dc * HD
            bas = (1 - dc) * BG + dc * BD
            I_reduced[L][C] = (1 - dl) * haut + dl * bas

    return I_reduced
