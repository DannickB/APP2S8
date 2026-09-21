import numpy as np

def reduce(I_source, LIGNES, COLONNES):
    I_source = np.array(I_source)
    source_L, source_c = I_source.shape
    I_reduced = np.zeros((LIGNES, COLONNES), dtype=np.uint8)
    for L in range (0, LIGNES):
        for C in range (0, COLONNES):
            ll = np.round(L * source_L//LIGNES)
            cc = np.round(C * source_c//COLONNES)
            I_reduced[L][C] = I_source[ll][cc]
    return I_reduced
