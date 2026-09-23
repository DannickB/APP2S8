import numpy as np
from scipy.fft import dctn, idctn


quantization_table = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
                               [12, 12, 14, 19,  26,  58,  60,  55],
                               [14, 13, 16, 24,  40,  57,  69,  56],
                               [14, 17, 22, 29,  51,  87,  80,  62],
                               [18, 22, 37, 56,  68, 109, 103,  77],
                               [24, 35, 55, 64,  81, 104, 113,  92],
                               [49, 64, 78, 87, 103, 121, 120, 101],
                               [72, 92, 95, 98, 112, 100, 103,  99]])

def flatten_diagonally(matrix):
    matrix_diagonal = [[] for _ in range(2*matrix.shape[0] - 1)]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            diagonal = i+j
            matrix_diagonal[diagonal].append(matrix[i][j])
            
    matrix_flatten = []
    [matrix_flatten.extend(x) for x in matrix_diagonal]

    return matrix_flatten

def reconstitute(matrix_flatten, bloc_size = 8):
    matrix_flatten_copy = list(matrix_flatten.copy())
    element_per_diagonal = [x+1 if x < bloc_size else 2*bloc_size-1-x for x in range(2*bloc_size-1)]

    matrix_diagonal = [[] for _ in range(2*bloc_size-1)]
    for i in range(len(matrix_diagonal)):
        matrix_diagonal[i].extend([matrix_flatten_copy.pop(0) for _ in range(element_per_diagonal[i])])

    matrix = np.zeros((bloc_size, bloc_size))
    for i in range(bloc_size):
        for j in range(bloc_size):
            diagonal = i+j
            matrix[i][j] = matrix_diagonal[diagonal].pop(0)

    return matrix

def remove_trailing_zeros(matrix_flatten):
    matrix_flatten_copy = matrix_flatten.copy()
    for i in range(len(matrix_flatten_copy)-1, 0, -1):
        if matrix_flatten_copy[i] == 0:
            matrix_flatten_copy = np.delete(matrix_flatten_copy, i)
        else:
            break

    return matrix_flatten_copy

def add_trailing_zeros(matrix_flatten, bloc_size = 8):
    matrix_flatten_copy = matrix_flatten.copy()
    zeros = np.zeros(bloc_size**2 - len(matrix_flatten_copy))
    return np.concatenate((matrix_flatten_copy, zeros))


def encode(Is, bloc_size = 8):
    Ir = Is.copy()

    # dividable by bloc_size
    if Ir.shape[0] % bloc_size:
        pad = bloc_size - Ir.shape[0] % bloc_size
        Ir = np.vstack((Ir, np.tile(Ir[[-1], :], (pad, 1))))
    if Ir.shape[1] % bloc_size:
        pad = bloc_size - Ir.shape[1] % bloc_size
        Ir = np.hstack((Ir, np.tile(Ir[:, [-1]], (1, pad))))

    Ir = Ir - 2**(8-1)

    blocks = []
    for i in range(Ir.shape[0] // bloc_size):
        for j in range(Ir.shape[1] // bloc_size):
            block = Ir[i * bloc_size : (i + 1) * bloc_size,
                       j * bloc_size: (j + 1) * bloc_size]

            dct_result = dctn(block, norm='ortho')

            quantization_result = np.round(dct_result / quantization_table).astype(int)

            flatten_matrix = flatten_diagonally(quantization_result)

            encoded_bloc = remove_trailing_zeros(flatten_matrix)

            blocks.append(encoded_bloc)

    return blocks, Ir.shape[0] // bloc_size, Ir.shape[1] // bloc_size

def decode(blocks, row, col,
           row_b, col_b, bloc_size = 8):
    blocks_copy = blocks.copy()
    I_before_trunk = np.zeros((row_b * bloc_size, col_b * bloc_size))
    for i in range(row_b):
        for j in range(col_b):
            encoded_bloc = blocks_copy.pop(0)

            flatten_matrix = add_trailing_zeros(encoded_bloc, bloc_size)
            quantization_result = reconstitute(flatten_matrix, bloc_size)

            dct_result = quantization_result * quantization_table

            block = idctn(dct_result, norm='ortho')

            I_before_trunk[i * bloc_size:(i + 1) * bloc_size,
                           j * bloc_size:(j + 1) * bloc_size] = block

    return I_before_trunk[:row, :col] + 2**(8-1)






