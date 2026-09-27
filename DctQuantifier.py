from collections import Counter

import numpy as np
from scipy.fft import dctn, idctn
from unicodedata import category

import HuffmanTable

quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                               [12, 12, 14, 19, 26, 58, 60, 55],
                               [14, 13, 16, 24, 40, 57, 69, 56],
                               [14, 17, 22, 29, 51, 87, 80, 62],
                               [18, 22, 37, 56, 68, 109, 103, 77],
                               [24, 35, 55, 64, 81, 104, 113, 92],
                               [49, 64, 78, 87, 103, 121, 120, 101],
                               [72, 92, 95, 98, 112, 100, 103, 99]])


def flatten_diagonally(matrix):
    matrix_diagonal = [[] for _ in range(2 * matrix.shape[0] - 1)]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            diagonal = i + j
            matrix_diagonal[diagonal].append(matrix[i][j])

    for i in range(0, len(matrix_diagonal), 2):
        matrix_diagonal[i].reverse()

    matrix_flatten = []
    [matrix_flatten.extend(x) for x in matrix_diagonal]

    return matrix_flatten


def reconstitute(matrix_flatten, bloc_size=8):
    matrix_flatten_copy = list(matrix_flatten.copy())
    element_per_diagonal = [x + 1 if x < bloc_size else 2 * bloc_size - 1 - x for x in
                            range(2 * bloc_size - 1)]

    matrix_diagonal = [[] for _ in range(2 * bloc_size - 1)]
    for i in range(len(matrix_diagonal)):
        matrix_diagonal[i].extend(
            [matrix_flatten_copy.pop(0) for _ in range(element_per_diagonal[i])])

    for i in range(0, len(matrix_diagonal), 2):
        matrix_diagonal[i].reverse()

    matrix = np.zeros((bloc_size, bloc_size))
    for i in range(bloc_size):
        for j in range(bloc_size):
            diagonal = i + j
            matrix[i][j] = matrix_diagonal[diagonal].pop(0)

    return matrix


def remove_trailing_zeros(matrix_flatten):
    matrix_flatten_copy = matrix_flatten.copy()
    for i in range(len(matrix_flatten_copy) - 1, 0, -1):
        if matrix_flatten_copy[i] == 0:
            matrix_flatten_copy = np.delete(matrix_flatten_copy, i)
        else:
            break

    return matrix_flatten_copy


def add_trailing_zeros(matrix_flatten, bloc_size=8):
    matrix_flatten_copy = matrix_flatten.copy()
    zeros = np.zeros(bloc_size ** 2 - len(matrix_flatten_copy))
    return np.concatenate((matrix_flatten_copy, zeros))

def to_huffman_encoding_AC(nZeros, value):
    category = 0 if value == 0 else abs(int(value)).bit_length()
    code = HuffmanTable.huffman_codes[f"{nZeros},{category}"]
    if category == 0:
        value_bits = ""
    elif value > 0:
        value_bits = format(value, f"0{category}b")
    else:
        value_bits = format(2**category - 1 + value, f"0{category}b")
    return code + value_bits

def from_huffman_encoding_AC(bits):
    ZrC = HuffmanTable.reversed_huffman_code.get(bits, "")
    if ZrC == "":
        return -1, -1

    values = [int(x) for x in ZrC.split(",")]
    return values[0], values[1]

def to_huffman_encoding_DC(value):
    category = 0 if value == 0 else abs(int(value)).bit_length()
    code = HuffmanTable.huffman_code_DC[f"{category}"]
    value_bits = ""
    if category == 0:
        value_bits = ""
    elif value > 0:
        value_bits = format(value, f"0{category}b")
    else:
        value_bits = format(2 ** category - 1 + value, f"0{category}b")
    return code + value_bits

def from_huffman_encoding_DC(bits):
    categ = HuffmanTable.reversed_huffman_code_DC.get(bits, "")
    if categ == "":
        return -1

    return int(categ)

def encode_to_huffman(Is):
    data = []
    for row in Is:
        dc = row[0]
        values = []
        nZeros = 0
        for i in range(1, len(row)):
            if row[i] == 0:
                nZeros += 1
                if nZeros == 16:
                    values.append((15, 0))
                    nZeros = 0
            else:
                values.append((nZeros, row[i]))
                nZeros = 0
        dc_code = to_huffman_encoding_DC(dc)
        ac_code = "".join(to_huffman_encoding_AC(r, v) for r, v in values)
        EOB_code = to_huffman_encoding_AC(0, 0)
        data.extend(dc_code + ac_code + EOB_code)
    return data

def _amplitude_value(bits, category):
    if category == 0:
        return 0
    code = int(bits, 2)
    half = 2 ** (category - 1)
    return code if code >= half else code - (2 ** category - 1)

def decode_from_huffman(data):
    Is = []
    sub_array = []
    # 0 - read DC
    # 1 - read value DC
    # 2 - read AC
    # 3 - read value AC
    state = 0
    bits_to_read = 0
    bits = ""
    for bit in data:
        match state:
            case 0: # read DC
                bits += bit
                categ = from_huffman_encoding_DC(bits)
                if categ != -1:
                    bits = ""
                    if categ == 0:
                        sub_array.append(0)
                        state = 2
                    else:
                        bits_to_read = categ
                        state = 1

            case 1: # read value DC
                bits += bit
                if len(bits) == bits_to_read:
                    sub_array.append(_amplitude_value(bits, bits_to_read))
                    bits = ""
                    state = 2
                    bits_to_read = 0

            case 2: # read AC
                bits += bit
                z, c = from_huffman_encoding_AC(bits)
                if z != -1:
                    bits = ""
                    if z == 0 and c == 0:
                        Is.append(sub_array)
                        sub_array = []
                        state = 0
                        bits_to_read = 0
                    elif z == 15 and c == 0: # 16 zeros
                        sub_array.extend([0] * 16)
                        state = 2
                    else :
                        sub_array.extend([0] * z)
                        state = 3
                        bits_to_read = c

            case 3: # read value AC
                bits += bit
                if len(bits) == bits_to_read:
                    sub_array.append(_amplitude_value(bits, bits_to_read))
                    bits = ""
                    state = 2
                    bits_to_read = 0

    return Is

def encode(Is, factor_K = 1, bloc_size=8):
    Ir = Is.copy()

    # dividable by bloc_size
    if Ir.shape[0] % bloc_size:
        pad = bloc_size - Ir.shape[0] % bloc_size
        Ir = np.vstack((Ir, np.tile(Ir[[-1], :], (pad, 1))))
    if Ir.shape[1] % bloc_size:
        pad = bloc_size - Ir.shape[1] % bloc_size
        Ir = np.hstack((Ir, np.tile(Ir[:, [-1]], (1, pad))))

    Ir = Ir - 2 ** (8 - 1)

    blocks = []
    for i in range(Ir.shape[0] // bloc_size):
        for j in range(Ir.shape[1] // bloc_size):
            block = Ir[i * bloc_size: (i + 1) * bloc_size,
            j * bloc_size: (j + 1) * bloc_size]

            dct_result = dctn(block)

            quantization_result = np.floor(
                dct_result / (factor_K * quantization_table) + 0.5).astype(int)

            flatten_matrix = flatten_diagonally(quantization_result)

            encoded_bloc = remove_trailing_zeros(flatten_matrix)

            blocks.append(encoded_bloc)

    return encode_to_huffman(blocks), Ir.shape[0] // bloc_size, Ir.shape[1] // bloc_size


def decode(huffman_encoding, row, col,
           row_b, col_b, factor_K = 1, bloc_size=8):
    blocks = decode_from_huffman(huffman_encoding)
    I_before_trunk = np.zeros((row_b * bloc_size, col_b * bloc_size))
    for i in range(row_b):
        for j in range(col_b):
            encoded_bloc = blocks.pop(0)

            flatten_matrix = add_trailing_zeros(encoded_bloc, bloc_size)
            quantization_result = reconstitute(flatten_matrix, bloc_size)

            dct_result = quantization_result * (factor_K * quantization_table)

            block = idctn(dct_result)

            I_before_trunk[i * bloc_size:(i + 1) * bloc_size,
            j * bloc_size:(j + 1) * bloc_size] = block

    return I_before_trunk[:row, :col] + 2 ** (8 - 1) + 1

if __name__ == "__main__":
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(matrix)
    flatten_matrix = flatten_diagonally(matrix)
    print(flatten_matrix)
    print(reconstitute(flatten_matrix, 3))