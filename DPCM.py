import numpy as np
import matplotlib.pyplot as plt

Uniform_distrib = {2: 1.732, 4: 0.866, 8: 0.433, 16: 0.217, 32: 0.108}
Gaussian_distrib = {2: 1.596, 4: 0.9957, 8: 0.5860, 16: 0.3352, 32: 0.1881}
Laplacian_distrib = {2: 1.414, 4: 1.0873, 8: 0.7309, 16: 0.4609, 32: 0.2799}
e_mean_range = (-32, 32)
e_std_dev_range = (0, 200)
n_niveau_uniform = 64


def DPCM_encode(I_source, ArgumentX):
    n_niveaux = 2 ** ArgumentX[0]
    distribution = ArgumentX[1]
    match distribution:
        case "uniform":
            delta = Uniform_distrib[n_niveaux]
        case "gaussian":
            delta = Gaussian_distrib[n_niveaux]
        case "laplacian":
            delta = Laplacian_distrib[n_niveaux]
        case _:
            raise ValueError("Distribution inconnue. Choisissez parmi 'uniform', 'gaussian', ou 'laplacian'.")
    errors = boucle_initial(I_source)
    e_mean = np.mean(errors)
    e_std_dev = np.std(errors)
    print(f" mean = {e_mean}, std_dev = {e_std_dev}")
    e_mean_encoded = encode_uniforme(e_mean, e_mean_range[0], e_mean_range[1], n_niveau_uniform)
    e_std_dev_encoded = encode_uniforme(e_std_dev, e_std_dev_range[0], e_std_dev_range[1], n_niveau_uniform)

    e_mean_decoded = decode_uniforme(e_mean_encoded, e_mean_range[0], e_mean_range[1], n_niveau_uniform)
    e_std_dev_decoded = decode_uniforme(e_std_dev_encoded, e_std_dev_range[0], e_std_dev_range[1], n_niveau_uniform)
    print(f"delta = {delta}, decoded mean = {e_mean_decoded}, decoded std_dev = {e_std_dev_decoded}")
    L, C = I_source.shape
    I_decoded = np.zeros((L, C), dtype=np.float32)
    I_encoded = np.zeros((L, C), dtype=np.uint16)
    for id_l in range(0, L):
        for id_c in range(0, C):
            pred = predict(I_decoded, id_l, id_c)
            e = I_source[id_l, id_c] - pred
            e_squashed = (e - e_mean_decoded)/e_std_dev_decoded
            I_encoded[id_l][id_c] = encode(e_squashed, delta, n_niveaux, 0, n_niveaux-1)
            e_decoded = decode(I_encoded[id_l][id_c], delta, n_niveaux)
            e_decoded_unsquased = e_decoded * e_std_dev_decoded + e_mean_decoded
            I_decoded[id_l, id_c] =  e_decoded_unsquased + pred

    return I_encoded, e_mean_encoded, e_std_dev_encoded

def DPCM_decode(I_encoded, e_mean_encoded, e_std_dev_encoded, ArgumentX):
    n_niveaux = 2 ** ArgumentX[0]
    distribution = ArgumentX[1]
    match distribution:
        case "uniform":
            delta = Uniform_distrib[n_niveaux]
        case "gaussian":
            delta = Gaussian_distrib[n_niveaux]
        case "laplacian":
            delta = Laplacian_distrib[n_niveaux]
        case _:
            raise ValueError("Distribution inconnue. Choisissez parmi 'uniform', 'gaussian', ou 'laplacian'.")

    e_mean = decode_uniforme(e_mean_encoded, e_mean_range[0], e_mean_range[1], n_niveau_uniform)
    e_std_dev = decode_uniforme(e_std_dev_encoded, e_std_dev_range[0], e_std_dev_range[1], n_niveau_uniform)
    L, C = I_encoded.shape
    I_decoded= np.zeros((L, C), dtype=np.float32)
    for id_l in range(0, L):
        for id_c in range(0, C):
            pred = predict(I_decoded, id_l, id_c)
            decoded  = decode(I_encoded[id_l, id_c], delta, n_niveaux)
            unsquashed = decoded * e_std_dev + e_mean
            I_decoded[id_l][id_c] = unsquashed + pred

    return I_decoded

def encode(value, delta, n_niveaux, min, max):
    ind = np.floor(value / delta + (n_niveaux) // 2)
    ind = np.maximum(ind, min)
    ind = np.minimum(ind, max)
    return ind

def decode(ind, delta, n_niveaux):
    ind = float(ind)
    return (ind - (n_niveaux) // 2 + 0.5) * delta

def encode_uniforme(value, vmin, vmax, n_niveaux):
    pas = (vmax - vmin) / n_niveaux
    ind = np.floor((value - vmin) / pas)
    ind = np.clip(ind, 0, n_niveaux - 1)
    return ind.astype(np.uint8)

def decode_uniforme(ind, vmin, vmax, n_niveaux):
    pas = (vmax - vmin) / n_niveaux
    return vmin + (float(ind) + 0.5) * pas

def boucle_initial(I_source):
    L, C = I_source.shape
    errors = np.zeros((L, C), dtype=np.float32)
    for id_l in range(0, L):
        for id_c in range(0, C):
            pred = predict(I_source, id_l, id_c)
            errors[id_l][id_c] = I_source[id_l][id_c] - pred
    return errors

def predict(I, id_l, id_c):
    if id_l == id_c == 0:
        return 128
    elif id_l != id_c == 0:
        return I[id_l-1][id_c]
    elif id_c != id_l == 0:
        return I[id_l][id_c-1]
    else:
        a = I[id_l][id_c - 1]
        b = I[id_l - 1][id_c]
        c = I[id_l - 1][id_c - 1]
        if c >= max(a, b):
            return min(a, b)
        elif c <= min(a, b):
            return max(a, b)
        else:
            return a + b - c
