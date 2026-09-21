import numpy as np
import matplotlib.pyplot as plt
Uniform_distrib = {2:1.732, 4:0.866, 8:0.433, 16:0.217, 32:0.108}
Gaussian_distrib = {2:1.596, 4:0.9957, 8:0.5860, 16:0.3352, 32:0.1881}
Laplacian_distrib = {2:1.414, 4:1.0873, 8:0.7309, 16:0.4609, 32:0.2799}

def QS_encode(I_source, ArgumentX):
    n_niveaux = 2**ArgumentX["n_nits"]
    distribution = ArgumentX["distribution"]
    match distribution:
        case "uniform":
            delta = Uniform_distrib[n_niveaux]
        case "gaussian":
            delta = Gaussian_distrib[n_niveaux]
        case "laplacian":
            delta = Laplacian_distrib[n_niveaux]
        case _:
            raise ValueError("Distribution inconnue. Choisissez parmi 'uniform', 'gaussian', ou 'laplacian'.")
    mean  = np.mean(I_source)
    std_dev  = np.std(I_source)
    print(f"delta = {delta}, mean = {mean}, std_dev = {std_dev}")
    L, C = I_source.shape
    I_squashed = np.zeros((L, C), dtype=np.float32)
    I_encoded = np.zeros((L, C), dtype=np.uint16)
    for id_l in range (0, L):
        for id_c in range (0, C):
            I_squashed[id_l][id_c] = (float(I_source[id_l][id_c])-mean)/std_dev
            ind = np.floor(I_squashed[id_l][id_c]/delta+(n_niveaux)//2)
            ind = np.maximum(ind, 0)
            ind = np.minimum(ind, n_niveaux-1)
            I_encoded[id_l][id_c] = ind
    
    print(I_squashed[id_l][id_c].dtype)
    I_metadata = {'mean': mean, 'std_dev': std_dev, 'n_nits': ArgumentX["n_nits"], 'distribution': distribution}
    return I_encoded, I_metadata