import numpy as np
Uniform_distrib = {2:1.732, 4:0.866, 8:0.433, 16:0.217, 32:0.108}
Gaussian_distrib = {2:1.596, 4:0.9957, 8:0.5860, 16:0.3352, 32:0.1881}
Laplacian_distrib = {2:1.414, 4:1.0873, 8:0.7309, 16:0.4609, 32:0.2799}

def QS_decode(I_encoded, metadata):
    n_niveaux = 2**metadata["n_nits"]
    distribution = metadata["distribution"]
    match distribution:
        case "uniform":
            delta = Uniform_distrib[n_niveaux]
        case "gaussian":
            delta = Gaussian_distrib[n_niveaux]
        case "laplacian":
            delta = Laplacian_distrib[n_niveaux]
        case _:
            raise ValueError("Distribution inconnue. Choisissez parmi 'uniform', 'gaussian', ou 'laplacian'.")

    mean  = metadata["mean"]
    std_dev  = metadata["std_dev"]
    L, C = I_encoded.shape
    I_unsquashed = np.zeros((L, C), dtype=np.float32)
    for id_l in range (0, L):
        for id_c in range (0, C):
            decoded = (int(I_encoded[id_l][id_c])-(n_niveaux)//2)*delta
            I_unsquashed[id_l][id_c] = decoded*std_dev+mean

    return I_unsquashed
