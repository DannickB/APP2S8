import numpy as np

# Table de Lloy-Max
lloyd_max_seuil = [
    -np.inf,
    -3.7123, -2.5864, -1.8700, -1.3393, -0.9167, -0.5651, -0.2637, 0.0000,
     0.2637,  0.5651,  0.9167,  1.3393,  1.8700,  2.5864,  3.7123,
     np.inf
]
lloyd_max_valeurs = [
    -4.4201, -3.0045, -2.1682, -1.5717,
    -1.1069, -0.7264, -0.4038, -0.1235,
     0.1235,  0.4038,  0.7264,  1.1069,
     1.5717,  2.1682,  3.0045,  4.4201
]
# Valeurs pour le quantificateur uniforme des metadonnées
e_mean_range = (-32, 32)
e_std_dev_range = (0, 200)
n_niveau_uniform = 64


def DPCM_encode(I_source):
    #Encodeur du DPCM
    errors = boucle_initial(I_source)
    e_mean = np.mean(errors)
    e_std_dev = np.std(errors)
    #Encodage des metadonnées
    e_mean_encoded = encode_uniforme(e_mean, e_mean_range[0], e_mean_range[1], n_niveau_uniform)
    e_std_dev_encoded = encode_uniforme(e_std_dev, e_std_dev_range[0], e_std_dev_range[1], n_niveau_uniform)

    #Décodage des métadonnées
    e_mean_decoded = decode_uniforme(e_mean_encoded, e_mean_range[0], e_mean_range[1], n_niveau_uniform)
    e_std_dev_decoded = decode_uniforme(e_std_dev_encoded, e_std_dev_range[0], e_std_dev_range[1], n_niveau_uniform)
    L, C = I_source.shape
    I_decoded = np.zeros((L, C), dtype=np.float32)
    I_encoded = np.zeros((L, C), dtype=np.uint16)
    for id_l in range(0, L):
        for id_c in range(0, C):
            pred = predict(I_decoded, id_l, id_c)
            e = I_source[id_l, id_c] - pred
            e_squashed = (e - e_mean_decoded)/e_std_dev_decoded
            I_encoded[id_l][id_c] = encode(e_squashed)
            # Décodeur local
            e_decoded = decode(I_encoded[id_l][id_c])
            e_decoded_unsquased = e_decoded * e_std_dev_decoded + e_mean_decoded
            I_decoded[id_l, id_c] =  e_decoded_unsquased + pred

    return I_encoded, e_mean_encoded, e_std_dev_encoded

def encode(value):
    # la table de Lloyd-Max est dilaté par un facteur de 1.3 pour mieux représenter les donneés
    ind = np.digitize(value/1.3, lloyd_max_seuil) - 1
    ind = np.maximum(ind, 0)
    ind = np.minimum(ind, 15)
    return ind

def decode(ind):
    # la table de Lloyd-Max est dilaté par un facteur de 1.3 pour mieux représenter les donneés
    return lloyd_max_valeurs[ind]*1.3

def encode_uniforme(value, vmin, vmax, n_niveaux):
    #encodeur des métadonnées
    pas = (vmax - vmin) / n_niveaux
    ind = np.floor((value - vmin) / pas)
    ind = np.clip(ind, 0, n_niveaux - 1)
    return ind.astype(np.uint8)

def decode_uniforme(ind, vmin, vmax, n_niveaux):
    # decodeur des métadonnées
    pas = (vmax - vmin) / n_niveaux
    return vmin + (float(ind) + 0.5) * pas

def boucle_initial(I_source):
    # Fait la prédiction sur toute l'image avant l'encodage
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
        #moyenne des pixels haut et gauche
        return (I[id_l][id_c-1]+I[id_l-1][id_c])/2
