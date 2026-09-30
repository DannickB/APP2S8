import sys
import time

import VectorialQuantifier
from convert import convert
from reduce import reduce
from QS import QS_encode, QS_decode
from DPCM_encode import DPCM_encode
from DPCM_decode import DPCM_decode
import DctQuantifier as dct
from transmit import transmit
from computePSNR import computePSNR

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================================
#
# S7 Codage de l'information APP2 - Solution
#
# La solution est divisee en 2 parties, le codage et le decodage.
# Aucune donnee ne peut passer directement du codage au decodage,
# i.e. sans passer par la variable Data qui simule la couche physique.
#
# ==========================================================================
#
# SELECTION DES PARAMETRES
#
# ==========================================================================
# Choix de la quantification
# 1 = Technique de quantification vectorielle (QV)
# 2 = Technique de quantification differentielle (DPCM)
# 3 = Technique de quantification scalaire (QS)
# 4 = Technique de quantification par transformee en cosinus discrete (DCT)
Choix = 1
# Charge l'image source
# A FAIRE : remplacer par votre propre chargement d'image (pas de librairie utilisee ici)
nom = 'AUTRE21'
I_source = Image.open(f"./ressources/{nom}.bmp")

tic = time.time()
# ==========================================================================
# CONVERSION DE FORMAT DE CODAGE DES COULEURS
# ==========================================================================
try:
    I_source = convert(I_source)
except:
    I_source = np.array(I_source)
# ==========================================================================
# REDUCTION DE DIMENSIONS
# ==========================================================================
# Dimensions desirees
LIGNES = 256
COLONNES = 256
# Appelle la fonction d'interpolation
I_reduced = reduce(I_source, LIGNES, COLONNES)
plt.figure(1)
plt.title('Image reduite')
plt.imshow(I_reduced, cmap='gray')

# ==========================================================================
# CODAGE
# ==========================================================================
# ----------------------------------------------
# CODEUR - QV
# ----------------------------------------------
if Choix == 1:
    vector_size = 3
    n_bits_per_vector = 9
    I_encoded, representatives = VectorialQuantifier.encode(I_reduced, vector_size, n_bits_per_vector)
    I_metadata = representatives

# ----------------------------------------------
# CODEUR - DPCM
# ----------------------------------------------
if Choix == 2:
    # Appelle la fonction de codage
    I_encoded, e_mean, e_std_dev = DPCM_encode(I_reduced)
    I_metadata = (e_mean, e_std_dev)
    print(np.unique(I_encoded))
    I_encoded = I_encoded.tolist()

# ----------------------------------------------
# CODEUR - QS
# ----------------------------------------------
if Choix == 3:
    args = {"n_nits":3, "distribution":"gaussian"}
    I_encoded, I_metadata = QS_encode(I_reduced, args)
    print(np.unique(I_encoded))

# ----------------------------------------------
# CODEUR - DCT
# ----------------------------------------------
if Choix == 4:
    facteur_K = 1.25
    I_encoded, a, b = dct.encode(I_reduced, facteur_K)
    I_metadata = (a, b)

# ==========================================================================
#
# INTERFACE AVEC LA COUCHE PHYSIQUE
#
# ==========================================================================
Data = {}
Data[4] = I_encoded
Data[6] = I_metadata
# Appelle de la fonction de transmission
if Choix == 2:
    Budget = transmit(Data)
else:
    Budget = -1
# Si une erreur a ete detectee par la fonction d'interface
if Budget < 0:
    # Affichage de l'erreur
    print('Erreur : Une donnee depasse la gamme dynamique a la cellule %i.' % -Budget)

# ==========================================================================
# RECOMPOSITION
# ==========================================================================

I_encoded_Rx = Data[4]
I_metadata_Rx = Data[6]

# ==========================================================================
# DECODAGE
# ==========================================================================

# ----------------------------------------------
# DECODEUR - QV
# ----------------------------------------------
if Choix == 1:
    I_decoded = VectorialQuantifier.decode(I_encoded, vector_size, representatives, I_reduced.shape[0], I_reduced.shape[1])
    nbits_per_pixel = VectorialQuantifier.find_bits_per_pixel(I_encoded, vector_size, n_bits_per_vector, I_reduced.shape[0] * I_reduced.shape[1])
    print("Methode vectoriel nbits/pixel: ",  nbits_per_pixel)

# ----------------------------------------------
# DECODEUR - DPCM
# ----------------------------------------------
if Choix == 2:
    I_encoded_Rx = np.array(I_encoded_Rx)
    I_decoded = DPCM_decode(I_encoded_Rx, *I_metadata_Rx)
    taille_metadata = sys.getsizeof(I_metadata_Rx)
    Rate = Budget / (len(I_decoded) * len(I_decoded[0])) + taille_metadata/(LIGNES*COLONNES)
    print("Taille des métadonnées : ",  taille_metadata)
    print("Methode DPCM bits/pixel: ", Rate)

# ----------------------------------------------
# DECODEUR - QS
# ----------------------------------------------
if Choix == 3:
    I_decoded = QS_decode(I_encoded, I_metadata)

# ----------------------------------------------
# DECODEUR - DCT
# ----------------------------------------------
if Choix == 4:
    l, c = I_reduced.shape
    I_decoded = dct.decode(I_encoded,l, c, a, b, facteur_K)
    # L == C toujours alors peut être exclue des calculs car est dupliqué. Même chose pour A == B
    print("Taille des métadonnées : ",  sys.getsizeof(l) + sys.getsizeof(a) + sys.getsizeof(facteur_K))
    print("Methode DCT bits/pixel: ", (len(I_encoded) + 8*(sys.getsizeof(l) + sys.getsizeof(a) + sys.getsizeof(facteur_K)))
                                            / (I_reduced.shape[0] * I_reduced.shape[1]))

# Affiche l'image quantifiee
plt.figure(2)
plt.title('Image decodée')
plt.imshow(I_decoded, cmap='gray')
plt.imsave(f"./ressources/{nom}_decoded.bmp", I_decoded, cmap='gray')

# ==========================================================================
# CALCUL DE LA PERFORMANCE
# ==========================================================================
# Fin du chronometre
Time = time.time() - tic
print(f"PSRN : {computePSNR(I_reduced, I_decoded)}")
print(f"Time : {Time}s")
plt.show()

