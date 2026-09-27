import sys
import time

import VectorialQuantifier
from convert import convert
from reduce import reduce
from QS import QS_encode, QS_decode
from DPCM import DPCM_encode, DPCM_decode
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
# 5 = Technique de quantification par troncature de blocs (BTC)
# 6 = Technique de quantification adaptative (QA)
Choix = 2
# Charge l'image source
# A FAIRE : remplacer par votre propre chargement d'image (pas de librairie utilisee ici)
nom = 'cman'
I_source = Image.open(f"./ressources/{nom}.tif")

# Affiche l'image source
# A FAIRE : remplacer par votre propre affichage d'image
# Debut du chronometre
tic = time.time()
# ==========================================================================
#
# CONVERSION DE FORMAT DE CODAGE DES COULEURS
#
# ==========================================================================
try:
    I_source = convert(I_source)
except:
    I_source = np.array(I_source)
# ==========================================================================
#
# REDUCTION DE DIMENSIONS
#
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
#
# CODAGE
#
# ==========================================================================
args = [4, "laplacian"]
# ----------------------------------------------
# CODEUR - QV
# ----------------------------------------------
if Choix == 1:
    vector_size = 2
    n_bits_per_vector = 8
    I_encoded, representatives = VectorialQuantifier.encode(I_reduced, vector_size, n_bits_per_vector)
    I_metadata = representatives

# ----------------------------------------------
# CODEUR - DPCM
# ----------------------------------------------
if Choix == 2:
    # Appelle la fonction de codage
    I_encoded, e_mean, e_std_dev = DPCM_encode(I_reduced, args)
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
    facteur_K = 1.5
    I_encoded, a, b = dct.encode(I_reduced, facteur_K)
    I_metadata = (a, b)

# ----------------------------------------------
# CODEUR - BTC
# ----------------------------------------------
if Choix == 5:
    pass  # A FAIRE : Au choix.

# ----------------------------------------------
# CODEUR - QA
# ----------------------------------------------
if Choix == 6:
    pass  # A FAIRE : Au choix.

# ==========================================================================
#
# INTERFACE AVEC LA COUCHE PHYSIQUE
#
# ==========================================================================
# La fonction transmit envoie les donnees a transmettre sous un format
# compris par la couche physique. Elle prend en entree un dictionnaire de
# cellules ou la cellule N doit contenir les donnees qui seront codees sur
# N bits.
# Convention :
# Les elements de la cellule N doivent etre entre 0 et 2^N-1.
# A FAIRE : Remplir le dictionnaire de cellules Data a partir de I_encoded et
# I_metadata en respectant la convention de la couche physique.
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

# %%
# ==========================================================================
#
# RECOMPOSITION
#
# ==========================================================================
# A FAIRE : Recomposer I_encoded_Rx et I_metadata_Rx a partir de Data.
I_encoded_Rx = Data[4]
I_metadata_Rx = Data[5]
# ==========================================================================
#
# DECODAGE
#
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
    # Appelle la fonction de decodage
    I_encoded_Rx = np.array(I_encoded_Rx)
    I_decoded = DPCM_decode(I_encoded_Rx, *I_metadata_Rx, args)
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
    bits = dct.encode_to_huffman(I_encoded)
    # L == C toujours alors peut être exclue des calculs car est dupliqué. Même chose pour A == B
    print("Taille des métadonnées : ",  sys.getsizeof(l) + sys.getsizeof(a) + sys.getsizeof(facteur_K))
    print("Methode DCT bits/pixel: ", (len(bits) + 8*(sys.getsizeof(l) + sys.getsizeof(a) + sys.getsizeof(facteur_K)))
                                            / (I_reduced.shape[0] * I_reduced.shape[1]))

# ----------------------------------------------
# DECODEUR - BTC
# ----------------------------------------------
if Choix == 5:
    pass
# ----------------------------------------------
# DECODEUR - QA
# ----------------------------------------------
if Choix == 6:
    pass  # A FAIRE : Au choix.

# Affiche l'image quantifiee
# A FAIRE : remplacer par votre propre affichage d'image
plt.figure(2)
plt.title('Image decodée')
plt.imshow(I_decoded, cmap='gray')
plt.imsave(f"./ressources/{nom}_decoded.bmp", I_decoded, cmap='gray')
# ==========================================================================
#
# CALCUL DE LA PERFORMANCE
#
# ==========================================================================
# Fin du chronometre
Time = time.time() - tic
print(f"PSRN : {computePSNR(I_reduced, I_decoded)}")
print(f"Time : {Time}s")
plt.show()

