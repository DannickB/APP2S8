import math


def computePSNR(I_reduced, I_decoded):
    # Difference de l'image synthetisee et de l'image source
    image_MSE = [[I_decoded[i][j] - I_reduced[i][j] for j in range(len(I_reduced[0]))] for i in range(len(I_reduced))]
    # Rearrangement sur une ligne
    image_MSE = [val for row in image_MSE for val in row]
    # Calcul de l'erreur quadratique moyenne
    MSE = sum(val ** 2 for val in image_MSE) / len(image_MSE)
    # Calcul du PSNR
    PSNR = 10 * math.log10(255 ** 2 / MSE)
    return PSNR
