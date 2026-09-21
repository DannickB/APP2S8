function PSNR = computePSNR(I_reduced,I_decoded)
% Différence de l'image synthétisée et de l'image source
image_MSE = I_decoded-I_reduced;
% Réarrangement sur une ligne
image_MSE = reshape(image_MSE,1,numel(I_reduced));
% Calcul de l'erreur quadratique moyenne
MSE = sum((image_MSE).^2)/length(image_MSE);
% Calcul du PSNR
PSNR = 10*log10(255^2/MSE);