%==========================================================================
%
% S7 Codage de l'information APP2 - Solution
%
% La solution est divisée en 2 parties, le codage et le décodage. Ces deux
% sections sont délimitées par le symbole %% et peuvent être démarrées 
% séparément à l'aide de la commande ctrl+enter. Aucune donnée ne peut
% passer directement du codage au décodage, i.e. sans passer par la variable
% Data qui simule la couche physique.
%
%%
clear all
close all
clc
%==========================================================================
%
% SÉLECTION DES PARAMÈTRES
%
%==========================================================================
% Choix de la quantification
% 1 = Technique de quantification vectorielle (QV)
% 2 = Technique de quantification différentielle (DPCM)
% 3 = Technique de quantification scalaire (QS)
% 4 = Technique de quantification par transformée en cosinus discrète (DCT)
% 5 = Technique de quantification par troncature de blocs (BTC)
% 6 = Technique de quantification adaptative (QA)
Choix = 1;
% Charge l'image source
I_source = im2double(imread('crest.bmp'))*255;
% Affiche l'image source
figure(1)
imshow(I_source/255);
% Début du chronomètre
tic;
%==========================================================================
%
% CONVERSION DE FORMAT DE CODAGE DES COULEURS
%
%==========================================================================
I_source = convert(I_source);

%==========================================================================
%
% RÉDUCTION DE DIMENSIONS
%
%==========================================================================
% Dimensions désirées
LIGNES = 256;
COLONNES = 256;
% Appelle la fonction d'interpolation
I_reduced = reduce(I_source,LIGNES,COLONNES);

%==========================================================================
%
% CODAGE
%
%==========================================================================
%----------------------------------------------
% CODEUR - QV
%----------------------------------------------
if Choix == 1
    % Paramètres d'entrée
    % À FAIRE : Remplacer ArgumentX par votre/vos paramètre(s) d'entrée
    ArgumentX = 0;
    % Appelle la fonction de codage
    [I_encoded,I_metadata] = QV_encode(I_reduced,ArgumentX);
end

%----------------------------------------------
% CODEUR - DPCM
%----------------------------------------------
if Choix == 2
    % Paramètres d'entrée
    % À FAIRE : Remplacer ArgumentX par votre/vos paramètre(s) d'entrée
    ArgumentX = 0;
    % Appelle la fonction de codage
    [I_encoded,I_metadata] = DPCM_encode(I_reduced,ArgumentX);
end

%----------------------------------------------
% CODEUR - QS
%----------------------------------------------
if Choix == 3
% À FAIRE : Au choix.
end

%----------------------------------------------
% CODEUR - DCT
%----------------------------------------------
if Choix == 4
% À FAIRE : Au choix.
end

%----------------------------------------------
% CODEUR - BTC
%----------------------------------------------
if Choix == 5
% À FAIRE : Au choix.
end

%----------------------------------------------
% CODEUR - QA
%----------------------------------------------
if Choix == 6
% À FAIRE : Au choix.
end

%==========================================================================
%
% INTERFACE AVEC LA COUCHE PHYSIQUE
%
%==========================================================================
% La fonction transmit envoie les données à transmettre sous un format
% compris par la couche physique. Elle prend en entrée un tableau de
% cellules où la cellule N doit contenir les données qui seront codées sur
% N bits.
% Convention :
% Les éléments de la cellule N doivent être entre 0 et 2^N-1.
% À FAIRE : Remplir le tableau de cellules Data à partir de I_encoded et
% I_metadata en respectant la convention de la couche physique.
Data{8} = I_encoded;
Data{1} = I_metadata;
% Appelle de la fonction de transmission
Budget = transmit(Data);
% Si une erreur a été détectée par la fonction d'interface
if Budget < 0
    % Affichage de l'erreur
    fprintf('Erreur : Une donnée dépasse la gamme dynamique à la cellule %i.\n',-Budget) 
end
%%
%==========================================================================
%
% RECOMPOSITION
%
%==========================================================================
% À FAIRE : Recomposer I_encoded_Rx et I_metadata_Rx à partir de Data.
I_encoded_Rx = Data{8};
I_metadata_Rx = Data{1};
%==========================================================================
%
% DÉCODAGE
%
%==========================================================================
%----------------------------------------------
% DÉCODEUR - QV
%----------------------------------------------
if Choix == 1
    % Paramètres d'entrée
    % À FAIRE : Remplacer ArgumentY par votre/vos paramètre(s) d'entrée
    ArgumentY = 0;
    % Appelle la fonction de décodage  
    I_decoded = QV_decode(I_encoded_Rx,I_metadata_Rx,ArgumentY);
end
%----------------------------------------------
% DÉCODEUR - DPCM
%----------------------------------------------
if Choix == 2
    % Paramètres d'entrée
    % À FAIRE : Remplacer ArgumentY par votre/vos paramètre(s) d'entrée
    ArgumentY = 0;
    % Appelle la fonction de décodage  
    I_decoded = DPCM_decode(I_encoded_Rx,I_metadata_Rx,ArgumentY);    
end

%----------------------------------------------
% DÉCODEUR - QS
%----------------------------------------------
if Choix == 3
% À FAIRE : Au choix.
end

%----------------------------------------------
% DÉCODEUR - DCT
%----------------------------------------------
if Choix == 4
% À FAIRE : Au choix.
end

%----------------------------------------------
% DÉCODEUR - BTC
%----------------------------------------------
if Choix == 5
% À FAIRE : Au choix.
end

%----------------------------------------------
% DÉCODEUR - QA
%----------------------------------------------
if Choix == 6
% À FAIRE : Au choix.
end

% Affiche l'image quantifiée
figure(2)
imshow(I_decoded/255);

%==========================================================================
%
% CALCUL DE LA PERFORMANCE
%
%==========================================================================
% Fin du chronomètre
Time = toc;
% Si aucune erreur n'a été détectée
if Budget > 0
    % Calcul du PSNR
    PSNR = computePSNR(I_reduced,I_decoded);
    % Calcul du débit
    Rate = Budget/numel(I_decoded);
    % Affichage des performances
    fprintf('********* Résultats *********\n')
    fprintf('Temps écoulé: %.2f s\nPSNR: %.2f dB\nRate: %.2f bits/pixel\n', Time, PSNR, Rate)
    fprintf('*****************************\n')
end

