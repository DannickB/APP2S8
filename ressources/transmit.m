function budget_data = transmit(Data)
% Initialisation
budget_data = 0;
% Pour chaque cellule
for i = 1:numel(Data)
	% Si la donnée est plus petite que la gamme dynamique
	if min(reshape(Data{i},1,numel(Data{i}))) < 0
		% Sortir un code d'erreur
		budget_data = -i;
		break;
	end
	% Si la donnée est plus grande que la gamme dynamique
	if max(reshape(Data{i},1,numel(Data{i}))) > 2^i-1
		% Sortir un code d'erreur
		budget_data = -i;
		break;
	end	
	% Calcul de la quantité de données dans cette cellule
	budget_data = budget_data + numel(Data{i}) * i;	
end