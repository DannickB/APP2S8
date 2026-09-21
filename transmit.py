def transmit(Data):
    # Initialisation
    budget_data = 0
    # Pour chaque cellule
    for i in range(1, max(Data.keys()) + 1):
        if i not in Data:
            continue
        cell = Data[i]
        flat = [val for row in cell for val in row] if isinstance(cell[0], list) else list(cell)
        # Si la donnee est plus petite que la gamme dynamique
        if min(flat) < 0:
            # Sortir un code d'erreur
            budget_data = -i
            break
        # Si la donnee est plus grande que la gamme dynamique
        if max(flat) > 2 ** i - 1:
            # Sortir un code d'erreur
            budget_data = -i
            break
        # Calcul de la quantite de donnees dans cette cellule
        budget_data = budget_data + len(flat) * i
    return budget_data
