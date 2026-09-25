import random
from collections import Counter

import numpy as np

def _initialize_representatives_random(space, n_representative):
    _representatives = random.choices(space, k=n_representative)
    representatives = {}
    for i in range(n_representative):
        representatives[i] = _representatives[i]
    return representatives

def _initialize_representatives_distance(space, n_representative, distance_threshold=0,
                                          decay=10, min_threshold=0.0):
    space = np.asarray(space, dtype=float)
    n_points = space.shape[0]
    if n_representative > n_points:
        raise ValueError("Cannot select more representatives than points available")

    remaining_idx = list(range(n_points))

    representatives = {}
    found = 0

    first_idx = random.choice(remaining_idx)
    representatives[found] = space[first_idx]
    remaining_idx.remove(first_idx)
    found += 1

    while len(representatives) < n_representative:
        reps_array = np.array(list(representatives.values()))
        candidates = space[remaining_idx]

        dists = np.linalg.norm(candidates[:, None, :] - reps_array[None, :, :], axis=2)
        min_dist_to_reps = dists.min(axis=1)

        acceptance = np.argmax(min_dist_to_reps)
        best_distance = min_dist_to_reps[acceptance]

        if best_distance >= distance_threshold:
            chosen_idx = remaining_idx[acceptance]
            representatives[found] = space[chosen_idx]
            remaining_idx.remove(chosen_idx)
            found += 1
        else:
            distance_threshold = max(distance_threshold - decay, min_threshold)

    return representatives

def _calculate_cluster(space,representatives):
    clusters = {}

    for i in range(len(representatives)):
        clusters[i] = []

    space_array = np.array(space)
    reps_array = np.array(list(representatives.values()))

    sq_space = np.sum(space_array ** 2, axis=1)[:, None]
    sq_reps = np.sum(reps_array ** 2, axis=1)[None, :]
    cross = space_array @ reps_array.T
    dist_sq = sq_space + sq_reps - 2 * cross
    labels = np.argmin(dist_sq, axis=1)
    for i in range(len(labels)):
        clusters[labels[i]].append(space[i])
    return clusters

def _find_representative(space, n_representative, number_of_iterations):
    representatives = _initialize_representatives_distance(space, n_representative, 1000)

    for i in range(number_of_iterations):
        clusters = _calculate_cluster(space, representatives)

        # Recalculate representatives
        for i in range(len(representatives)):
            representatives[i] = np.average(clusters[i], axis=0)

    return representatives

def encode(Is, vector_size, n_bits):
    # Divide Is into space (X number of vector[n_features])
    Ir = Is.copy()
    n_representative = 2**n_bits
    space = []
    vector_side_row = np.round(vector_size**0.5).astype(int)
    vector_side_col = vector_size // vector_side_row

    if Ir.shape[0] % vector_side_row:
        pad = vector_side_row - Ir.shape[0] % vector_side_row
        Ir = np.vstack((Ir, np.tile(Ir[[-1], :], (pad, 1))))
    if Ir.shape[1] % vector_side_col:
        pad = vector_side_col - Ir.shape[1] % vector_side_col
        Ir = np.hstack((Ir, np.tile(Ir[:, [-1]], (1, pad))))

    for i in range(Ir.shape[0] // vector_side_row):
        for j in range(Ir.shape[1] // vector_side_col):
            vector = Ir[i * vector_side_row : i * vector_side_row + vector_side_row,
                        j * vector_side_col : j * vector_side_col + vector_side_col]

            space.append(vector.flatten())

    # Find representative
    representatives = _find_representative(space, n_representative, 10)

    # encode
    Ie = np.zeros((Is.shape[0] // vector_side_row + 1, Is.shape[1] // vector_side_col + 1))
    for i in range(Ir.shape[0] // vector_side_row):
        for j in range(Ir.shape[1] // vector_side_col):
            vector = Ir[i * vector_side_row: i * vector_side_row + vector_side_row,
                        j * vector_side_col: j * vector_side_col + vector_side_col].flatten()

            distances = [np.linalg.norm(vector - representatives[i]) for i in range(len(representatives))]
            label = distances.index(min(distances))

            Ie[i][j] = label

    counter = Counter()
    for x in range(Ie.shape[0]):
        counter += Counter(Ie[x])
    print("nbits", np.log2(len(counter)))
    print(counter)
    return Ie, representatives

def decode(Is, vector_size, representatives, row, col):
    vector_side_row = np.round(vector_size**0.5).astype(int)
    vector_side_col = vector_size // vector_side_row
    Ir = np.zeros((Is.shape[0] * vector_side_row, Is.shape[1] * vector_side_col))

    for i in range(Ir.shape[0] // vector_side_row):
        for j in range(Ir.shape[1] // vector_side_col):
            vector = representatives[Is[i, j]].reshape(vector_side_row, vector_side_col)
            Ir[i * vector_side_row : i * vector_side_row + vector_side_row,
               j * vector_side_col : j * vector_side_col + vector_side_col] = vector

    return Ir[:row, :col]

def find_bits_per_pixel(Is, vector_size, n_bit_vector, n_pixels):
    n_representative = 2**n_bit_vector

    n_bits_metadata = vector_size * 8 * n_representative
    n_bits_image = Is.shape[0] * Is.shape[1] * n_bit_vector

    return (n_bits_metadata + n_bits_image) / n_pixels