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
    """
    Farthest-point selection with a relaxing minimum-distance threshold.

    Instead of recomputing a full (remaining-candidates x all-chosen-reps)
    distance matrix from scratch every time a representative is added --
    O(n_representative^2 * n_points) overall, and the dominant cost at
    n_representative=256 -- maintain a running "distance to the nearest
    chosen representative so far" array and update it in O(n_points) each
    time a new representative is added, via a single new-point-to-all-points
    norm() call. That brings the total down to O(n_representative * n_points).
    """
    space = np.asarray(space, dtype=float)
    n_points = space.shape[0]
    if n_representative > n_points:
        raise ValueError("Cannot select more representatives than points available")

    first_idx = random.randrange(n_points)
    chosen_idx = [first_idx]

    min_dist = np.linalg.norm(space - space[first_idx], axis=1)
    min_dist[first_idx] = -np.inf  # never reselect an already-chosen point

    while len(chosen_idx) < n_representative:
        candidate_idx = np.argmax(min_dist)
        best_distance = min_dist[candidate_idx]

        if best_distance >= distance_threshold:
            chosen_idx.append(candidate_idx)
            new_dist = np.linalg.norm(space - space[candidate_idx], axis=1)
            min_dist = np.minimum(min_dist, new_dist)
            min_dist[candidate_idx] = -np.inf
        else:
            distance_threshold = max(distance_threshold - decay, min_threshold)

    return {i: space[idx] for i, idx in enumerate(chosen_idx)}


def _pairwise_sq_dist(a, b):
    """
    Vectorized squared-distance matrix (len(a), len(b)) via the
    ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y identity, so the heavy lifting is
    one BLAS matrix multiply instead of one norm() call per (point, rep) pair.
    """
    sq_a = np.sum(a ** 2, axis=1)[:, None]
    sq_b = np.sum(b ** 2, axis=1)[None, :]
    cross = a @ b.T
    return sq_a + sq_b - 2 * cross


def _calculate_cluster(space, representatives):
    clusters = {i: [] for i in range(len(representatives))}

    space_array = np.array(space)
    reps_array = np.array(list(representatives.values()))

    dist_sq = _pairwise_sq_dist(space_array, reps_array)
    labels = np.argmin(dist_sq, axis=1)
    for i in range(len(labels)):
        clusters[labels[i]].append(space[i])
    return clusters


def _find_representative(space, n_representative, number_of_iterations):
    representatives = _initialize_representatives_distance(space, n_representative, 1000)

    for _ in range(number_of_iterations):
        clusters = _calculate_cluster(space, representatives)
        empty_ids = [i for i in range(len(representatives)) if len(clusters[i]) == 0]

        for i in range(len(representatives)):
            if i not in empty_ids:
                representatives[i] = np.average(clusters[i], axis=0)

        if empty_ids:
            pool = []
            for j in range(len(representatives)):
                if j in empty_ids or len(clusters[j]) < 2:
                    continue
                pts = np.array(clusters[j])
                d = np.linalg.norm(pts - representatives[j], axis=1)
                worst = np.argmax(d)
                pool.append((d[worst], pts[worst]))
            pool.sort(key=lambda t: -t[0])
            for k, i in enumerate(empty_ids):
                if k < len(pool):
                    representatives[i] = pool[k][1]
                # else: fewer distinct clusters of data exist than
                # n_representative asks for -- nothing left to hand out.

    return representatives


def encode(Is, vector_size, n_bits):
    # Divide Is into space (X number of vector[n_features])
    Ir = Is.copy()
    n_representative = 2 ** n_bits
    space = []
    vector_side_row = np.round(vector_size ** 0.5).astype(int)
    vector_side_col = vector_size // vector_side_row

    if Ir.shape[0] % vector_side_row:
        pad = vector_side_row - Ir.shape[0] % vector_side_row
        Ir = np.vstack((Ir, np.tile(Ir[[-1], :], (pad, 1))))
    if Ir.shape[1] % vector_side_col:
        pad = vector_side_col - Ir.shape[1] % vector_side_col
        Ir = np.hstack((Ir, np.tile(Ir[:, [-1]], (1, pad))))

    for i in range(Ir.shape[0] // vector_side_row):
        for j in range(Ir.shape[1] // vector_side_col):
            vector = Ir[i * vector_side_row: i * vector_side_row + vector_side_row,
                        j * vector_side_col: j * vector_side_col + vector_side_col]
            space.append(vector.flatten())

    # Find representative
    representatives = _find_representative(space, n_representative, 10)

    # encode -- vectorized labeling pass, same fix as _calculate_cluster,
    # since this had the identical one-point-at-a-time norm() pattern
    space_array = np.array(space)
    reps_array = np.array([representatives[i] for i in range(len(representatives))])
    dist_sq = _pairwise_sq_dist(space_array, reps_array)
    labels = np.argmin(dist_sq, axis=1)

    n_row_blocks = Ir.shape[0] // vector_side_row
    n_col_blocks = Ir.shape[1] // vector_side_col
    # fixed: sized to exactly match the loop bounds (was Is.shape[...] + 1,
    # which left an extra unfilled row/col of zeros that silently inflated
    # label 0's frequency in the Counter below)
    Ie = labels.reshape(n_row_blocks, n_col_blocks).astype(float)

    counter = Counter(Ie.reshape(-1))
    print("nbits", np.log2(len(counter)))
    print(counter)
    return Ie, representatives


def decode(Is, vector_size, representatives, row, col):
    vector_side_row = np.round(vector_size ** 0.5).astype(int)
    vector_side_col = vector_size // vector_side_row
    Ir = np.zeros((Is.shape[0] * vector_side_row, Is.shape[1] * vector_side_col))

    for i in range(Ir.shape[0] // vector_side_row):
        for j in range(Ir.shape[1] // vector_side_col):
            vector = representatives[Is[i, j]].reshape(vector_side_row, vector_side_col)
            Ir[i * vector_side_row: i * vector_side_row + vector_side_row,
               j * vector_side_col: j * vector_side_col + vector_side_col] = vector

    return Ir[:row, :col]


def find_bits_per_pixel(Is, vector_size, n_bit_vector, n_pixels):
    n_representative = 2 ** n_bit_vector

    n_bits_metadata = vector_size * 8 * n_representative
    n_bits_image = Is.shape[0] * Is.shape[1] * n_bit_vector

    return (n_bits_metadata + n_bits_image) / n_pixels