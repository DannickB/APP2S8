import numpy as np

class VectorialQuantifier:
    def __init__(self):
        ...

    def _initialize_representatives(self, space, n_representative, distance_threshold):
        found = 0
        representatives = {}
        while found < n_representative:
            if found == 0 :
                representatives[0] = space[0]
                found += 1
                continue

            for vector in space:

                if found >= n_representative:
                    break

                distances = [np.linalg.norm(vector - representatives[i]) for i in range(len(representatives))]
                if min(distances) > distance_threshold:
                    representatives[found] = vector
                    found += 1

            distance_threshold -= 5

        return representatives

    def _calculate_cluster(self, space,representatives):
        clusters = {}

        for i in range(len(representatives)):
            clusters[i] = []

        for data in space:

            euc_dist = []
            for j in range(len(representatives)):
                euc_dist.append(np.linalg.norm(data - representatives[j]))

            clusters[euc_dist.index(min(euc_dist))].append(data)
        return clusters

    def _find_representative(self, space, n_representative, number_of_iterations):
        representatives = self._initialize_representatives(space, n_representative, 1000)

        for i in range(number_of_iterations):
            clusters = self._calculate_cluster(space, representatives)

            # Recalculate representatives
            for i in range(len(representatives)):
                representatives[i] = np.average(clusters[i], axis=0)

        return representatives

    def encode(self, Is, vector_size, n_bits, ):
        # Divide Is into space (X number of vector[n_features])
        n_representative = n_bits**2
        space = []
        vector_side = np.round(vector_size**0.5).astype(int)
        for i in range(Is.shape[0] // vector_side):
            for j in range(Is.shape[1] // vector_side):
                vector = Is[i * vector_side : i * vector_side + vector_side,
                            j * vector_side : j * vector_side + vector_side]

                space.append(vector.flatten())

        # Find representative
        representatives = self._find_representative(space, n_representative, 10)

        # encode
        Ir = np.zeros((Is.shape[0] // vector_side, Is.shape[1] // vector_side))
        for i in range(Is.shape[0] // vector_side):
            for j in range(Is.shape[1] // vector_side):
                vector = Is[i * vector_side: i * vector_side + vector_side,
                            j * vector_side: j * vector_side + vector_side].flatten()

                distances = [np.linalg.norm(vector - representatives[i]) for i in range(len(representatives))]
                label = distances.index(min(distances))

                Ir[i][j] = label

        return Ir, representatives

    def decode(self, Is, vector_size, representatives):
        vector_side = np.round(vector_size**0.5).astype(int)
        Ir = np.zeros((Is.shape[0] * vector_side, Is.shape[1] * vector_side))

        for i in range(Ir.shape[0] // vector_side):
            for j in range(Ir.shape[1] // vector_side):
                vector = representatives[Is[i, j]].reshape(vector_side, vector_side)
                Ir[i * vector_side : i * vector_side + vector_side,
                   j * vector_side : j * vector_side + vector_side] = vector

        return Ir
