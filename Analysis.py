import scipy.sparse.linalg
import numpy as np


class Analysis:
    def __init__(self, K, M):
        self.K = K
        self.M = M
        self.n = []
        # Решение частичной проблемы собственных значений
        self.X, self.D = scipy.sparse.linalg.eigs(self.K, 4, self.M)

    def formation_of_natural_frequencies_and_shapes(self):
        for i in range(0, 4):
            W = sorted(np.sqrt(abs(self.X)))   # i-ая собст. частота (рад/с)
            print(f'{i+1}-ая собственная частота(рад/с):{W[i]}')

        # Создаем вектор сортировки для собственных форм
        w = np.sqrt(abs(self.X))
        for k in range(0, len(w)):
            for c in range(0, len(W)):
                if w[k] == W[c]:
                    self.n.insert(c, k)

        self.D = self.D[:, self.n]
        V1 = self.D[:, 0]      # вектор первой формы колебаний
        V2 = self.D[:, 1]      # -\\- второй
        V3 = self.D[:, 2]      # -\\- третьей
        V4 = self.D[:, 3]
        return V1, V2, V3, V4
