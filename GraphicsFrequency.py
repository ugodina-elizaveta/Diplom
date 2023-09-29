import matplotlib.pyplot as plt
import numpy as np


class GraphicsFrequency:
    def __init__(self, V, l):
        self.V = V
        self.l = l

    def finding_areas_of_oscillation(self):
        self.x = np.arange(0, sum(self.l) + sum(self.l) / len(self.l), sum(self.l) / len(self.l)-1)
        return self.x

    # Построение графиков форм колебаний для каждой частоты
    def plotting(self, i):
        self.finding_areas_of_oscillation()
        plt.plot(self.x[:4], self.V, label=f'{i} форма колебаний')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('Длина L, м')
        plt.ylabel('Форма колебаний, м')