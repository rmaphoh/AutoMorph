import matplotlib.pyplot as plt
import numpy as np
from unittest import TestCase
from retipy import tortuosity_measures as tm
from retipy.retina import Retina


class BenchmarkTortuosity(TestCase):
    _line: np.ndarray
    _points = 200
    _measures = 6

    def setUp(self):
        f = 1  # the frequency of the signal

        x = np.arange(self._points)  # the points on the x axis for plotting
        # compute the value (amplitude) of the sin wave at the for each sample

        self._line = np.empty([5, 2, self._points])
        self._line[0, 0] = x
        self._line[0, 1] = 0
        for i in range(1, 5):
            self._line[i, 0] = x
            self._line[i, 1] =[np.sin(2 * np.pi * f*(i*3) * (j / self._points)) for j in x]

    def test_view(self):
        plt.ylim(-10, 10)
        # for i in range(1, 5):
        i = 4
        plt.plot(self._line[i, 0], self._line[i, 1])
        plt.show()

    def test_benchmark_density(self):
        outputs = np.empty([self._measures, 5, 2])
        for i in range(0, 5):
            outputs[:, i, 0] = i
            outputs[0, i, 1] = tm.tortuosity_density(self._line[i, 0], self._line[i, 1])
            outputs[1, i, 1] = tm.linear_regression_tortuosity(self._line[i, 0], self._line[i, 1], 10)
            outputs[2, i, 1] = tm.distance_inflection_count_tortuosity(self._line[i, 0], self._line[i, 1])
            outputs[3, i, 1] = tm.distance_measure_tortuosity(self._line[i, 0], self._line[i, 1])
            outputs[4, i, 1] = tm.squared_curvature_tortuosity(self._line[i, 0], self._line[i, 1])
            line = Retina(None, "line_{}.png".format(i))
            line.threshold_image()
            outputs[5, i, 1] = tm.fractal_tortuosity(line)

        #  ## UNCOMMENT THIS TO NORMALISE DATA ## #
        # # get max and min values to normalize
        # top_values = np.empty([self._measures, 2])
        # top_values[:, 0] = 999999
        # top_values[:, 1] = -999999
        # for t in range(0, self._measures):
        #     for i in range(0, 5):
        #         if top_values[t, 0] > outputs[t, i, 1]:
        #             top_values[t, 0] = outputs[t, i, 1]
        #         if top_values[t, 1] < outputs[t, i, 1]:
        #             top_values[t, 1] = outputs[t, i, 1]
        # # normalisation here
        # for t in range(0, self._measures):
        #     for i in range(0, 5):
        #         outputs[t, i, 1] = (outputs[t, i, 1] - top_values[t, 0]) / (top_values[t, 1] - top_values[t, 0])

        print(outputs[1, :, 1])
        plt.plot(outputs[0, :, 0], outputs[0, :, 1], label="density")
        plt.plot(outputs[1, :, 0], outputs[1, :, 1], label="linear regression")
        plt.plot(outputs[2, :, 0], outputs[2, :, 1], label="inflection count")
        plt.plot(outputs[3, :, 0], outputs[3, :, 1], label="distance measure")
        plt.plot(outputs[4, :, 0], outputs[4, :, 1], label="squared curvature")
        plt.plot(outputs[5, :, 0], outputs[5, :, 1], label="fractal")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.xlabel("line")
