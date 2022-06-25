from abc import abstractmethod
from typing import Tuple, overload
import numpy as np


class Interpolator:

    def __init__(self, xmax: float, N: int) -> None:
        self.xmax = xmax
        self.N = N

    @abstractmethod
    def interpolate(self, x: float, nodes) -> float:
        pass


class LinearInterpolator(Interpolator):

    def interpolate(self, x: float, nodes) -> float:
        k = np.min([int(np.floor(x * self.N / self.xmax)), self.N - 1])

        x0 = k * self.xmax/self.N
        kp = np.min([self.N-1, k+1])
        return nodes[k] + self.N*(x-x0)/self.xmax * (nodes[kp] - nodes[k])

    def __str__(self):
        return 'LinearInterpolator %3.1f' % self.xmax
