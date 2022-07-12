from abc import ABC, abstractmethod

import numpy as np
from scipy import interpolate


class NodeCollection:
    __slots__ = ("positions", "params", "dim", "len")

    def __init__(self, positions: np.ndarray, params: np.array) -> None:
        self.positions = positions
        self.params = params
        self.dim = self.positions.shape[1]
        self.len = self.positions.shape[0]

    @property
    def t_max(self):
        return self.params[-1]

    @property
    def t_min(self):
        return self.params[0]

    def xdim(self, dim: int):
        pass


class Interpolator(ABC):
    __slots__ = ("t_min", "t_max")

    def prepare(self, nodes: NodeCollection) -> None:
        self.t_min = nodes.t_min
        self.t_max = nodes.t_max

    @abstractmethod
    def interpolate(self, t: float) -> np.array:
        pass

    @abstractmethod
    def derivative(self, t: float, n: int = 1) -> np.array:
        pass


class LinearInterpolator(Interpolator):
    __slots__ = ("interp_funcs")

    def interpolate(self, t: float) -> np.array:
        return np.array([f(t) for f in self.interp_funcs])

    def prepare(self, nodes: NodeCollection) -> None:
        super().prepare(nodes)
        self.interp_funcs = []

        for dim in range(len(nodes.positions[0])):
            vals = nodes.positions[:, int(dim)]

            f = interpolate.UnivariateSpline(nodes.params, vals, k=1, s=1)

            self.interp_funcs.append(f)

    def derivative(self, t: float, n: int = 1) -> np.array:
        return np.array([f.derivatives(t)[n] for f in self.interp_funcs])


class SplineInterpolator(Interpolator):
    __slots__ = ("tck")

    def prepare(self, nodes: NodeCollection) -> None:
        super().prepare(nodes)
        self.tck, _ = interpolate.splprep(
            nodes.positions.T, u=nodes.params, k=3, s=0)

    def interpolate(self, t: float) -> np.array:
        return interpolate.splev(t, self.tck)

    def derivative(self, t: float, n: int = 1) -> np.array:
        return interpolate.splev(t, self.tck, der=n)
