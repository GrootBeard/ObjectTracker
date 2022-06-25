from typing import Tuple, Type
import numpy as np

from interpolator import Interpolator, LinearInterpolator


class Path2D:

    # nodes describe position (x,y) at discretized, evenly-space time intervals
    #       eg. q_k = q(t_k), k = 0, 1, ..., N
    #                 t_k = k * t_max/N

    # interpolator is a function that returns a position (x,y) given a time t
    # and the set of nodes describing the path

    def __init__(self, tmax: float, nodes: np.ndarray, interpolatorClass: Interpolator.__class__) -> None:
        if (nodes.shape[1]) != 2:
            raise PathWrongDimensionException(2, nodes.shape[1])

        self.t_max = tmax
        self.nodes = nodes
        self.N = len(self.nodes)

        self.interpolator = interpolatorClass(self.t_max, self.N)

        print(self.interpolator)

    def pos(self, t) -> np.ndarray:
        x = self.interpolator.interpolate(t, self.nodes[:, 0])
        y = self.interpolator.interpolate(t, self.nodes[:, 1])
        return np.array([x, y])

    def vel(self, t) -> np.ndarray:
        k = np.min([int(np.floor(t * self.N / self.t_max)), self.N - 1])
        kp = np.min([self.N-1, k+1])

        dt = self.t_max/self.N

        vx = (self.nodes[kp, 0] - self.nodes[k, 0])/dt
        vy = (self.nodes[kp, 1] - self.nodes[k, 1])/dt

        return np.array([vx, vy])

    def __str__(self) -> str:
        return 'Path [0, %2.2f] interpolated by %s' % (self.t_max, self.interpolator.__str__())


class PathWrongDimensionException(Exception):

    def __init__(self, reqDim: int, givenDim: int) -> None:
        super().__init__('Wrong path dimension given. Given dim is %i. Required dim is %i' %
                         (givenDim, reqDim))
