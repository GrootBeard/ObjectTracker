
import numpy as np
from tracking.util.interpolator import Interpolator, NodeCollection, LinearInterpolator


class Path2D:

    # nodes describe position (x,y) at discretized, evenly-space time intervals
    #       eg. q_k = q(t_k), k = 0, 1, ..., N
    #                 t_k = k * t_max/N

    # interpolator is a function that returns a position (x,y) given a time t
    # and the set of nodes describing the path

    def __init__(self, nodes: NodeCollection, interpolatorClass: Interpolator.__class__, uid:int=-1) -> None:
        self.interpolator = interpolatorClass()
        self.interpolator.prepare(nodes)
        self.uid = uid

    def pos(self, t) -> np.ndarray:
        return self.interpolator.interpolate(t)

    def vel(self, t) -> np.ndarray:
        return self.interpolator.derivative(t)

    @property
    def t_min(self):
        return self.interpolator.t_min

    @property
    def t_max(self):
        return self.interpolator.t_max

    def __str__(self) -> str:
        return 'Path [0, %2.2f] interpolated by %s' % (self.t_max, self.interpolator.__str__())


class PathFactory:
    def __init__(self):
        self.uid = 0

    def create(self, nodes: list[NodeCollection], intp:Interpolator=LinearInterpolator.__class__):
        paths = []
        for n in nodes:
            paths.append(Path2D(n, interpolatorClass=intp, uid=self.uid))
            self.uid += 1
        return paths


class PathWrongDimensionException(Exception):

    def __init__(self, reqDim: int, givenDim: int) -> None:
        super().__init__('Wrong path dimension given. Given dim is %i. Required dim is %i' %
                         (givenDim, reqDim))
