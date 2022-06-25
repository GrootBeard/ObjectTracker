import numpy as np
from pyparsing import List
from path import Path2D


class RadarGenerator:

    def __init__(self, paths: List[Path2D], sigma_pos, sigma_vel):
        self.paths = paths
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel

    def make_measurement_series(self, probes):
        measurements = []
        for probe in probes:
            for path in self.paths:
                # TODO add measurement noise
                pt = probe[0]
                px = probe[1]
                py = probe[2]
                x, y = path.pos(pt) + np.random.normal(0, self.sigma_pos, 2)

                r = np.array([px-x, py-y])
                vr = r.dot(path.vel(pt)) / np.linalg.norm(r) + \
                    np.random.normal(0, self.sigma_vel)
                measurements.append(Measurement(
                    probe[0], x-probe[1], y-probe[2], vr))

        return measurements


class Measurement:

    def __init__(self, t, x, y, vr):
        self.t = t
        self.x = x
        self.y = y
        self.vr = vr
