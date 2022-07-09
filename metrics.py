from ast import List
import numpy as np
from path import Path2D


class RadarGenerator:

    def __init__(self, paths: List(Path2D), sigma_pos, sigma_vel):
        self.paths = paths
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel

    def make_measurement_series(self, probes):
        scans = []
        for probe in probes:
            measurements = []
            for path in self.paths:
                # TODO add measurement noise model
                pt = probe[0]
                px = probe[1]
                py = probe[2]
                if pt <= path.t_min or pt >= path.t_max:
                    continue
                x = path.pos(pt)[0] + np.random.normal(0, self.sigma_pos)
                y = path.pos(pt)[1] + np.random.normal(0, self.sigma_pos)

                r = np.array([px-x, py-y])
                vr = r.dot(path.vel(pt)) / np.linalg.norm(r) + \
                    np.random.normal(0, self.sigma_vel)
                measurements.append(Measurement(x-probe[1], y-probe[2], vr))
            scans.append(Scan(t=probe[0], measurements=measurements))

        return scans


class Measurement:

    def __init__(self, x, y, vr):
        self.x = x
        self.y = y
        self.vr = vr


class Scan:

    def __init__(self, t: float, measurements: List(Measurement)) -> None:
        self.measurements = measurements
        self.t = t
