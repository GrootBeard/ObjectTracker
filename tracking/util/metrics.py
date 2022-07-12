from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from tracking.util.path import Path2D


class RadarGenerator:

    def __init__(self, paths: list[Path2D], sigma_pos, sigma_vel):
        self.paths = paths
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel

    def make_scans_series(self, probes):
        scans = []
        for probe in probes:
            measurements = []
            for path in self.paths:
                # TODO add measurement noise model
                pt = probe[0]
                # px = probe[1]
                # py = probe[2]
                if pt <= path.t_min or pt >= path.t_max:
                    continue
                x = path.pos(pt)[0] + np.random.normal(0, self.sigma_pos)
                y = path.pos(pt)[1] + np.random.normal(0, self.sigma_pos)

                # r = np.array([px-x, py-y])
                # vr = r.dot(path.vel(pt)) / np.linalg.norm(r) + \
                #     np.random.normal(0, self.sigma_vel)
                measurements.append(Measurement(
                    np.array([x-probe[1], y-probe[2]])))
            if len(measurements) > 0:
                scans.append(Scan(time=probe[0], measurements=measurements))

        return scans


@dataclass(slots=True)
class Measurement:
    z: np.array
    is_clutter: bool = False


class Scan:
    __slots__ = ("_measurements", "time")

    def __init__(self, time: float, measurements: list[Measurement]) -> None:
        # measurement with index 0 is reserved for 'no-measurement'
        self._measurements = {i+1: mt for i, mt in enumerate(measurements)}
        self.time = time

    @property
    def measurements(self):
        return self._measurements.items()

    @property
    def measurements_list(self):
        return list(self._measurements.values())

    @property
    def measurements_indices(self):
        return list(self._measurements.keys())

    @property
    def values(self):
        return [mt.z for mt in self.measurements_list]


class LogEntryType(Enum):
    PREDICTION = auto()
    MEASUREMENT = auto()


@dataclass(slots=True)
class LogEntry:
    x: np.array
    P: np.array
    time: float
    type: LogEntryType
    metadata: dict = Optional[dict]


class TrackLog:
    __slots__ = ("x_model", "P_model", "entries")

    def __init__(self, x_model: np.ndarray, P_model: np.ndarray) -> None:
        self.x_model = x_model
        self.P_model = P_model
        self.entries = []

    def add_entry(self, x, P, time: float, type: LogEntryType) -> None:
        xval = self.x_model.dot(x)
        Pval = self.P_model.dot(P)

        self.entries.append(LogEntry(x=xval, P=Pval, time=time, type=type))

    def flatten(self):
        return (self.flatten_x(), self.flatten_P(), self.flatten_time())

    def flatten_x(self) -> np.ndarray:
        return np.array([[v.x[col] for v in self.entries] for col, _ in enumerate(self.entries[0].x)])

    def flatten_P(self) -> np.ndarray:
        return np.array([[v.P[col] for v in self.entries] for col, _ in enumerate(self.entries[0].P)])

    def flatten_time(self) -> np.array:
        return np.array([e.time for e in self.entries])
