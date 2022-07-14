from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from tracking.util.path import Path2D


class ClutterModel:
    __slots__ = ("x_bounds", "y_bounds", "object_density",
                 "random_density", "clutter_objects")

    def __init__(self, bounds, object_density, random_density) -> None:
        self.x_bounds = (bounds[0], bounds[1])
        self.y_bounds = (bounds[2], bounds[3])
        self.object_density = object_density
        self.random_density = random_density
        area = (self.x_bounds[1] - self. x_bounds[0]) * \
            (self.y_bounds[1] - self. y_bounds[0])
        N_objects = self.object_density * area
        N_random = self.random_density * area
        # for n in range(N_objects):
        # self.clutter_objects.append(np.random.random(size=()))

    @property
    def clutter(self):
        return None


class RadarGenerator:

    def __init__(self, paths: list[Path2D], sigma_pos, sigma_vel):
        self.paths = paths
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel

    def make_scans_series(self, probes):
        scans = []
        for scan_id, probe in enumerate(probes):
            measurements = []
            for mt_id, path in enumerate(self.paths):
                # TODO add measurement noise model
                pt = probe[0]
                # px = probe[1]
                # py = probe[2]
                if pt <= path.t_min or pt >= path.t_max:
                    continue
                x = path.pos(pt)[0] + np.random.normal(0, self.sigma_pos)
                y = path.pos(pt)[1] + np.random.normal(0, self.sigma_pos)

                # vr = r.dot(path.vel(pt)) / np.linalg.norm(r) + \
                #     np.random.normal(0, self.sigma_vel)
                measurements.append(Measurement(
                    np.array([x-probe[1], y-probe[2]]), scan_id=scan_id, mt_id=mt_id+1, origin_id=path.uid))
            if measurements:
                scans.append(Scan(time=probe[0], measurements=measurements, scan_id=scan_id))

        return scans


@dataclass(slots=True)
class Measurement:
    z: np.array
    scan_id: int
    mt_id: int
    origin_id: int
    is_clutter: bool = False


class Scan:
    __slots__ = ("_measurements", "time", "scan_id")

    def __init__(self, time: float, measurements: list[Measurement], scan_id:int) -> None:
        # measurement with index 0 is reserved for 'no-measurement'
        self._measurements = {i+1: mt for i, mt in enumerate(measurements)}
        self.time = time
        self.scan_id = scan_id

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
    UPDATE = auto()
    ANY = auto()


@dataclass(slots=True)
class LogEntry:
    x: np.array
    P: np.array
    time: float
    type: LogEntryType
    considered_measurements: list[Measurement] 
    metadata: dict = Optional[dict]

    def __repr__(self):
        return f'LogEntry time: {self.time}, x-value: {self.x}'


class TrackLog:
    __slots__ = ("x_model", "P_model", "entries")

    def __init__(self, x_model: np.ndarray, P_model: np.ndarray) -> None:
        self.x_model = x_model
        self.P_model = P_model
        self.entries = []

    def add_entry(self, x, P, time: float, type: LogEntryType, considered_measurements: list[Measurement]) -> None:
        xval = self.x_model.dot(x)
        Pval = self.P_model.dot(P)

        self.entries.append(LogEntry(x=xval, P=Pval, time=time, type=type, considered_measurements=considered_measurements))

    def filter_by_type(self, type: LogEntryType | list[LogEntryType]= LogEntryType.ANY) -> list[LogEntry]:
        if isinstance(type, LogEntryType):
            type_filter = [type]
        if LogEntryType.ANY in type_filter:
            return self.entries
        return [entry for entry in self.entries if entry.type in type_filter]

    def flatten(self, type_filter: LogEntryType | list[LogEntryType]= LogEntryType.ANY):
        return (self.flatten_x(type_filter), self.flatten_P(type_filter), self.flatten_time(type_filter))

    def flatten_x(self, type_filter: LogEntryType | list[LogEntryType] = LogEntryType.ANY) -> np.ndarray:
        filtered_entries = self.filter_by_type(type_filter)
        return np.array([[v.x[col] for v in filtered_entries] for col, _ in enumerate(filtered_entries[0].x)])

    def flatten_P(self, type_filter: LogEntryType | list[LogEntryType]= LogEntryType.ANY) -> np.ndarray:
        filtered_entries = self.filter_by_type(type_filter)
        return np.array([[v.P[col] for v in filtered_entries] for col, _ in enumerate(filtered_entries[0].P)])

    def flatten_time(self, type_filter: LogEntryType | list[LogEntryType]= LogEntryType.ANY) -> np.array:
        filtered_entries = self.filter_by_type(type_filter)
        return np.array([e.time for e in filtered_entries])
