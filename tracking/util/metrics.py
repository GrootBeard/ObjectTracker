from dataclasses import dataclass 
from enum import Enum, auto
from typing import Optional

import numpy as np
from tracking.util.path import Path2D


class PoissonClutter:
    __slots__ = ("rng", "x_bounds", "y_bounds", "clutter_density", "width", "height")

    def __init__(self, bounds, clutter_density, seed: int = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.x_bounds = (bounds[0], bounds[1])
        self.y_bounds = (bounds[2], bounds[3])
        self.clutter_density = clutter_density

        self.width = self.x_bounds[1] - self. x_bounds[0]
        self.height = self.y_bounds[1] - self. y_bounds[0]
    
    def generate_clutter(self):
        m_k = self.rng.poisson(self.width * self.height * self.clutter_density)
        return self.rng.random((m_k, 2)) * np.array([self.width, self.height]) + np.array([self.x_bounds[0], self.y_bounds[0]])
         

class RadarGenerator:

    def __init__(self, paths: list[Path2D], sigma_pos, sigma_vel):
        self.paths = paths
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel

    def make_scans_series(self, probes, clutter_generator = None):
        scans = []
        for scan_id, probe in enumerate(probes):
            measurements = []
            for mt_id, path in enumerate(self.paths):
                pt = probe[0]
                if pt <= path.t_min or pt >= path.t_max:
                    continue
                x = path.pos(pt)[0] + np.random.normal(0, self.sigma_pos)
                y = path.pos(pt)[1] + np.random.normal(0, self.sigma_pos)

                measurements.append(Measurement(
                    np.array([x-probe[1], y-probe[2]]), scan_id=scan_id, mt_id=mt_id+1, origin_id=path.uid))

            if clutter_generator:
                clutter_id = len(measurements) + 1
                for clutter in clutter_generator.generate_clutter():
                    measurements.append(Measurement(
                        np.array([clutter[0], clutter[1]]), scan_id=scan_id, mt_id=clutter_id, origin_id=-1, is_clutter=True))
                    clutter_id += 1

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
    __slots__ = ("x_model", "P_model", "entries", "epochs")

    def __init__(self, x_model: np.ndarray, P_model: np.ndarray) -> None:
        self.x_model = x_model
        self.P_model = P_model
        self.entries = []
        self.epochs = [[]]

    def add_entry(self, x, P, time: float, type: LogEntryType, considered_measurements: list[Measurement]) -> None:
        xval = self.x_model.dot(x)
        Pval = self.P_model.dot(P)

        self.epochs[-1].append(len(self.entries)) 
        if type is LogEntryType.UPDATE:
            self.epochs.append([])

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
    
