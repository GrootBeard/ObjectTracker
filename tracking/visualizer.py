import matplotlib.pyplot as plt
import numpy as np

from tracking.filters.jpdaf import Track, Cluster
from tracking.util.metrics import TrackLog, LogEntry, LogEntryType

class TrackVisualizer:
    __slots__ = ("loggers", "logger_index", "actuals", "considered_measurements", "redline", "considered_clutter", "predictions")

    def __init__(self, logger_index: int = 0):
        self.logger_index = logger_index

    def initialize(self, tracks: list[Track]):
        self.loggers = [t._loggers[self.logger_index] for t in tracks] 

        redline = [log.filter_by_type(LogEntryType.ANY) for log in self.loggers] 
        self.redline = [np.array([entry.x for entry in track]) for track in redline]

        self._init_updates()
        self._init_predictions()

    def render(self, plot: plt.Axes, track_indices: list[int], epochs: list[int]):
        # TODO: implement epochs
        # TODO: check if track_indices, epochs are in bounds
        for track in track_indices:
            plot.plot(self.redline[track][:, 0], self.redline[track][:,1], color='red', linewidth=0.3)
            plot.scatter(self.actuals[track][:, 0], self.actuals[track][:, 1], color='blue', s=5)
            plot.scatter(self.predictions[track][:, 0], self.predictions[track][:,1], color='purple', s=3)
            for entry in self.considered_measurements[track]:
                for mt in entry:
                    plot.scatter(mt[0], mt[1], color="green", s=5)
            for entry in self.considered_clutter[track]:
                for mt in entry:
                    plot.scatter(mt[0], mt[1], color="gray", s=3)
        plt.show()

    def _init_updates(self):
        updates = [log.filter_by_type(LogEntryType.UPDATE) for log in self.loggers] 
        self.actuals = [np.array([entry.x for entry in track]) for track in updates]
        self.considered_measurements = [[[mt.z for mt in entry.considered_measurements if not mt.is_clutter] for entry in track] for track in updates]
        self.considered_clutter = [[[mt.z for mt in entry.considered_measurements if mt.is_clutter] for entry in track] for track in updates]

    def _init_predictions(self):
        predictions = [log.filter_by_type(LogEntryType.PREDICTION) for log in self.loggers] 
        self.predictions = [np.array([entry.x for entry in track]) for track in predictions]
