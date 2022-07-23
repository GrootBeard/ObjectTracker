import matplotlib.pyplot as plt
import numpy as np

from tracking.filters.jipdaf import Cluster, Track
from tracking.util.metrics import LogEntry, LogEntryType, TrackLog


class TrackVisualizer:
    __slots__ = ("loggers", "logger_index", "actuals", "considered_measurements", "redline",
                 "considered_clutter", "predictions", "colors", "_actuals_buffer", "_prediction_buffer",
                 "_considered_mts_buffer", "_considered_clutter_buffer", "_is_initialized", "_buffered_epochs",
                 "red_lines_to_render")

    def __init__(self, logger_index: int = 0):
        self.logger_index = logger_index
        self._is_initialized = False

    def initialize(self, tracks: list[Track]):
        self.loggers = [t._loggers[self.logger_index] for t in tracks]
        redline = [log.filter_by_type(LogEntryType.ANY)
                   for log in self.loggers]
        self.redline = [np.array([entry.x for entry in track])
                        for track in redline]

        self._init_updates()
        self._init_predictions()
        self._clear_buffers()
        self._is_initialized = True
        self.red_lines_to_render = []

    def render(self, plot: plt.Axes, clear: bool = True):
        if not self._is_initialized:
            raise NotInitializedException

        # TODO: check if track_indices, epochs are in bounds
        if clear:
            plot.cla()

        for i in self.red_lines_to_render:
            self._render_track_redline(plot, i)
        plot.scatter([b[0] for b in self._prediction_buffer],
                     [b[1] for b in self._prediction_buffer],
                     color='purple', s=6)
        plot.scatter([b[0] for b in self._actuals_buffer],
                     [b[1] for b in self._actuals_buffer],
                     color='blue', s=10)
        plot.scatter([b[0] for b in self._considered_mts_buffer],
                     [b[1] for b in self._considered_mts_buffer],
                     color='orange', s=8)
        plot.scatter([b[0] for b in self._considered_clutter_buffer],
                     [b[1] for b in self._considered_clutter_buffer],
                     color='grey', s=8)

    def buffer(self, track: int, epochs: int | list[int], clear_buffer: bool = False):
        if not self._is_initialized:
            raise NotInitializedException

        if clear_buffer:
            self._clear_buffers()

        if isinstance(epochs, int):
            epochs = [epochs]

        for e in set(epochs) - self._buffered_epochs:
            self._buffer_track_epoch(track, e)

    def _buffer_track_epoch(self, track: int, epoch: int):
        self._actuals_buffer += [[self.actuals[track][epoch, 0],
                                  self.actuals[track][epoch, 1]]]
        self._considered_mts_buffer += [[mt[0], mt[1]]
                                        for mt in self.considered_measurements[track][epoch]]
        self._considered_clutter_buffer += [[mt[0], mt[1]]
                                            for mt in self.considered_clutter[track][epoch]]

        epoch_predictions = self.loggers[track].epochs[epoch]
        for ep in epoch_predictions[:-1]:
            self._prediction_buffer += [[self.predictions[track][ep-epoch, 0],
                                         self.predictions[track][ep-epoch, 1]]]
        self._buffered_epochs.add(epoch)

    def _render_track_redline(self, plot: plt.Axes, track: int) -> None:
        pos = self.loggers[track].epochs[max(self._buffered_epochs)][-1]
        plot.plot(self.redline[track][:pos, 0],
                  self.redline[track][:pos, 1],
                  color='red', linewidth=0.7)

    def _init_updates(self) -> None:
        updates = [log.filter_by_type(LogEntryType.UPDATE)
                   for log in self.loggers]
        self.actuals = [np.array([entry.x for entry in track])
                        for track in updates]
        self.considered_measurements = [[[mt.z for mt in entry.considered_measurements if not mt.is_clutter]
                                         for entry in track] for track in updates]
        self.considered_clutter = [[[mt.z for mt in entry.considered_measurements if mt.is_clutter]
                                    for entry in track] for track in updates]

    def _init_predictions(self):
        predictions = [log.filter_by_type(LogEntryType.PREDICTION)
                       for log in self.loggers]

        self.predictions = [np.array([entry.x for entry in track])
                            for track in predictions]

    def _clear_buffers(self):
        self._actuals_buffer = []
        self._prediction_buffer = []
        self._considered_mts_buffer = []
        self._considered_clutter_buffer = []
        self._buffered_epochs = set()


class NotInitializedException(Exception):

    def __init__(self) -> None:
        super().__init__('The track visualizer was not yet initialized')
