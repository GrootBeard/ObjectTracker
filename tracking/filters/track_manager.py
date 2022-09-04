import numpy as np

from tracking.filters.jipdaf import Track, build_clusters, track_betas, PDA
from tracking.util.metrics import Scan, Measurement


class TrackManager:

    def __init__(self) -> None:
        self._tracks = []
        self.track_id_counter = 0
        self.logger = Logger()

    def predict_tracks(self, time, dt: float) -> None:
        F1 = np.array([[1, dt], [0, 1]])
        F = np.kron(np.eye(2), F1)
        v = np.array([dt**2 / 2, dt]).reshape(2, 1)
        Q1 = v.dot(v.T)
        Q = np.kron(np.eye(2), Q1)

        for track in self._tracks:
            track.predict(F=F, Q=Q, time=time)
            track.prob_existence *= 1-dt/1000

        self.log_epoch(time=time+dt, measurements_map=None, update=False)

    def update_tracks(self, scan: Scan, time: float) -> None:
        dt = scan.time - time
        self.predict_tracks(time, dt)

        for track in self._tracks:
            track.measurement_selection(scan, 50)

        clusters = build_clusters(self._tracks)
        for clus in clusters:
            betas = track_betas(clus.tracks, clus.mts_indices,
                                sans_existence=False, clutter_density=clus.avg_clutter_density())

            for tau, track in enumerate(clus.tracks):
                PDA(track, betas[tau], scan)

        self.log_epoch(
            time=scan.time, measurements_map=scan.measurements, update=True)

    def log_epoch(self, time: float, measurements_map: dict, update: bool) -> None:
        self.logger.log_epoch(time=time, active_tracks=self._tracks,
                              measurements_map=measurements_map, is_update=update)

    def initialize_track(self, mean, 
            cov=np.eye(4),
            ms_mat=np.array([[1, 0,  0, 0], [0, 0, 1, 0]]),
            ms_uncertainty_mat=np.diag([1, 1])) -> None:
        self._tracks.append(
            Track(mean, cov, ms_mat, ms_uncertainty_mat, uid=self.track_id_counter))
        self.track_id_counter += 1

    def one_point_init(self):
        pass

    def two_point_init(self):
        pass


class Logger:

    def __init__(self) -> None:
        self.epochs = []
        self.prediction_backlog = []
        self.epochs_per_track = {}

    def log_epoch(self, time: float, active_tracks: list[Track], measurements_map: dict, is_update: bool) -> None:
        track_data = {track.uid: {
            'selected_measurements': [track._last_scan.measurements_list[i - 1].uid for i in track.sel_mts_indices if i != 0]
            if is_update else [],
            'position': [track.x[0], track.x[2]],
            'velocity': [track.x[1], track.x[3]],
            'covariance': track.P}
            for track in active_tracks}

        epoch = LogEpoch(time=time, track_data=track_data,
                         is_update=is_update, measurements_map=measurements_map)

        if is_update:
            for track in epoch.tracks:
                if track not in self.epochs_per_track.keys():
                    self.epochs_per_track.update({track: []})
                self.epochs_per_track[track].append(len(self.epochs))

            epoch.set_child_epochs(self.prediction_backlog)
            self.prediction_backlog = []
            self.epochs.append(epoch)

        else:
            self.prediction_backlog.append(epoch)

    def buffer_epoch(self, epoch: int) -> dict:
        # return {
        #         ''
        #     }
        pass

    def buffer_track(self, track_uid) -> dict:
        # return {'actuals': []}
        # for ep in self.epochs_per_track[track_uid]:
        # pass
        pass

    def tracks_in_epoch(self, epoch: int) -> list[int]:
        if epoch < 0 or epoch >= len(self.epochs):
            # TODO: raise error
            return []

        return self.epochs[epoch].tracks


class LogEpoch:

    def __init__(self, time: float, track_data: dict, is_update: bool, measurements_map: dict = None):
        self.track_data = track_data
        self.is_update = is_update
        self.measurements_map = measurements_map
        self.child_epochs = []

    @property
    def tracks(self) -> list[int]:
        return self.track_data.keys()

    def track_data(self, track_uid: int) -> dict:
        return self.track_data[track_uid]

    def set_child_epochs(self, child_epochs: list):
        self.child_epochs = child_epochs


class TrackingVisualizer:

    def __init__(self, log: Logger):
        self.log = log
        self.render_config = {"epoch_min": 0,
                              "epoch_max": 0,
                              "excluded_tracks": [],
                              "fixed_tracks": []}
        self.num_rendered_epochs = 5

    def render(self, plot):
        tracks_in_rendered_epochs = []
        for ep in range(self.render_config["epoch_min"], self.render_config["epoch_max"]+1):
            tracks_in_rendered_epochs.extend(self.log.tracks_in_epoch(ep))
        # {self.log.tracks_in_epoch(epoch) for epoch in range(
            # self.render_config["epoch_min"], self.render_config["epoch_max"]+1)}
        tracks_in_rendered_epochs = set(tracks_in_rendered_epochs).difference(
            set(self.render_config["excluded_tracks"]))

        actuals = []
        predictions = []
        measurements = []
        clutter_mts = []

        for track in tracks_in_rendered_epochs:
            rendered_track = self.render_track(
                track, self.render_config["epoch_min"], self.render_config["epoch_max"])

            # red line has to be rendered here

            actuals.extend(rendered_track['actuals'])
            predictions.extend(rendered_track['predictions'])

            for ep, mts in rendered_track['measurements'].items():
                measurements.extend(
                    [self.log.epochs[ep].measurements_map[mt].z for mt in mts if not self.log.epochs[ep].measurements_map[mt].is_clutter])
                clutter_mts.extend(
                    [self.log.epochs[ep].measurements_map[mt].z for mt in mts if self.log.epochs[ep].measurements_map[mt].is_clutter])

        plot.scatter([b[0] for b in measurements],
                     [b[1] for b in measurements],
                     color='orange', s=8)
        plot.scatter([b[0] for b in clutter_mts],
                     [b[1] for b in clutter_mts],
                     color='grey', s=8)
        plot.scatter([p[0] for p in predictions],
                     [p[1] for p in predictions],
                     color='purple', s=6)
        plot.scatter([a[0] for a in actuals],
                     [a[1] for a in actuals],
                     color='blue', s=10)

    def render_track(self, track_uid, epoch_min, epoch_max):
        actuals = []
        predictions = []
        measurements = {}

        track_epochs = self.log.epochs_per_track[track_uid]

        for ep in set(track_epochs).intersection(set(range(epoch_min, epoch_max+1))):
            actuals.append(
                self.log.epochs[ep].track_data[track_uid]['position'])

            # measurements[ep] = self.log.epochs[ep].track_data[track_uid]["selected_measurements"]
            measurements[ep] = self.log.epochs[ep].track_data[track_uid]["selected_measurements"]


            predictions.extend(child_ep.track_data[track_uid]['position']
                               for child_ep in self.log.epochs[ep].child_epochs)

        return {"actuals": actuals, "predictions": predictions, "measurements": measurements}

    def clear(self):
        pass

    def advance_epoch(self) -> bool:
        if self.render_config['epoch_max'] >= len(self.log.epochs):
            return False
        self.render_config['epoch_max'] += 1
        
        if self.render_config['epoch_max'] > self.num_rendered_epochs:
            self.render_config['epoch_min'] += 1
        return True

    def retread_epoch(self):
        if self.render_config['epoch_min'] > 0:
            self.render_config['epoch_min'] -= 1
        else:
            return False

        if self.render_config['epoch_max'] > self.num_rendered_epochs:
            self.render_config['epoch_max'] -= 1

        return True

    def set_number_rendered_epochs(self, num_epochs: int):
        if num_epochs < 1:
            return

        self.num_rendered_epochs = num_epochs
        self.render_config['epoch_min'] = max(
            0, self.render_config['epoch_max'] - self.num_rendered_epochs)

