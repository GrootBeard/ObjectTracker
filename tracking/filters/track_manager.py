import numpy as np

from tracking.filters.jipdaf import PDA, Track, build_clusters, track_betas
from tracking.filters.dynamics_models import DynamicsModel
from tracking.util.metrics import Measurement, Scan


class TrackManager:

    def __init__(self, dynamics_model: DynamicsModel) -> None:
        self._tracks = []
        self.track_id_counter = 0
        self.logger = Logger()
        self.last_scan = None
        self.mts_non_association_probs = {}
        self.dynamics_model = dynamics_model

    def predict_tracks(self, time, dt: float) -> None:
        for track in self._tracks:
            track.predict(F=self.dynamics_model.F(dt), Q=self.dynamics_model.Q(dt), time=time)
            track.prob_existence *= 1-dt/10000

        self.log_epoch(time=time+dt, measurements_map=None, update=False)

    def update_tracks(self, scan: Scan, time: float) -> None:
        # print(f'number of active tracks: {len(self._tracks)}')
        dt = scan.time - time
        self.predict_tracks(time, dt)

        for track in self._tracks:
            track.measurement_selection(scan, 14)

        clusters = build_clusters(self._tracks, confirmation_threshold=0.30, max_cluster_size=9)
        self.mts_non_association_probs = {mt.uid: 1 for mt in scan.measurements.values()}
        
        for clus in clusters:
            if len(clus.tracks) > 1:
                print(f'cluster size: {len(clus.tracks)}')
            betas = track_betas(clus.tracks, clus.mts_indices,
                                sans_existence=False, clutter_density=clus.avg_clutter_density())

            for tau, track in enumerate(clus.tracks):
                PDA(track, betas[tau], scan)
                
                for ms_index, beta_tau_i in betas[tau].items():
                    if ms_index == 0:
                        continue
                    self.mts_non_association_probs[ms_index] -= beta_tau_i * track.prob_existence 

        self.log_epoch(
            time=scan.time, measurements_map=scan.measurements, update=True)

        self.last_scan = scan

    def log_epoch(self, time: float, measurements_map: dict, update: bool) -> None:
        self.logger.log_epoch(time=time, active_tracks=self._tracks,
                              measurements_map=measurements_map, is_update=update,
                              pos_indices=self.dynamics_model.pos_indices,
                              vel_indices=self.dynamics_model.vel_indices)

    def initialize_track(self, mean,
                         cov,
                         existence_prob=1.0) -> None:
        self._tracks.append(
            Track(mean, cov, self.dynamics_model.H(), self.dynamics_model.R([1, 1, 1]), 
                dim=self.dynamics_model.dim, uid=self.track_id_counter, existence_prob=existence_prob))
        self.track_id_counter += 1

    def one_point_init(self, vmax: float, p0: float, disable_clutter_tracks=False):
        assigments = self.track_assigned_measurements()

        for ms, tracks in assigments.items():
            if disable_clutter_tracks and self.last_scan.measurements[ms].is_clutter:
                continue
            
            if len(tracks) > 0:
                continue

            pos = self.dynamics_model.pos_one_point_init((self.last_scan.measurements[ms].z[0], self.last_scan.measurements[ms].z[1]))
            cov = self.dynamics_model.cov_one_point_init(vmax)
            existence_prob = p0 * self.mts_non_association_probs[ms]

            self.initialize_track(pos, cov, existence_prob=existence_prob)

    def two_point_init(self):
        pass

    def track_assigned_measurements(self):
        assignments = {mt: [] for mt in self.last_scan.measurements.keys()}
        for track in self._tracks:
            for sel_mt in track.sel_mts_indices.difference({0}):
                assignments[sel_mt].append(track.uid)

        return assignments

    def delete_false_tracks(self, existence_threshold: float) -> None:
        for track in self._tracks:
            if track.prob_existence < existence_threshold:
                self._tracks.remove(track)


class Logger:

    def __init__(self) -> None:
        self.epochs = []
        self.prediction_backlog = []
        self.epochs_per_track = {}
        self.metadata = {'track_data': {}}

    def log_epoch(self, time: float, active_tracks: list[Track], measurements_map: dict, is_update: bool, pos_indices: tuple, vel_indices: tuple) -> None:
        track_data = {track.uid: {
            'selected_measurements': [track._last_scan.measurements_list[i - 1].uid for i in track.sel_mts_indices if i != 0]
            if is_update else [],
            'position': [track.x[pos_indices[0]], track.x[pos_indices[1]]],
            'velocity': [track.x[vel_indices[0]], track.x[vel_indices[1]]],
            'covariance': track.P,
            'existance_probability': track.prob_existence,}
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

            for track in active_tracks:
                if track.uid not in self.metadata['track_data'].keys():
                    self.metadata['track_data'].update({track.uid: {
                        'first_epoch': len(self.epochs),
                        'last_epoch': len(self.epochs),
                        'number_of_epochs': 1,
                    }})
                else:
                    self.metadata['track_data'][track.uid]['last_epoch'] += 1
                    self.metadata['track_data'][track.uid]['number_of_epochs'] += 1

        else:
            self.prediction_backlog.append(epoch)

    def tracks_in_epoch(self, epoch: int) -> list[int]:
        if epoch < 0 or epoch >= len(self.epochs):
            # TODO: raise error
            return []

        return self.epochs[epoch].tracks

    def track_lengths(self):
        return [track['number_of_epochs'] for track in self.metadata['track_data'].values()]

    def display_track_lengths(self, min_length: int):
        if min_length < 0:
            return

        filtered_tracks = {t: {'length': data['number_of_epochs']} for t, data in self.metadata['track_data'].items() if data['number_of_epochs'] > min_length}
        print(filtered_tracks)
        return filtered_tracks

    def track_existence_probability_history(self, track_uid: int):
        return [self.epochs[epoch].track_data[track_uid]['existance_probability'] for epoch in self.epochs_per_track[track_uid]]        
    
    def track_selected_measurements_history(self, track_uid: int):
        data = {}
        for epoch in self.epochs_per_track[track_uid]:
            num_actual_measurements = 0
            num_clutter_measurements = 0
            for mt in self.epochs[epoch].track_data[track_uid]['selected_measurements']:
                if self.epochs[epoch].measurements_map[mt].is_clutter:
                    num_clutter_measurements += 1
                else:
                    num_actual_measurements += 1

            data[epoch] = {'num_actual_measurements': num_actual_measurements, 'num_clutter_measurements': num_clutter_measurements}

        return data

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
                              "excluded_tracks": set(),
                              "fixed_tracks": set()}
        self.num_rendered_epochs = 5

    def render_epochs_in_range(self, plot, ep_min: int, ep_max: int):
        if ep_min > ep_max:
            # raise error
            return

        ep_min = max(ep_min, 0)
        ep_min = min(ep_min, len(self.log.epochs))
        ep_max = len(self.log.epochs) if ep_max < 0 else ep_max
        ep_max = min(ep_max, len(self.log.epochs))

        print(f'rendering all epochs in range ({ep_min}, {ep_max})')

        __epoch_min = self.render_config['epoch_min']
        __epoch_max = self.render_config['epoch_max']

        self.render_config['epoch_min'] = ep_min
        self.render_config['epoch_max'] = ep_max

        self.render(plot)

        self.render_config['epoch_min'] = __epoch_min
        self.render_config['epoch_max'] = __epoch_max

    def render(self, plot):
        tracks_in_rendered_epochs = []
        for ep in range(self.render_config["epoch_min"], self.render_config["epoch_max"]+1):
            tracks_in_rendered_epochs.extend(self.log.tracks_in_epoch(ep))

        tracks_in_rendered_epochs = set(tracks_in_rendered_epochs).difference(
            set(self.render_config["excluded_tracks"]))

        actuals = []
        predictions = []
        actual_mts = []
        clutter_mts = []

        for track in tracks_in_rendered_epochs:
            buffered_track = self.buffer_track(
                track, self.render_config["epoch_min"], self.render_config["epoch_max"])
            
            deleted_earlier = self.render_config['epoch_max'] > self.log.metadata['track_data'][track]['last_epoch']
            pos = buffered_track['actuals']
            
            if deleted_earlier:
                plot.plot([p[0] for p in pos],
                    [p[1] for p in pos],
                    color='grey',
                    linewidth=1.0)
                continue
            
            plot.plot([p[0] for p in pos],
                      [p[1] for p in pos],
                      color='red',
                      linewidth=0.7)

            actuals.extend(buffered_track['actuals'])
            predictions.extend(buffered_track['predictions'])

            actual_mts.extend(buffered_track['actual_mts'])
            clutter_mts.extend(buffered_track['clutter_mts'])
            # for ep, mts in buffered_track['measurement_uids'].items():
            #     measurements.extend(
            #         [self.log.epochs[ep].measurements_map[mt].z for mt in mts if not self.log.epochs[ep].measurements_map[mt].is_clutter])
            #     clutter_mts.extend(
            #         [self.log.epochs[ep].measurements_map[mt].z for mt in mts if self.log.epochs[ep].measurements_map[mt].is_clutter])

        self.render_buffer(plot, buffer={
                'actuals': actuals,
                'predictions': predictions,
                'actual_mts': actual_mts,
                'clutter_mts': clutter_mts,
            })

    def render_buffer(self, plot, buffer: dict):
        plot.scatter([b[0] for b in buffer['clutter_mts']],
                     [b[1] for b in buffer['clutter_mts']],
                     color='grey', s=8)
        plot.scatter([p[0] for p in buffer['predictions']],
                     [p[1] for p in buffer['predictions']],
                     color='purple', s=6)
        plot.scatter([b[0] for b in buffer['actual_mts']],
                     [b[1] for b in buffer['actual_mts']],
                     color='orange', s=8)
        plot.scatter([a[0] for a in buffer['actuals']],
                     [a[1] for a in buffer['actuals']],
                     color='blue', s=10)

    def render_track(self, plot, track_uid):
        buffer = self.buffer_track(track_uid, 0, len(self.log.epochs))
        self.render_buffer(plot, buffer)
        
    def buffer_track(self, track_uid, epoch_min, epoch_max):
        actuals = []
        predictions = []
        measurement_uids = {}
        actual_mts = []
        clutter_mts = []

        track_epochs = self.log.epochs_per_track[track_uid]

        for ep in set(track_epochs).intersection(set(range(epoch_min, epoch_max+1))):
            actuals.append(
                self.log.epochs[ep].track_data[track_uid]['position'])

            measurement_uids[ep] = self.log.epochs[ep].track_data[track_uid]["selected_measurements"]

            for ep, mts in measurement_uids.items(): 
                actual_mts.extend(
                    [self.log.epochs[ep].measurements_map[mt].z for mt in mts if not self.log.epochs[ep].measurements_map[mt].is_clutter])
                clutter_mts.extend(
                    [self.log.epochs[ep].measurements_map[mt].z for mt in mts if self.log.epochs[ep].measurements_map[mt].is_clutter])

            predictions.extend(child_ep.track_data[track_uid]['position']
                               for child_ep in self.log.epochs[ep].child_epochs)

        return {"actuals": actuals, "predictions": predictions, "actual_mts": actual_mts, "clutter_mts": clutter_mts, "measurement_uids": measurement_uids}

    def render_measurements(self, plot):
        if self.render_config['epoch_max'] >= len(self.log.epochs):
            return
        
        measurements = self.log.epochs[self.render_config['epoch_max']].measurements_map.values()
        plot.scatter([mt.z[0] for mt in measurements],
                     [mt.z[1] for mt in measurements],
                     color='brown', s=15)

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
        if num_epochs < 0:
            return
        if num_epochs == 0:
            num_epochs = len(self.log.epochs)

        self.num_rendered_epochs = num_epochs
        self.render_config['epoch_min'] = max(
            0, self.render_config['epoch_max'] - self.num_rendered_epochs)

    def filter_tracks_by_length(self, length, geq=True):
        if geq:
            filtered_uids = {uid for uid in self.log.metadata['track_data'].keys(
            ) if self.log.metadata['track_data'][uid]['number_of_epochs'] <= length}
        else:
            filtered_uids = {uid for uid in self.log.metadata['track_data'].keys(
            ) if self.log.metadata['track_data'][uid]['number_of_epochs'] >= length}

        self.render_config['excluded_tracks'] = self.render_config['excluded_tracks'].union(
            filtered_uids)

    def clear_filter(self):
        self.render_config['excluded_tracks'] = set()
