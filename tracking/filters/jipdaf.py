import numpy as np
from scipy.special import gamma
from tracking.util.metrics import LogEntryType, Scan, TrackLog


class Track:
    __slots__ = ("dim", "uid", "_x", "_P", "_prob_existence", "H", "R", "prob_gate", "prob_detection", "gate_size",
                 "mts_likelihoods", "sel_mts_indices", "_loggers", "_last_scan")

    def __init__(self, x: np.array, P: np.ndarray, H: np.ndarray, R: np.ndarray, dim: int, existence_prob=1.0, uid=-1) -> None:
        self.dim = dim
        self._x = x
        self._P = P
        self._prob_existence = existence_prob
        self.H = H
        self.R = R
        self.prob_gate = 0.99
        self.prob_detection = 0.99
        self.gate_size = 0
        self.mts_likelihoods = {}
        self.sel_mts_indices = set()
        self._last_scan = None

        self.uid = uid
        self._loggers = []

    def predict(self, F: np.ndarray, Q: np.ndarray, time: float) -> None:
        new_x = F.dot(self._x)
        new_P = F.dot(self._P).dot(F.T) + Q
        self.update(new_x, new_P, time, LogEntryType.PREDICTION)

    def measurement_selection(self, scan: Scan, g: float) -> None:
        self.gate_size = g
        S = self.S
        S_inv = np.linalg.inv(S)
        # norm = np.sqrt((2 * np.pi)**len(self.x) * np.linalg.det(S))
        norm = np.sqrt((2 * np.pi)**self.dim * np.linalg.det(S))
        norm *= self.prob_gate

        y = self.H.dot(self._x)
        self.mts_likelihoods = {0: 1}
        self.sel_mts_indices = {0}
        for _, mt in scan.measurements.items():
            y_ = (mt.z - y).reshape(len(y), 1)
            error_measure = y_.T.dot(S_inv).dot(y_)[0, 0]
            if error_measure < self.gate_size:
                likelihood = np.exp(-error_measure) / norm
                self.mts_likelihoods.update({mt.uid: likelihood})
                self.sel_mts_indices.add(mt.uid)
            else:
                self.mts_likelihoods.update({mt.uid: 0})

        self._last_scan = scan

    def update(self, new_x, new_P, time: float, hist_entry_type=LogEntryType.UPDATE):
        self._x = new_x
        self._P = new_P
        for logger in self._loggers:
            considered_measurements = [] if hist_entry_type is not LogEntryType.UPDATE else [
                self._last_scan.measurements_list[i-1] for i in self.sel_mts_indices if i != 0]
            logger.add_entry(self.x, self.P, time,
                             hist_entry_type, considered_measurements)

    @property
    def x(self):
        return self._x

    @property
    def P(self):
        return self._P

    @property
    def prob_existence(self):
        return self._prob_existence

    @prob_existence.setter
    def prob_existence(self, value: float):
        self._prob_existence = value

    @property
    def S(self) -> np.ndarray:
        lt = self.H.dot(self._P).dot(self.H.T)
        rt = self.R
        print(lt)
        print(rt)
        return self.H.dot(self._P).dot(self.H.T) + self.R

    def gate_volume(self):
        # dim = len(self.x)
        return np.power(np.pi, self.dim/2) * np.sqrt(self.gate_size * np.linalg.det(self.S)) / gamma(self.dim/2 + 1)

    def selected_mts_count(self):
        return len(self.sel_mts_indices) - 1

    def add_logger(self, hist: TrackLog):
        self._loggers.append(hist)


def track_betas(cluster_tracks: list[Track], cluster_mts_indices: set[int], clutter_density: float, sans_existence=False) -> dict:
    assignments = [t.sel_mts_indices for t in cluster_tracks]
    tracks_betas = []
    for tau, track in enumerate(cluster_tracks):
        association_probabilities = [_calculate_association_prob(
            cluster_tracks, tau, i, assignments, clutter_density) for i in cluster_mts_indices]

        weight = sum(association_probabilities)
        association_probabilities = [
            ap / weight for ap in association_probabilities] if weight > 0 else [0]*len(association_probabilities)

        association_probabilities[0] *= ((1 - track.prob_gate * track.prob_detection) * track.prob_existence) / (
            1 - track.prob_gate * track.prob_detection * track.prob_existence)

        existence_prob = 1 if sans_existence else sum(association_probabilities)

        betas_tau = {
            t: association_probabilities[i] / existence_prob for i, t in enumerate(cluster_mts_indices)}

        track.prob_existence = existence_prob

        tracks_betas.append(betas_tau)
    return tracks_betas


def _calculate_association_prob(cluster_tracks: list[Track], tau: int, mt_index: int, assignments: list[int], clutter_density: float) -> float:
    tau_i_events = _generate_tau_i_events(tau, mt_index, assignments)
    return sum(np.prod([_lookup_event_track_weight(cluster_tracks[t], mt, clutter_density) for t, mt in enumerate(e)]) for e in tau_i_events)


def _lookup_event_track_weight(track: Track, mt_index: int, clutter_density: float) -> float:
    if mt_index > 0:
        # print(f'clutter density: {clutter_density}')
        return track.prob_gate * track.prob_detection * track.mts_likelihoods[mt_index] * track.prob_existence / clutter_density
        # return track.prob_gate * track.prob_detection * track.mts_likelihoods[mt_index] * track.prob_existence / 1e-10 
    return 1 - track.prob_gate * track.prob_detection * track.prob_existence


def _generate_tau_i_events(t_index: int, mt_index: int, assignments: list[int]) -> list[list[int]]:
    M = assignments.copy()
    M.pop(t_index)
    u = [] if mt_index == 0 else [mt_index]
    if 0 in u:
        u.remove(0)

    events = []
    _enumerate_events(M, events, u, [])

    for e in events:
        e.insert(t_index, mt_index)

    return events


def _enumerate_events(M, E, u, v, d=0) -> None:
    if d == len(M):
        E.append(v)
        return

    for i in M[d]:
        if i not in u:
            vnew = v.copy()
            vnew.append(i)
            unew = u.copy()
            # feasible events can have multiple tracks with no measurement assigned
            if i != 0:
                unew.append(i)
            _enumerate_events(M, E, unew, vnew, d+1)


def PDA(track: Track, betas: dict, scan: Scan) -> None:
    keys = list(betas.keys())
    keys.pop(0)
    if not keys:
        return

    K = track.P.dot(track.H.T).dot(np.linalg.inv(track.S))

    yi = np.array([scan.measurements_list[i-1].z for i in keys])
    innovation = np.vstack(
        [np.zeros(len(track.H.dot(track.x))), yi - track.H.dot(track.x)])

    beta_i = np.array([list(betas.values())])

    xi_kk = track.x + K.dot(innovation.T).T
    summand_x = beta_i.T * xi_kk
    x_kk = np.sum(summand_x, axis=0)

    Pi_kk = (np.eye(len(track.P)) - K.dot(track.H)).dot(track.P)
    error_prod = [np.array([row]).T.dot(np.array([row]))
                  for row in np.array(xi_kk - x_kk)]
    summand_P = np.array([beta_i]).T * (Pi_kk + error_prod)
    P_kk = np.sum(summand_P, axis=0)

    track.update(x_kk, P_kk, scan.time, LogEntryType.UPDATE)


class Cluster:
    __slots__ = ("mts_indices", "_tracks")

    def __init__(self) -> None:
        self.mts_indices = set()
        self._tracks: list[Track] = []

    def add_track(self, track: Track) -> None:
        self._tracks.append(track)
        self.mts_indices.update(track.sel_mts_indices)

    def overlap(self, indices: list[int]) -> bool:
        return any(i != 0 and i in self.mts_indices for i in indices)

    @property
    def tracks(self):
        return self._tracks

    def cluster_selection_area(self) -> float:
        num_cluster_measurements = len(self.mts_indices) - 1
        if num_cluster_measurements < 1:
            return 1.0

        track_selection_areas = [track.gate_volume() for track in self.tracks]
        overlap_ratio = num_cluster_measurements / \
            sum(track.selected_mts_count() for track in self.tracks)

        return max(sum(track_selection_areas) * overlap_ratio, max(track_selection_areas))

    def avg_clutter_density(self):
        avg_clutter_mts = 0.0
        for i in list(self.mts_indices)[1:]:
            avg_clutter_mts += np.prod([1 - tau.prob_gate * tau.prob_detection * tau.prob_existence * tau.mts_likelihoods[i] / sum(list(tau.mts_likelihoods.values())[1:])
                                        for tau in self.tracks])

        return avg_clutter_mts / self.cluster_selection_area()


def build_clusters(tracks: list[Track], confirmation_threshold: float, max_cluster_size: int) -> list[Cluster]:
    clusters = []
    for t in tracks:
        overlap = False

        # if track existence probability is low revert to disjoint PDA to reduce computation
        if t.prob_existence >= confirmation_threshold: 
            for clus in clusters:
                # if len(clus.tracks) >= max_cluster_size or len(clus.mts_indices) + len(t.sel_mts_indices) > 35:
                    # continue

                if clus.overlap(t.sel_mts_indices):
                    clus.add_track(t)
                    overlap = True
                    break
        if not overlap:
            new_clus = Cluster()
            new_clus.add_track(t)
            clusters.append(new_clus)

    return clusters
