import numpy as np
from tracking.util.metrics import LogEntryType, Scan, TrackLog


class Track:
    __slots__ = ("_x", "_P", "_prob_existence", "H", "R", "prob_gate", "prob_detection",
                 "mts_likelihoods", "sel_mts_indices", "_loggers", "_last_scan")

    def __init__(self, x: np.array, P: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        self._x = x
        self._P = P
        self._prob_existence = 0.999
        self.H = H
        self.R = R
        self.prob_gate = 0.99999
        self.prob_detection = 0.99
        self.mts_likelihoods = {}
        self.sel_mts_indices = set()
        self._last_scan = None

        self._loggers = []

    def predict(self, F: np.ndarray, Q: np.ndarray, time: float) -> None:
        new_x = F.dot(self._x)
        new_P = F.dot(self._P).dot(F.T) + Q
        # TODO: Update probability of existence
        # dt is required to do that
        self.update(new_x, new_P, time, LogEntryType.PREDICTION)

    def measurement_selection(self, scan: Scan, g: float) -> None:
        S = self.S
        S_inv = np.linalg.inv(S)
        norm = np.sqrt((2 * np.pi)**len(self.x) * np.linalg.det(S))
        norm *= self.prob_gate

        y = self.H.dot(self._x)
        self.mts_likelihoods = {0: 1}
        self.sel_mts_indices = {0}
        for _, mt in scan.measurements:
            y_ = (mt.z - y).reshape(len(y), 1)
            error_measure = y_.T.dot(S_inv).dot(y_)[0, 0]
            if error_measure < g:
                likelihood = np.exp(-error_measure) / norm
                self.mts_likelihoods.update({mt.mt_id: likelihood})
                self.sel_mts_indices.add(mt.mt_id)
            else:
                self.mts_likelihoods.update({mt.mt_id: 0})

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
        return self.H.dot(self._P).dot(self.H.T) + self.R

    def add_logger(self, hist: TrackLog):
        self._loggers.append(hist)


def track_betas(cluster_tracks: list[Track], cluster_mts_indices: set[int], sans_existence: bool = False) -> dict:
    assignments = [t.sel_mts_indices for t in cluster_tracks]
    tracks_betas = []
    for tau, track in enumerate(cluster_tracks):
        association_probabilities = [_calculate_association_prob(
            cluster_tracks, tau, i, assignments) for i in cluster_mts_indices]

        weight = sum(association_probabilities)
        association_probabilities = [
            ap / weight for ap in association_probabilities] if weight > 0 else [0]*len(association_probabilities)

        association_probabilities[0] *= ((1 - track.prob_gate * track.prob_detection) * track.prob_existence) / (
            1 - track.prob_gate * track.prob_detection * track.prob_existence)

        existence_prob = 1 if sans_existence else sum(association_probabilities)
        
        betas_tau = {
            t: association_probabilities[i] / existence_prob for i, t in enumerate(cluster_mts_indices)}

        print(f'probability of existence: {existence_prob}')
        track.prob_existence = existence_prob
        
        tracks_betas.append(betas_tau)
    return tracks_betas


def _calculate_association_prob(cluster_tracks: list[Track], tau: int, mt_index: int, assignments: list[int]) -> float:
    tau_i_events = _generate_tau_i_events(tau, mt_index, assignments)
    return sum(np.prod([_lookup_event_track_weight(cluster_tracks[t], mt) for t, mt in enumerate(e)]) for e in tau_i_events)


def _lookup_event_track_weight(track: Track, mt_index: int) -> float:
    if mt_index > 0:
        return track.prob_gate * track.prob_detection * track.mts_likelihoods[mt_index] * track.prob_existence / 0.0015
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


def build_clusters(tracks: list[Track]) -> list[Cluster]:
    clusters = []
    for t in tracks:
        overlap = False
        for clus in clusters:
            if clus.overlap(t.sel_mts_indices):
                clus.add_track(t)
                overlap = True
                break
        if not overlap:
            new_clus = Cluster()
            new_clus.add_track(t)
            clusters.append(new_clus)

    return clusters
