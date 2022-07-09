from ast import List
import numpy as np


class Measurement:

    def __init__(self, i: int, z: np.array) -> None:
        self.i = i
        self.z = z


class Track:

    def __init__(self, x: np.array, P: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        self.x = x
        self.P = P
        self.H = H
        self.R = R
        self.P_G = 0.95
        self.P_D = 0.9

        self.mts_likelihoods = {}
        self.sel_mts_indices = set()

    def predict(self) -> None:
        pass

    def measurement_selection(self, mts: List(Measurement), g: float) -> None:
        S = self.S
        S_inv = np.linalg.inv(S)
        norm = 1 / np.sqrt(2 * np.pi * np.linalg.det(S))

        y = self.H.dot(self.x)
        self.mts_likelihoods = {0: 1}
        self.sel_mts_indices = set([0])
        for mt in mts:
            y_ = (mt.z - y).reshape(len(y), 1)
            if y_.T.dot(S_inv).dot(y_) < g:
                likelihood = norm * np.exp(-y_.T.dot(S).dot(y_))[0, 0]
                self.mts_likelihoods.update({mt.i: likelihood})
                self.sel_mts_indices.add(mt.i)
            else:
                self.mts_likelihoods.update({mt.i: 0})
        # self.sel_mts_indices = set(self.mts_likelihoods.keys())

    @property
    def S(self) -> np.ndarray:
        return self.H.dot(self.P).dot(self.H.T) + self.R


def track_betas(cluster_tracks: List(Track)) -> List(float):
    assignments = [t.sel_mts_indices for t in cluster_tracks]
    tracks_betas = []
    for track in range(len(cluster_tracks)):
        betas_t = {}
        for i in cluster_tracks[track].sel_mts_indices:
            beta_i = 0
            t_i_events = __generate_tau_i_events(track, i, assignments)
            # sum over each event
            for e in t_i_events:
                beta_i += np.prod([__lookup_event_track_weight(cluster_tracks[t], e[t])
                                  for t in range(len(e))])
            betas_t.update({i: beta_i})
        # normalize betas
        tracks_betas.append(betas_t)


def __lookup_event_track_weight(track: Track, mt_index: int):
    if mt_index > 0:
        return track.P_G * track.P_D * track.mts_likelihoods[mt_index]
    else:
        return 1 - track.P_G * track.P_D


def __generate_tau_i_events(t_index, mt_index, assignments):
    M = assignments.copy()
    M.pop(t_index)
    u = [] if mt_index == 0 else [mt_index]
    if 0 in u:
        u.remove(0)

    events = []
    __enumerate_events(M, events, u, [])

    for e in events:
        e.insert(t_index, mt_index)
    return events


def __enumerate_events(M, E, u, v, d=0):
    if d is len(M):
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
            __enumerate_events(M, E, unew, vnew, d+1)


class Cluster:

    def __init__(self) -> None:
        self.mts_indices = set()
        self.tracks: List(Track) = []

    def add_track(self, track: Track) -> None:
        self.tracks.append(track)
        self.mts_indices.update(track.sel_mts_indices)

    def overlap(self, indices: List(int)) -> bool:
        for i in indices:
            if i != 0 and i in self.mts_indices:
                return True
        return False


def build_clusters(tracks: List(Track)) -> List(Cluster):
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
