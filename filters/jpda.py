from ast import Dict, List
from dataclasses import dataclass
from typing import Set
import numpy as np


@dataclass
class Measurement:
    z: np.array


class Scan:
    def __init__(self, measurements: List(Measurement)) -> None:
        # measurement with index 0 is reserved for 'no-measurement'
        self._measurements = {i+1: mt for i, mt in enumerate(measurements)}

    @property
    def measurements(self):
        return self._measurements.items()

    @property
    def measurements_list(self):
        return list(self._measurements.values())

    @property
    def measurements_indices(self):
        return list(self._measurements.keys())


class Track:

    def __init__(self, x: np.array, P: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        self.x = x
        self.P = P
        self.H = H
        self.R = R
        self.P_G = 0.95
        self.P_D = 0.97

        self.mts_likelihoods = {}
        self.sel_mts_indices = set()

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

    def measurement_selection(self, scan: Scan, g: float) -> None:
        S = self.S
        S_inv = np.linalg.inv(S)
        norm = 1 / np.sqrt(2 * np.pi * np.linalg.det(S))

        y = self.H.dot(self.x)
        self.mts_likelihoods = {0: 1}
        self.sel_mts_indices = set([0])
        for i_mt, mt in scan.measurements:
            y_ = (mt.z - y).reshape(len(y), 1)
            if y_.T.dot(S_inv).dot(y_) < g:
                likelihood = norm * np.exp(-y_.T.dot(S).dot(y_))[0, 0]
                self.mts_likelihoods.update({i_mt: likelihood})
                self.sel_mts_indices.add(i_mt)
            else:
                self.mts_likelihoods.update({i_mt: 0})

    @property
    def S(self) -> np.ndarray:
        return self.H.dot(self.P).dot(self.H.T) + self.R


def track_betas(cluster_tracks: List(Track)):
    assignments = [t.sel_mts_indices for t in cluster_tracks]
    tracks_betas = []
    for tau, track in enumerate(cluster_tracks):
        betas_t = {}
        for i in track.sel_mts_indices:
            beta_i = 0
            t_i_events = __generate_tau_i_events(tau, i, assignments)
            for e in t_i_events:
                beta_i += np.prod([__lookup_event_track_weight(cluster_tracks[t], mt)
                                  for t, mt in enumerate(e)])
            betas_t.update({i: beta_i})
        # TODO: normalize betas
        sum = np.sum(list(betas_t.values()))
        # print(f'betas: {betas_t}')
        betas_t = {i: v/sum for i, v in betas_t.items()}
        tracks_betas.append(betas_t)
    return tracks_betas


def __lookup_event_track_weight(track: Track, mt_index: int) -> float:
    if mt_index > 0:
        return track.P_G * track.P_D * track.mts_likelihoods[mt_index]
    return 1 - track.P_G * track.P_D


def __generate_tau_i_events(t_index, mt_index, assignments) -> List(List(int)):
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


def __enumerate_events(M, E, u, v, d=0) -> None:
    if d is len(M):
        # Try pre-allocating the array v to be a zeros array and then set value
        # at index d
        E.append(v)
        return

    for i in M[d]:
        # loop over i in M[d] \ u
        if i not in u:
            vnew = v.copy()
            vnew.append(i)
            unew = u.copy()
            # feasible events can have multiple tracks with no measurement assigned
            if i != 0:
                unew.append(i)
            __enumerate_events(M, E, unew, vnew, d+1)


def PDA(track: Track, betas: Dict, scan: Scan):
    K = track.P.dot(track.H.T).dot(np.linalg.inv(track.S))
    keys = list(betas.keys())
    keys.pop(0)
    yi = np.array([scan.measurements_list[i-1].z for i in keys])
    innovation = np.vstack(
        [np.zeros(len(track.H.dot(track.x))), yi - track.H.dot(track.x)])

    beta_i = np.array([list(betas.values())])

    xi_kk = track.x + K.dot(innovation.T).T
    summand_x = beta_i.T * xi_kk
    x_kk = np.sum(summand_x, axis=0)

    Pi_kk = (np.eye(len(track.P)) - K.dot(track.H)).dot(track.P)
    # TODO: Properly vectorize this code
    error_prod = [np.array([row]).T.dot(np.array([row]))
                  for row in np.array(xi_kk - x_kk)]
    summand_P = np.array([beta_i]).T * (Pi_kk + error_prod)
    P_kk = np.sum(summand_P, axis=0)

    track.x = x_kk
    track.P = P_kk


class Cluster:

    def __init__(self) -> None:
        self.mts_indices = set()
        self._tracks: List(Track) = []

    def add_track(self, track: Track) -> None:
        self._tracks.append(track)
        self.mts_indices.update(track.sel_mts_indices)

    def overlap(self, indices: Set) -> bool:
        return bool(self.mts_indices.intersection(indices))

    @property
    def tracks(self):
        return self._tracks


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
