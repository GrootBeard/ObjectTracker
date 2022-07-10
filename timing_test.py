
import timeit
import numpy as np
from filters.jpda import Scan, Track, Measurement, build_clusters, track_betas, PDA


P = np.eye(4)
R = np.diag([1, 1])
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

initial_states = [
    np.array([20.8, -8, -0.1, 0]),
    np.array([20, 0, 0, 0]),
    np.array([-22, 0, 0.5, 0]),
    np.array([-21, 0, 80, 0]),
    np.array([-20.5, 0, -0.5, 0])
]

scan = Scan([
    Measurement(np.array([21, 0])),
    Measurement(np.array([-20.4, 80])),
    Measurement(np.array([-20.4, -0.3])),
    Measurement(np.array([-20.4, -0.2])),
    Measurement(np.array([-20.4, -0.1])),
    Measurement(np.array([-20.4, -0.0])),
    Measurement(np.array([18, 0])),
    Measurement(np.array([0, 0])),
    Measurement(np.array([-21, -0.3])),
    Measurement(np.array([-22, -0.2])),
    Measurement(np.array([-22.5, -0.1])),
    Measurement(np.array([-21, -0.1]))
])


def fn():
    tracks = [Track(state, P, H, R) for state in initial_states]
    for track in tracks:
        track.measurement_selection(scan, 25)

    clusters = build_clusters(tracks)

    for clus in clusters:
        betas = track_betas(clus.tracks)
        for t, track in enumerate(clus.tracks):
            PDA(track, betas[t], scan)


N = 100
elapsed_time = timeit.timeit(stmt=fn, number=N)
print(f'elapsed time: {elapsed_time/N * 1000}')
