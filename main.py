
import numpy as np

from tracking.filters.jpdaf import PDA, Track, build_clusters, track_betas
from tracking.util.interpolator import LinearInterpolator, NodeCollection
from tracking.util.metrics import RadarGenerator, Scan, TrackHistory
from tracking.util.path import Path2D


def main():
    P = np.eye(4)
    R = np.diag([1, 1]) * 0.5
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    x_model = np.array([[1, 0, 0, 0]])
    P_model = np.eye(4)

    TMAX = 10

    nc1 = NodeCollection(np.array([[20, 20], [-20, 20]]), [0, TMAX])
    nc2 = NodeCollection(np.array([[-20, -20], [20, -20]]), [0, TMAX])

    l1 = Path2D(nc1, interpolatorClass=LinearInterpolator)
    l2 = Path2D(nc2, interpolatorClass=LinearInterpolator)

    nprobes = 100
    probexy = np.zeros(nprobes)
    probetimes = np.linspace(0.3, TMAX-0.5, nprobes)
    radar = RadarGenerator([l1, l2], 0.0, 1)
    scans = radar.make_scans_series(
        np.array([probetimes, probexy, probexy]).T)

    # for sc in scans[0:10]:
    #     print(len(sc.measurements_list))

    initial_states = [
        np.array([20, 0, 20, 0]),
        np.array([-20, 0, -20, 0]),
    ]

    tracks = [Track(state, P, H, R) for state in initial_states]
    for track in tracks:
        track.add_history(TrackHistory(x_model, P_model))

    nsteps = 10000
    dt = 0.02
    t = 0

    F1 = np.array([[1, dt], [0, 1]])
    F = np.kron(np.eye(2), F1)
    v = np.array([dt**2 / 2, dt]).reshape(2, 1)
    Q1 = v.dot(v.T)
    Q = np.kron(np.eye(2), Q1)

    for s in range(nsteps):
        t += dt

        for track in tracks:
            track.predict(F=F, Q=Q, time=t)

        if t+dt >= scans[0].time:
            for track in tracks:
                track.measurement_selection(scans[0], 25)

            clusters = build_clusters(tracks)

            for clus in clusters:
                betas = track_betas(clus.tracks)
                for t, track in enumerate(clus.tracks):
                    PDA(track, betas[t], scans[0])

            scans.pop(0)
            if len(scans) == 0:
                break

    print(tracks[0]._history_objects[0].flatten_x())


if __name__ == "__main__":
    main()
