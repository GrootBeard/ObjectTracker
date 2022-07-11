import matplotlib.pyplot as plt
import numpy as np

from tracking.filters.jpdaf import PDA, Track, build_clusters, track_betas
from tracking.util.interpolator import LinearInterpolator, NodeCollection
from tracking.util.metrics import RadarGenerator, Scan, TrackLog
from tracking.util.path import Path2D


def main():
    P = np.eye(4)
    R = np.diag([1, 1])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    x_model = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    P_model = np.eye(4)

    TMAX = 50

    nc1 = NodeCollection(np.array([[20, 20], [-20, 20]]), [0, TMAX])
    nc2 = NodeCollection(np.array([[-20, -20], [20, -20]]), [0, TMAX])

    l1 = Path2D(nc1, interpolatorClass=LinearInterpolator)
    l2 = Path2D(nc2, interpolatorClass=LinearInterpolator)

    nprobes = 30
    probexy = np.zeros(nprobes)
    probetimes = np.linspace(0.3, TMAX-0.5, nprobes)
    radar = RadarGenerator([l1, l2], 0.0, 1)
    scans = radar.make_scans_series(
        np.array([probetimes, probexy, probexy]).T)

    initial_states = [
        np.array([20, 0, 20, 0]),
        np.array([-20, 0, -20, 0]),
    ]

    tracks = [Track(state, P, H, R) for state in initial_states]
    for track in tracks:
        track.add_history(TrackLog(x_model, P_model))

    nsteps = 10000
    dt = 0.1
    time = 0

    F1 = np.array([[1, dt], [0, 1]])
    F = np.kron(np.eye(2), F1)
    v = np.array([dt**2 / 2, dt]).reshape(2, 1)
    Q1 = v.dot(v.T)
    Q = np.kron(np.eye(2), Q1)

    for s in range(nsteps):
        time += dt

        for track in tracks:
            track.predict(F=F, Q=Q, time=time)

        if time+dt >= scans[0].time:
            dt_ = scans[0].time - time
            time = scans[0].time

            F1_ = np.array([[1, dt_], [0, 1]])
            F_ = np.kron(np.eye(2), F1_)
            v_ = np.array([dt_**2 / 2, dt_]).reshape(2, 1)
            Q1_ = v.dot(v_.T)
            Q_ = np.kron(np.eye(2), Q1_)

            for track in tracks:
                track.predict(F=F_, Q=Q_, time=time)
                track.measurement_selection(scans[0], 25)

            clusters = build_clusters(tracks)

            for clus in clusters:
                betas = track_betas(clus.tracks)
                for t, track in enumerate(clus.tracks):
                    PDA(track, betas[t], scans[0])

            scans.pop(0)
            if len(scans) == 0:
                break

    data = tracks[0]._loggers[0].flatten_x()
    data_time = tracks[0]._loggers[0].flatten_time()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data_time, data[0])
    plt.subplot(2, 1, 2)
    plt.plot(data_time, data[1])
    plt.show()


if __name__ == "__main__":
    main()
