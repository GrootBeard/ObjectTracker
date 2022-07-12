import matplotlib.pyplot as plt
import numpy as np

from tracking.filters.jpdaf import PDA, Track, build_clusters, track_betas
from tracking.util.interpolator import LinearInterpolator, NodeCollection
from tracking.util.metrics import RadarGenerator, Scan, TrackLog
from tracking.util.path import Path2D


def main():
    dt = 0.05

    F1 = np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]])
    F = np.kron(np.eye(2), F1)
    v = np.array([dt**2 / 2, dt, 1]).reshape(3, 1)
    Q1 = v.dot(v.T)
    Q = np.kron(np.eye(2), Q1)

    P = np.eye(6)
    R = np.diag([1, 1])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    x_model = H
    P_model = P

    TMAX = 40

    nc1 = NodeCollection(np.array([[-20, 0], [20, 0]]), [0, TMAX])
    nc2 = NodeCollection(np.array([[0, -20], [0, 20]]), [0, TMAX])

    l1 = Path2D(nc1, interpolatorClass=LinearInterpolator)
    l2 = Path2D(nc2, interpolatorClass=LinearInterpolator)

    initial_states = [
        np.array([-20, 0, 0, 0, 0.2, 0]),
        np.array([0, -0.05, 0, -20, 0.4, 0]),
    ]

    nprobes = 200
    probexy = np.zeros(nprobes)
    probetimes = np.linspace(0.3, TMAX-0.5, nprobes)
    radar = RadarGenerator([l1, l2], 0.1, 1)
    scans = radar.make_scans_series(
        np.array([probetimes, probexy, probexy]).T)

    tracks = [Track(state, P, H, R) for state in initial_states]
    for track in tracks:
        track.add_logger(TrackLog(x_model, P_model))

    nsteps = int(TMAX//dt)
    time = 0

    working_scans = scans.copy()

    for s in range(nsteps):
        time += dt

        for track in tracks:
            track.predict(F=F, Q=Q, time=time)

        if time+dt >= working_scans[0].time:
            dt_ = working_scans[0].time - time
            time = working_scans[0].time

            F1_ = np.array([[1, dt_, dt_**2/2], [0, 1, dt_], [0, 0, 1]])
            F_ = np.kron(np.eye(2), F1_)
            v_ = np.array([dt_**2 / 2, dt_, 1]).reshape(3, 1)
            Q1_ = v.dot(v_.T)
            Q_ = np.kron(np.eye(2), Q1_)

            for track in tracks:
                track.predict(F=F_, Q=Q_, time=time)
                track.measurement_selection(working_scans[0], 50)

            clusters = build_clusters(tracks)

            for clus in clusters:
                betas = track_betas(clus.tracks)
                for t, track in enumerate(clus.tracks):
                    PDA(track, betas[t], working_scans[0])

            working_scans.pop(0)
            if len(working_scans) == 0:
                break

    data = tracks[0]._loggers[0].flatten_x()
    data2 = tracks[1]._loggers[0].flatten_x()
    data_time = tracks[0]._loggers[0].flatten_time()

    mts_lists = [sc.values for sc in scans]

    measurements = []
    for l in mts_lists:
        measurements.extend(l)
    ms_x = [mt[0] for mt in measurements]
    ms_y = [mt[1] for mt in measurements]

    plt.ioff()
    plt.figure()
    plt.plot(data[0], data[1], color='r')
    plt.grid(True)
    plt.plot(data2[0], data2[1], color='g')
    plt.scatter(ms_x, ms_y, s=3)
    plt.show(block=True)


if __name__ == "__main__":
    main()
