import matplotlib.pyplot as plt
import numpy as np

from tracking.filters.jpdaf import PDA, Track, build_clusters, track_betas
from tracking.util.interpolator import LinearInterpolator, NodeCollection, SplineInterpolator
from tracking.util.metrics import RadarGenerator, Scan, TrackLog, PoissonClutter
from tracking.util.path import Path2D, PathFactory
from tracking.visualizer import TrackVisualizer 


def main():
    dt = 0.15
    # F1 = np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]])
    # F = np.kron(np.eye(2), F1)
    # v = np.array([dt**2 / 2, dt, 1]).reshape(3, 1)
    # Q1 = v.dot(v.T)
    # Q = np.kron(np.eye(2), Q1)

    # P = np.eye(6)
    # R = np.diag([1, 1])
    # H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    F1 = np.array([[1, dt], [0, 1]])
    F = np.kron(np.eye(2), F1)
    v = np.array([dt**2 / 2, dt]).reshape(2, 1)
    Q1 = v.dot(v.T)
    Q = np.kron(np.eye(2), Q1)

    P = np.eye(4)
    R = np.diag([1, 1])
    H = np.array([[1, 0,  0, 0], [0, 0, 1, 0]])

    x_model = H
    P_model = P

    TMAX = 500

    nodes = [
         # NodeCollection(np.array([[-400, -100], [400,  100]]), [0, TMAX]),
         NodeCollection(np.array([[-400, 100],  [400, -100]]), [0, TMAX]),
         NodeCollection(np.array([[-2000, 0], [2000, 0]]), [0, TMAX]),
         NodeCollection(np.array([[0, 2000], [0, -2000]]), [0, TMAX]),
    ]
    node = NodeCollection(np.array([[-400, -100], [-200, 25], [0, 0], [200, -25], [400, 100]]), np.linspace(0, TMAX, 5)) 
    factory = PathFactory()
    paths = factory.create(nodes, LinearInterpolator)
    paths += factory.create([node], SplineInterpolator) 

    initial_states = [
        np.array([-400, 1.0,  -100, .0]),
        np.array([-400, 1.0,  100, -.5]),
        np.array([0, 0,  2000, -8.9]),
        np.array([-2000, 8.9, 0, 0]),
#       np.array([0, 0, 0, 200, -11, 0]),
#       np.array([-40, 1.9, 0, 10, -0.6, 0]),
#       np.array([-200, 9.9, 0, 0, 0, 0]),
#       np.array([0, 0, 0, 200, -11, 0]),
    ]

    nprobes = 200
    probexy = np.zeros(nprobes)
    probetimes = np.linspace(0.0, TMAX-0.1, nprobes)
    radar = RadarGenerator(paths, 0.1, 1)
    clutter_model = PoissonClutter([-250, 250, -250, 250], 0.0008)
    # clutter_model = None
    scans = radar.make_scans_series(
        np.array([probetimes, probexy, probexy]).T, clutter_model)

    tracks = [Track(state, P, H, R) for state in initial_states]
    for track in tracks:
        track.add_logger(TrackLog(x_model, P_model))

    nsteps = 10000
    time = 0

    working_scans = scans.copy()

    gate = 20
    print('starting tracking')
    
    for _ in range(nsteps):
        time += dt

        for track in tracks:
            track.predict(F=F, Q=Q, time=time)

        if time+dt >= working_scans[0].time:
            dt_ = working_scans[0].time - time
            time = working_scans[0].time
#           F1_ = np.array([[1, dt_, dt_**2/2], [0, 1, dt_], [0, 0, 1]])
#           F_ = np.kron(np.eye(2), F1_)
#           v_ = np.array([dt_**2 / 2, dt_, 1]).reshape(3, 1)
#           Q1_ = v.dot(v_.T)
#           Q_ = np.kron(np.eye(2), Q1_)

            F1_ = np.array([[1, dt], [0, 1]])
            F_ = np.kron(np.eye(2), F1_)
            v_ = np.array([dt_**2 / 2, dt_]).reshape(2, 1)
            Q1_ = v_.dot(v_.T)
            Q_ = np.kron(np.eye(2), Q1_)

            for track in tracks:
                track.predict(F=F_, Q=Q_, time=time)
                track.measurement_selection(working_scans[0], gate)
             
            clusters = build_clusters(tracks)
            for clus in clusters:
                if len(clus.tracks) > 1:
                    print(f"Multiple tracks in cluster: {len(clus.tracks)}")
                betas = track_betas(clus.tracks, clus.mts_indices)
                for t, track in enumerate(clus.tracks):
                    PDA(track, betas[t], working_scans[0])

            working_scans.pop(0)
            if len(working_scans) == 0:
                print("Processed all scans")
                break
             
    print('finished tracking')
    track_data = [t._loggers[0].flatten_x() for t in tracks]

    vz = TrackVisualizer(0)
    vz.initialize(tracks)

    measurements = []
    for l in [sc.values for sc in scans]:
        measurements.extend(l)
    ms_x = [mt[0] for mt in measurements]
    ms_y = [mt[1] for mt in measurements]

    plt.ion()
    plt.figure()
    plt.subplot(1,2,1)
    for t in track_data:
        plt.plot(t[0], t[1])  
    plt.grid(True)
    # plt.scatter(ms_x, ms_y, s=3)

    input('Press enter to continue')
    plot = plt.subplot(1,2,2)
    for e in range(180):
        vz.buffer(0, e)
        vz.render(plot, True)
        plt.pause(0.1)

    plt.show(block=True)


if __name__ == "__main__":
    main()
