import matplotlib.pyplot as plt
import numpy as np

from tracking.filters.jipdaf import PDA, Track, build_clusters, track_betas
from tracking.util.interpolator import LinearInterpolator, NodeCollection, SplineInterpolator
from tracking.util.metrics import RadarGenerator, Scan, TrackLog, PoissonClutter
from tracking.util.path import Path2D, PathFactory
from tracking.visualizer import TrackVisualizer
from tracking.filters.track_manager import TrackManager, TrackingVisualizer

def main():
    
    TMAX = 500

    nodes = [
        # NodeCollection(np.array([[-400, -100], [400,  100]]), [0, TMAX]),
        NodeCollection(np.array([[-400, 100],  [400, -100]]), [0, TMAX]),
        NodeCollection(np.array([[-2000, 0], [2000, 0]]), [0, TMAX]),
        NodeCollection(np.array([[0, 2000], [0, -2000]]), [0, TMAX]),
    ]
    node = NodeCollection(np.array(
        [[-400, -100], [-200, 25], [0, 0], [200, -25], [400, 100]]), np.linspace(0, TMAX, 5))
    factory = PathFactory()
    paths = factory.create(nodes, LinearInterpolator)
    paths += factory.create([node], SplineInterpolator)

    initial_states = [
        np.array([-400, 1.55,  -100, 1.95]),
        np.array([-400, 1.5 ,  100, -.5]),
        # np.array([0, 0,  2000, -8.9]),
        # np.array([-2000, 8.9, 0, 0]),
    ]

    nprobes = 200
    probexy = np.zeros(nprobes)
    probetimes = np.linspace(0.0, TMAX-0.1, nprobes)
    radar = RadarGenerator(paths, 0.3, 1)
    clutter_model = PoissonClutter([-500, 400, -250, 250], 0.0003)
    scans = radar.make_scans_series(
        np.array([probetimes, probexy, probexy]).T, clutter_model)

    # tracks = [Track(state, P, H, R) for state in initial_states]
    manager = TrackManager()
    for state in initial_states:
        manager.initialize_track(state)

    nsteps = 60000
    time = 0
    dt = 0.05
    
    working_scans = scans.copy()
    
    for _ in range(nsteps):
        time += dt

        manager.predict_tracks(time, dt)
        
        if time+dt >= working_scans[0].time:
            dt_ = working_scans[0].time - time
            time = working_scans[0].time

            manager.update_tracks(working_scans[0], time)
            

if __name__ == "__main__":
    main()
