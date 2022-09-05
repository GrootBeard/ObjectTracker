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
        NodeCollection(np.array([[-400, 100],  [400, -100]]), [0, TMAX]),
        NodeCollection(np.array([[-2000, 0], [2000, 0]]), [0, TMAX]),
        NodeCollection(np.array([[0, 2000], [0, -2000]]), [0, TMAX]),
    ]
    node = NodeCollection(np.array(
        [[-400, -100], [-200, 25], [0, 0], [200, -25], [400, 100]]), np.linspace(0, TMAX, 5))

    nprobes = 200
    probexy = np.zeros(nprobes)
    probetimes = np.linspace(0.0, TMAX-0.1, nprobes)
    radar = RadarGenerator(paths, 0.3, 1)
    clutter_model = PoissonClutter([-500, 400, -250, 250], 0.00005)
    scans = radar.make_scans_series(
        np.array([probetimes, probexy, probexy]).T, clutter_model)

    manager = TrackManager()

    nsteps = 60000
    time = 0
    dt = 0.05
    
    working_scans = scans.copy()
    
    for _ in range(nsteps):
        time += dt

        manager.predict_tracks(time, dt)
        
        if time+dt >= working_scans[0].time:
            time = working_scans[0].time

            manager.update_tracks(working_scans[0], time)
            manager.one_point_init(vmax=5, p0=0.8)
            manager.delete_false_tracks(.05)
            
            working_scans.pop(0)
            if len(working_scans) == 0:
                print("Processed all scans")
                break
        
    renderer = TrackingVisualizer(manager.logger)
    plt.ion()
    plt.figure()
    plot = plt.subplot(1, 2, 1)
     
    renderer.set_number_rendered_epochs(150)
    for i in range(145):
        renderer.advance_epoch()
    renderer.filter_tracks_by_length(7)
    renderer.render(plot)

    plt.show(block=True)
    

if __name__ == "__main__":
    main()
