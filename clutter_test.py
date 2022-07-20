
import matplotlib.pyplot as plt
from tracking.util.metrics import PoissonClutter

def main():
    clutter_generator = PoissonClutter([-1, 1, -1, 1], 0.5, seed=1)
    clutter = clutter_generator.generate_clutter() 

    plt.scatter(clutter[0], clutter[1])
    plt.show()

if __name__ == "__main__":
    main()
