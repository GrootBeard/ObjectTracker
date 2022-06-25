import numpy as np


class KalmanFilter:

    # x0 is the initial state vector
    # sigma_p is the process noise standard deviation
    def __init__(self, x0: np.array,
                 sigma_p: float) -> None:
        self.x_ = x0
        self.sigma_p = sigma_p
        self.P = np.eye(len(self.x_))

    def predict(self, dt: float):
        # F = np.array([[1, dt], [0, 1]])
        # G = np.array([dt**2 / 2, dt]).reshape((2, 1))

        F1 = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
        F = np.kron(np.eye(2), F1)
        # print(F)

        v = np.array([dt**2/2, dt, 1]).reshape(3, 1)
        Q1 = v.dot(v.T)
        # print(Q1)
        Q = np.kron(np.eye(2), Q1)

        x_new_ = F.dot(self.x_)
        P_new = F.dot(self.cov).dot(F.T) + self.sigma_p * Q

        self.x_ = x_new_
        self.P = P_new

    def update(self, z: np.array, sigma_mp: float, sigma_mvr: float):
        H = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        y = z - H.dot(self.x_)
        # S = H.dot(self.cov).dot(H.T) + np.array([sigma**2])
        S = H.dot(self.cov).dot(H.T) + np.array([sigma_mp**2, sigma_mp**2])
        K = self.cov.dot(H.T).dot(np.linalg.inv(S))

        new_x_ = self.x_ + K.dot(y)
        new_P = (np.eye(6) - K.dot(H)).dot(self.cov)

        self.x_ = new_x_
        self.P = new_P

    @property
    def cov(self):
        return self.P

    @property
    def prediction(self):
        return self.x_
