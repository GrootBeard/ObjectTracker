import numpy as np


class KalmanFilter:

    def __init__(self, x0: np.array,
                 sigma_p: float) -> None:
        self.x_ = x0
        self.sigma_p = sigma_p
        self.P = np.eye(len(self.x_))

    def predict(self, dt: float):
        F1 = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
        F = np.kron(np.eye(2), F1)

        v = np.array([dt**2 / 2, dt, 1]).reshape(3, 1)
        Q1 = v.dot(v.T)
        Q = np.kron(np.eye(2), Q1) * self.sigma_p**2

        self.x_ = F.dot(self.x_)
        self.P = F.dot(self.cov).dot(F.T) + Q

    def update(self, z: np.array, sigma_x: float, sigma_y: float):
        H = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        y = z - H.dot(self.x_)
        R = np.diag([sigma_x**2, sigma_y**2])

        S = H.dot(self.cov).dot(H.T) + R

        gate = y.T.dot(np.linalg.inv(S)).dot(y)
        if gate > 50:

            print('gate: ', gate)
            return
        K = self.cov.dot(H.T).dot(np.linalg.inv(S))
        D = (np.eye(6) - K.dot(H))

        new_x_ = self.x_ + K.dot(y)
        new_P = D.dot(self.cov).dot(D.T) + K.dot(R).dot(K.T)

        self.x_ = new_x_
        self.P = new_P

    @property
    def cov(self):
        return self.P

    @property
    def prediction(self):
        return self.x_
