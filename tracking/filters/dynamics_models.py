from abc import ABC, abstractmethod

import numpy as np


class DynamicsModel(ABC):

    # @property
    # @abstractmethod
    # def dim(self):
    #     pass

    @abstractmethod
    def F(self, dt):
        pass

    @abstractmethod
    def Q(self, dt):
        pass    

    @abstractmethod
    def H(self):
        pass

    @abstractmethod
    def R(self, sigma: list[int]):
        pass

    # @abstractmethod
    # def dot_indices(self, dot: int):
    #     pass

    @abstractmethod
    def cov_one_point_init(self, vmax: float):
        pass
    
    @property
    @abstractmethod
    def pos_indices(self):
        pass
    
    @property
    @abstractmethod
    def vel_indices(self):
        pass


class DefaultDynamicsPV2D(DynamicsModel):

    def F(self, dt):
        F1 = np.array([[1, dt], [0, 1]])
        return np.kron(np.eye(2), F1)
        
    def Q(self, dt):
        v = np.array([dt**2 / 2, dt]).reshape(2, 1)
        Q1 = v.dot(v.T)
        return np.kron(np.eye(2), Q1)
    
    def H(self):
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    def R(self, sigmas: list[int]):
        return None if len(sigmas) < 2 else np.diag([sigmas[0]**2, sigmas[1]**2])

    def pos_one_point_init(self, pos: tuple):
        return np.array([pos[0], 0.0, pos[1], 0.0])

    def cov_one_point_init(self, vmax: float):
        return np.diag([1.0, vmax**2/2.0, 1.0, vmax**2/2.0])
        
    @property
    def pos_indices(self):
        return (0, 2)
    
    @property
    def vel_indices(self):
        return (1, 3)
    
    
class DefaultDynamicsPVA2D(DynamicsModel):

    def F(self, dt):
        F1 = np.array([[1, dt, dt**2], [0, 1, dt], [0, 0, 1]])
        return np.kron(np.eye(2), F1)
    
    def Q(self, dt):
        v = np.array([dt**2 / 2, dt, 1]).reshape(3, 1)
        Q1 = v.dot(v.T)
        return np.kron(np.eye(2), Q1)
    
    def H(self):
        return np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    
    def R(self, sigmas: list[int]):
        return None if len(sigmas) < 3 else np.diag([sigmas[0]**2, sigmas[1]**2, sigmas[2]**2])
    
    def pos_one_point_init(self, pos: tuple):
        return np.array([pos[0], 0.0, 0.0, pos[1], 0.0, 0.0])
    
    def cov_one_point_init(self, vmax: float):
        return np.diag([1.0, vmax**2/3.0, vmax**2/3.0, 1.0, vmax**2/3.0, vmax**2/3.0])
    
    @property
    def pos_indices(self):
        return (0, 3)
    
    @property
    def vel_indices(self):
        return (1, 4)
