import sys
from typing import List

import numpy as np


def check_python_version():
    """
    Check if the python version is 3.6 or higher.
    """
    if sys.version_info < (3, 6):
        raise Exception("Python version must be 3.6 or higher")


def is_invertible(matrix: np.ndarray):
    """
    Check if a matrix is invertible.

    :param matrix: The matrix to check.
    :return: True if invertible, False otherwise.
    """
    try:
        np.linalg.inv(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


class KalmanFilter:
    def __init__(self, A: np.ndarray, B: np.ndarray, H: np.ndarray, x: np.ndarray, P: np.ndarray, Q: np.ndarray,
                 R: np.ndarray):
        """
        Initialize the kalman filter.

        :param A: The state transition matrix (n x n).
        :param B: The control input matrix (n x m).
        :param H: The measurement function matrix (m x n).
        :param x: The initial state estimate (n x 1).
        :param P: The initial estimate covariance (n x n).
        :param Q: The process noise covariance (n x n).
        :param R: The measurement noise covariance (m x m).
        """
        check_python_version()
        self.A = A
        self.B = B
        self.H = H
        self.x = x
        self.P = P
        self.Q = Q
        self.R = R

    def predict(self, u: np.ndarray = None) -> np.ndarray:
        """
        Predict the next state based on the current state and control input.

        :param u: The control input. Defaults to None (m x 1).
        :return: The predicted state (n x 1).
        """
        if u is None:
            self.x = self.A @ self.x
        else:
            self.x = self.A @ self.x + self.B @ u
        self.P = (self.A @ self.P) @ self.A.T + self.Q
        return self.x

    def update(self, measurements: List[np.ndarray], u: np.ndarray = None):
        """
        Update the Kalman filter using the latest measurements from all sensors.

        :param u: The control input. Defaults to None (m x 1).
        :param measurements: List of ndarrays, where each ndarray represents the measurement from one sensor.
        """
        for z in measurements:
            x_pred = self.A @ self.x + self.B @ u
            P_pred = (self.A @ self.P) @ self.A.T + self.Q
            y = z - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            self.x = x_pred + K @ y
            self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ P_pred
            print("x after update: ", self.x)

    def validate_input(self):
        """
        Validate the input matrices and vectors.
        """
        self.__check_matrix_shape()
        self.__check_invertibility()

    def __is_numpy_array(self):
        """
        Check if all inputs are numpy arrays.
        """
        if not all(isinstance(x, np.ndarray) for x in [self.A, self.B, self.H, self.x, self.P, self.Q, self.R]):
            raise Exception("All inputs must be numpy arrays.")

    def __check_matrix_shape(self):
        """
        Check if the input matrices and vectors have the correct shape.
        """
        if self.A.ndim != 2 or self.B.ndim != 2 or self.H.ndim != 2:
            raise Exception("A, B, H must be 2-dimensional numpy arrays.")
        n = self.A.shape[0]
        m = self.B.shape[1]
        p = self.H.shape[0]
        if self.A.shape[1] != n:
            raise Exception("A must be a square matrix (n x n).")
        if self.B.shape[0] != n:
            raise Exception("B must have the same number of rows as A (n x m).")
        if self.H.shape[1] != n:
            raise Exception("H must have the same number of columns as A (m x n).")

        if self.x.ndim != 2 or self.x.shape[0] != n or self.x.shape[1] != 1:
            raise Exception("x must be a column vector with the same number of rows as A (n x 1).")
        if self.P.ndim != 2 or self.P.shape[0] != n or self.P.shape[1] != n:
            raise Exception("P must be a square matrix with the same number of rows as A (n x n).")

        if self.Q.ndim != 2 or self.Q.shape[0] != n or self.Q.shape[1] != n:
            raise Exception("Q must be a square matrix with the same number of rows as A (n x n).")
        if self.R.ndim != 2 or self.R.shape[0] != p or self.R.shape[1] != p:
            raise Exception("R must be a square matrix with the same number of rows as H (p x p).")

    def __check_invertibility(self):
        """
        Check if the input matrices are invertible.
        """
        if not is_invertible(self.A):
            raise ValueError("State transition matrix A must be invertible.")
        if not is_invertible(self.R):
            raise ValueError("Measurement noise covariance matrix R must be invertible.")
        if not is_invertible(self.Q):
            raise ValueError("Process noise covariance matrix Q must be invertible.")
