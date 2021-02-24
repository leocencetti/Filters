# file created by Leonardo Cencetti on 11/24/20
from math import sqrt

import numpy as np
from scipy.linalg import cholesky, sqrtm

from .rank_1 import update, downdate
from utils.decorators import is_finite


class SRUKF:
    """
    Square Root Unscented Kalman Filter
    """

    def __init__(self, state_dimension: int, output_dimension: int, alpha: float = 1,
                 beta: float = 2, kappa: float = 0):
        """
        Initializes the filter
        :param int state_dimension: dimension of the state
        :param int output_dimension: dimension of the state output (as in Y = A*X)
        :param float alpha: scaling parameter. Determines the spread of the sigma points around the mean state value.
            It is usually a small positive value. The spread of sigma points is proportional to α.
            Smaller values correspond to sigma points closer to the mean state.
        :param float beta: scaling parameter. Incorporates prior knowledge of the distribution of the state.
            For Gaussian distributions, β = 2 is optimal.
        :param float kappa: scaling parameter, usually set to 0.
            Smaller values correspond to sigma points closer to the mean state.
            The spread is proportional to the square-root of κ.
        """
        # initialize dimensions
        self._state_dimension = state_dimension
        self._output_dimension = output_dimension
        self._L = self._state_dimension
        self._sigma_dimension = 2 * self._L + 1

        # initialize filter state
        self._state_initialized = False
        self.state_mean = None
        self.state_covariance = None

        # initialize transition function
        self._io_initialized = False
        self._output_transition = None
        self._state_transition = None
        self._postprocessing = lambda x: x

        # initialize process Noise
        self.noise_covariance = None

        # compute auxiliary parameters
        self._lambda = alpha ** 2 * (self._L + kappa) - self._L

        # initialize filter weights
        self._m_weights = np.zeros([self._sigma_dimension, 1])
        self._m_weights[0] = self._lambda / (self._L + self._lambda)
        self._m_weights[1:self._sigma_dimension] = 1 / (2 * (self._L + self._lambda))
        self._c_weights = self._m_weights.copy()
        self._c_weights[0] += 1 - alpha ** 2 + beta

    def init_state(self, initial_state_mean: np.ndarray, initial_state_covariance: np.ndarray,
                   noise_covariance: np.ndarray):
        """
        Initializes the filter state
        :param numpy.ndarray initial_state_mean: initial state mean (x0)
        :param numpy.ndarray initial_state_covariance: initial state covariance
        :param numpy.ndarray noise_covariance: process noise covariance
        """
        self.state_mean = initial_state_mean
        self.state_covariance = cholesky(initial_state_covariance, lower=True)
        self.noise_covariance = noise_covariance

        self._state_initialized = True

    def init_io(self, state_transition: callable, output_transition: callable, postprocessing: callable = lambda x: x):
        """
        Initializes the output transition
        :param callable state_transition: state transition function, equivalent to A matrix. Must have the following
            signature:
            ```python
            state_transition(
                delta_t: float,
                current_state: numpy.ndarray[state_dimension, 1]
            ) -> numpy.ndarray[state_dimension, 1]
            ```
        :param callable output_transition: output transition function, equivalent to C matrix. Takes the current state
            as input and returns the current output (Y). Must have the following signature:
            ```python
            output_transition(
                current_state: numpy.ndarray[state_dimension, 1]
            ) -> numpy.ndarray[output_dimension, 1]
            ```
        :param callable postprocessing: postprocessing function to be called after each call to the previous functions
            (e.g. to wrap the values to a specific range). Takes the state as input and returns the processed state.
             Must have the following signature:
            ```python
            postprocessing(
                current_state: numpy.ndarray[state_dimension, 1]
            ) -> numpy.ndarray[state_dimension, 1]
        """
        self._state_transition = state_transition
        self._output_transition = output_transition
        self._postprocessing = postprocessing
        self._io_initialized = True

    def reset(self):
        """
        Resets the filter to its initial conditions
        """
        self._state_initialized = self._io_initialized = False
        self.state_covariance = self.state_mean = None

    def step(self, deltaT: float, measurement_mean: np.ndarray, measurement_covariance: np.ndarray):
        """
        Performs one step (prediction + correction)
        :param float deltaT: time step in seconds
        :param numpy.ndarray measurement_mean: measurement values
        :param numpy.ndarray measurement_covariance: measurement covariance matrix
        :returns: state_estimate, covariance_estimate
        """
        if not self._state_initialized:
            raise ValueError('Initial conditions not initialized. Call init_state(...) before running.')
        if not self._io_initialized:
            raise ValueError('Output transition not initialized. Call init_io(...) before running.')

        # run ukf
        predicted_mean, predicted_covariance, predicted_sigma = self._predict(deltaT, self.state_mean,
                                                                              self.state_covariance)

        self.state_mean, self.state_covariance = self._correct(predicted_mean, predicted_covariance, measurement_mean,
                                                               measurement_covariance, predicted_sigma)

        return self.state_mean, self.state_covariance

    def _predict(self, deltaT: float, mean: np.ndarray, covariance: np.ndarray):
        """
        Performs the prediction step
        :param float deltaT: time step in seconds
        :param numpy.ndarray mean: state of previous step
        :param numpy.ndarray covariance: state covariance of previous step
        :returns: predicted_state, predicted_covariance, predicted_sigma_points
        """
        # compute sigma points
        sigma_points = self._compute_sigma_point(mean, covariance, self._L + self._lambda)
        # time update
        next_sigma_points = np.zeros([self._state_dimension, self._sigma_dimension])
        for ii in range(self._sigma_dimension):
            next_sigma_points[:, ii] = self._state_transition(deltaT, sigma_points[:, ii, None]).squeeze()

        next_state_mean = self._postprocessing(next_sigma_points @ self._m_weights)

        temp = next_sigma_points - next_state_mean
        next_state_covariance = np.linalg.qr(
            np.concatenate([sqrt(self._c_weights.item(1)) * temp[:, 1:], sqrtm(self.noise_covariance)], axis=1).T,
            mode='r')
        next_state_covariance = self.cholupdate(next_state_covariance, temp[:, 0], self._c_weights.item(0))
        return next_state_mean, next_state_covariance, next_sigma_points

    def _correct(self, predicted_state_mean: np.ndarray, predicted_state_covariance: np.ndarray,
                 measurement_mean: np.ndarray, measurement_covariance: np.ndarray, next_sigma_points: np.ndarray):
        """
        Performs the correction step
        :param numpy.ndarray predicted_state_mean: predicted state of the current step
        :param numpy.ndarray predicted_state_covariance: predicted state covariance matrix of current step
        :param numpy.ndarray measurement_mean: new measurement mean
        :param numpy.ndarray measurement_covariance: new measurement covariance matrix
        :returns: state_estimate, state_covariance_estimate
        """

        next_sigma_points = np.concatenate([next_sigma_points,
                                            self._compute_sigma_point(next_sigma_points[:, 0],
                                                                      sqrtm(self.noise_covariance),
                                                                      self._L + self._lambda)[:, 1:]], axis=1)

        output_sigma_points = np.zeros((self._output_dimension, 2 * self._sigma_dimension - 1))
        for ii in range(self._sigma_dimension):
            output_sigma_points[:, ii] = self._output_transition(next_sigma_points[:, ii, None]).squeeze()

        extended_m_weights = np.concatenate([self._m_weights, self._m_weights[1:]], axis=0)
        output_mean = output_sigma_points @ extended_m_weights

        # measurement update
        temp = output_sigma_points - output_mean
        output_covariance = np.linalg.qr(
            np.concatenate([sqrt(self._c_weights.item(1)) * temp[:, 1:], sqrtm(measurement_covariance)], axis=1).T,
            mode='r')
        output_covariance = self.cholupdate(output_covariance, temp[:, 0], self._c_weights.item(0))
        extended_c_weights = np.concatenate([self._c_weights, self._c_weights[1:]], axis=0)
        cross_covariance = np.multiply(next_sigma_points - predicted_state_mean, extended_c_weights.T) @ (
                output_sigma_points - output_mean).T

        if not is_finite(output_covariance, cross_covariance):
            print('Something is not finite!')
        # compute Kalman gain
        kalman_gain = \
            np.linalg.lstsq(output_covariance.T, np.linalg.lstsq(output_covariance, cross_covariance.T, rcond=None)[0],
                            rcond=None)[
                0].T

        state_mean_estimate = self._postprocessing(
            predicted_state_mean + kalman_gain @ (measurement_mean - output_mean))

        U = kalman_gain @ output_covariance

        # Cholesky rank-1 downdate with each column of U
        state_covariance_estimate = predicted_state_covariance.copy()
        for ii in range(U.shape[1]):
            state_covariance_estimate = self.cholupdate(state_covariance_estimate.copy(), U[:, ii].copy(), -1)

        return state_mean_estimate, state_covariance_estimate

    @staticmethod
    def _compute_sigma_point(mean: np.ndarray, covariance: np.ndarray, coefficient: float):
        """
        Computes the sigma point matrix
        :param numpy.ndarray mean: state mean
        :param numpy.ndarray covariance: state covariance
        :param float coefficient: scaling coefficient
        :return: sigma_points
        """
        mean = mean.reshape(-1, 1)
        sigma_points = np.concatenate(
            [mean, mean + sqrt(coefficient) * covariance, mean - sqrt(coefficient) * covariance], axis=1)
        return sigma_points

    @staticmethod
    def cholupdate1(L: np.ndarray, x: np.ndarray, factor=+1):
        """
        Performs a rank-1 update (or downdate)
        :param numpy.ndarray L: original Cholesky factorization of shape (M, M)
        :param numpy.ndarray x: update column vector of shape (M,1) or (M,)
        :param float factor: scaling factor. The sign determines if it's an update (+) or downdate(-)
        :return: updated Cholesky factor
        """
        sign = np.sign(factor)
        factor = sqrt(abs(factor))
        L = L + sign * factor * x @ x.T
        return L

    @staticmethod
    def cholupdate2(L: np.ndarray, x: np.ndarray, factor=+1):
        """
        Performs a rank-1 update (or downdate)
        :param numpy.ndarray L: original Cholesky factorization of shape (M, M)
        :param numpy.ndarray x: update column vector of shape (M,1) or (M,)
        :param float factor: scaling factor. The sign determines if it's an update (+) or downdate(-)
        :return: updated Cholesky factor
        """
        N = len(x)
        sign = np.sign(factor)
        factor = pow(abs(factor), 0.25)
        x *= factor
        for k in range(N):
            r = np.sqrt(L[k, k] ** 2 + sign * x[k] ** 2)
            c = r / L[k, k]
            s = x[k] / L[k, k]
            L[k, k] = r
            if k < N - 1:
                L[k + 1:N - 1, k] = (L[k + 1:N - 1, k] + sign * s * x[k + 1:N - 1]) / c
                x[k + 1:N - 1] = c * x[k + 1:N - 1] - s * L[k + 1:N - 1, k]
        return L

    @staticmethod
    def cholupdate(L: np.ndarray, x: np.ndarray, factor=+1):

        if factor > 0:
            R = update(L, pow(factor, 0.25) * x)
        else:
            R = downdate(L, pow(abs(factor), 0.25) * x)
        return R
