# file created by Leonardo Cencetti on 11/24/20
import numpy as np
from scipy.linalg import block_diag, sqrtm

import models.old.actrv_model as actrv


class AUKF:
    """
    Augmented Kalman Filter implementation
    """

    def __init__(self, state_dimension: int, noise_dimension: int, output_dimension: int, alpha: float = 1,
                 beta: float = 2, kappa: float = 0):
        """
        Initializes the filter
        :param int state_dimension: dimension of the state
        :param int noise_dimension: dimension of the process noise
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
        self._noise_dimension = noise_dimension
        self._output_dimension = output_dimension
        self._L = self._state_dimension + self._noise_dimension
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

        # initialize noise
        self.noise_mean = None
        self.noise_covariance = None

        # initialize sigma dimension
        self.aug_state_mean = None
        self.aug_state_covariance = None

        # compute auxiliary parameters
        self._lambda = alpha ** 2 * (self._L + kappa) - self._L

        # initialize filter weights
        self._m_weights = np.zeros([self._sigma_dimension, 1])
        self._m_weights[0] = self._lambda / (self._L + self._lambda)
        self._m_weights[1:self._sigma_dimension] = 1 / (2 * (self._L + self._lambda))
        self._c_weights = self._m_weights.copy()
        self._c_weights[0] += 1 - alpha ** 2 + beta

    def init_state(self, initial_state_mean: np.ndarray, initial_state_covariance: np.ndarray,
                   initial_noise_mean: np.ndarray, initial_noise_covariance: np.ndarray):
        """
        Initializes the filter state
        :param numpy.ndarray initial_state_mean: initial state mean (x0)
        :param numpy.ndarray initial_state_covariance: initial state covariance
        :param numpy.ndarray initial_noise_mean: initial process noise mean
        :param numpy.ndarray initial_noise_covariance: initial process noise covariance
        """
        self.state_mean = initial_state_mean
        self.state_covariance = initial_state_covariance
        self.noise_mean = initial_noise_mean
        self.noise_covariance = initial_noise_covariance

        # initialize augmented state
        self.aug_state_mean = np.concatenate((self.state_mean, self.noise_mean))
        self.aug_state_covariance = block_diag(self.state_covariance, self.noise_covariance)

        self._state_initialized = True

    def init_io(self, state_transition: callable, output_transition: callable, postprocessing: callable = lambda x: x):
        """
        Initializes the output transition
        :param callable state_transition: state transition function, equivalent to A matrix. Must have the following
            signature:
            ```python
            state_transition(
                delta_t: float,
                current_augmented_state: numpy.ndarray[augmented_state_dimension, 1]
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
        self.state_covariance = self.noise_mean = self.noise_covariance = \
            self.aug_state_mean = self.aug_state_covariance = None

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

        # run aukf
        predicted_mean, predicted_covariance, predicted_sigma = self._predict(deltaT, self.aug_state_mean,
                                                                              self.aug_state_covariance)

        self.state_mean, self.state_covariance = self._correct(predicted_mean, predicted_covariance, measurement_mean,
                                                               measurement_covariance)
        # update augmented state
        self.aug_state_mean = np.concatenate((self.state_mean, self.noise_mean))
        self.state_covariance = block_diag(self.state_covariance, self.noise_covariance)

        return self.state_mean, self.state_covariance

    def _predict(self, deltaT: float, aug_mean: np.ndarray, aug_covariance: np.ndarray):
        """
        Performs the prediction step
        :param float deltaT: time step in seconds
        :param numpy.ndarray aug_mean: augmented state of previous step
        :param numpy.ndarray aug_covariance: augmented state covariance of previous step
        :returns: predicted_state, predicted_covariance, predicted_sigma_points
        """
        # compute sigma points
        sigma_points = self._compute_sigma_point(aug_mean, aug_covariance, self._L + self._lambda)
        # time update
        next_sigma_points = np.zeros([self._state_dimension, self._sigma_dimension])
        for ii in range(self._sigma_dimension):
            next_sigma_points[:, ii] = self._state_transition(deltaT, sigma_points[:, ii, None]).squeeze()

        next_state_mean = self._postprocessing(next_sigma_points @ self._m_weights)

        temp = next_sigma_points - next_state_mean
        next_state_covariance = np.multiply(temp, self._c_weights.T) @ temp.T
        return next_state_mean, next_state_covariance, next_sigma_points

    def _correct(self, predicted_state_mean: np.ndarray, predicted_state_covariance: np.ndarray,
                measurement_mean: np.ndarray, measurement_covariance: np.ndarray):
        """
        Performs the correction step
        :param numpy.ndarray predicted_state_mean: predicted state of the current step
        :param numpy.ndarray predicted_state_covariance: predicted state covariance matrix of current step
        :param numpy.ndarray measurement_mean: new measurement mean
        :param numpy.ndarray measurement_covariance: new measurement covariance matrix
        :returns: state_estimate, state_covariance_estimate
        """
        # Augment state and covariance
        new_mean = np.concatenate((predicted_state_mean, np.zeros((self._noise_dimension, 1))))
        new_cov = block_diag(predicted_state_covariance, np.eye(self._noise_dimension))
        next_sigma_points = self._compute_sigma_point(new_mean, new_cov, self._L + self._lambda)
        # Slice sigma points (take state rows)
        next_sigma_points = next_sigma_points[:self._state_dimension, :]

        output_sigma_points = np.zeros((self._output_dimension, self._sigma_dimension))

        for ii in range(self._sigma_dimension):
            output_sigma_points[:, ii] = self._output_transition(next_sigma_points[:, ii])

        output_mean = output_sigma_points @ self._m_weights

        # measurement update
        temp = output_sigma_points - output_mean
        output_covariance = np.multiply(temp, self._c_weights.T) @ temp.T + measurement_covariance

        cross_covariance = np.multiply(next_sigma_points - predicted_state_mean, self._c_weights.T) @ (
                output_sigma_points - output_mean).T

        # compute Kalman gain
        kalman_gain = np.linalg.solve(output_covariance.T, cross_covariance.T).T

        state_mean_estimate = self._postprocessing(
            predicted_state_mean + kalman_gain @ (measurement_mean - output_mean))
        state_covariance_estimate = predicted_state_covariance - kalman_gain @ output_covariance @ kalman_gain.T

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
        temp = np.real(sqrtm(np.multiply(covariance, coefficient)))
        mean = mean.reshape(-1, 1)
        sigma_points = np.concatenate([mean, mean + temp, mean - temp], axis=1)
        return sigma_points


if __name__ == '__main__':

    # Initialization
    state_dim = 6  # number of state
    output_dim = 3  # number of outputs
    noise_dim = 3  # number of noise state variables

    kf = AUKF(state_dim, noise_dim, output_dim)

    initialState = np.array([[10, 12, 0, 0, 1.5, 0.35]]).T  # initial state
    initialCovariance = 1e-4 * np.eye(state_dim)  # initial state covariance

    gt_0 = np.array([[1, 1, 0, 1, 0.5, 0.05]]).T

    r = 1e-4  # std of measurement noise
    measurementNoise = r ** 2 * np.eye(output_dim)  # covariance of measurement noise
    processNoise = 1e-5 * np.eye(noise_dim)

    N = 150  # number of steps
    dt = 1  # sampling time

    # store states
    X_est = np.zeros([state_dim, N])
    X_gt = np.zeros([state_dim, N])
    X_gt[:, 0] = gt_0.squeeze()

    # GroundTruth data generation and UKF estimation
    X_est[:, 0] = initialState.squeeze()
    kf.init_state(initialState, initialCovariance, np.zeros([3, 1]), processNoise)
    kf.init_io(actrv.actrv, actrv.actrv_output_transition, actrv.wrapState)

    for i in range(1, N):
        X_gt[:, i] = actrv.actrv(dt, np.concatenate((X_gt[:, i - 1, None], np.array([[0, 0, 0]]).T)), True).squeeze()
        pose_gt = X_gt[:3, i, None]
        pose_noise = r * np.eye(3)

        m_estMean, _ = kf.step(dt, pose_gt, pose_noise)
        X_est[:, i] = m_estMean.squeeze()

    # Compute Root Mean Square Error between estimate and GT
    E = np.squeeze(X_est[:, :]) - X_gt
    SE = E ** 2
    MSE = np.mean(SE, axis=1)
    RMSE = np.sqrt(MSE)
    print('RMSE =', RMSE.squeeze())

    # Plots
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    titles = ['x', 'y', 'theta', 'xDot', 'yDot', 'thetaDot']

    fig_A = make_subplots(
        rows=2, cols=3,
        subplot_titles=titles)

    for i in range(3):
        fig_A.add_trace(go.Scatter(y=X_gt[i, :], name='{}_GT'.format(titles[i])), row=1, col=i + 1)
        fig_A.add_trace(go.Scatter(y=X_est[i, :], name='{}_EST'.format(titles[i])), row=1, col=i + 1)
        fig_A.add_trace(go.Scatter(y=X_gt[i, :], name='{}_GT'.format(titles[i + 3])), row=2, col=i + 1)
        fig_A.add_trace(go.Scatter(y=X_est[i, :], name='{}_EST'.format(titles[i + 3])), row=2, col=i + 1)

    fig_A.show()

    fig_B = go.Figure()
    fig_B.add_trace(go.Scatter(x=X_gt[0, :], y=X_gt[1, :], name='GT'))
    fig_B.add_trace(go.Scatter(x=X_est[0, :], y=X_est[1, :], name='EST'))
    fig_B.update_layout(title_text='Ground truth vs estimated trajectory')
    fig_B.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_B.show()
