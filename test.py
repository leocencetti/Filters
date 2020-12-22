# File created by Leonardo Cencetti on 12/17/20
from filters import UKF, SRUKF
import numpy as np
import models.old.ctrv_model as ctrv

if __name__ == '__main__':

    # Initialization
    state_dim = 6  # number of state
    output_dim = 3  # number of outputs

    kf = SRUKF(state_dim, output_dim)

    initialState = np.array([[10, 12, 0, 0, 1.5, 0.35]]).T  # initial state
    initialCovariance = 1e-4 * np.eye(state_dim)  # initial state covariance
    noiseCovariance = 1e-6 * np.eye(state_dim)  # process noise

    gt_0 = np.array([[1, 1, 0, 1, 0.5, 0.05]]).T

    r = 1e-4  # std of measurement noise
    measurementNoise = r ** 2 * np.eye(output_dim)  # covariance of measurement noise

    N = 15000  # number of steps
    dt = 1  # sampling time

    # store states
    X_est = np.zeros([state_dim, N])
    X_gt = np.zeros([state_dim, N])
    X_gt[:, 0] = gt_0.squeeze()

    # GroundTruth data generation and UKF estimation
    X_est[:, 0] = initialState.squeeze()
    kf.init_state(initialState, initialCovariance, noiseCovariance)
    kf.init_io(ctrv.ctrv, ctrv.ctrv_output_transition, ctrv.wrapState)

    for i in range(1, N):
        X_gt[:, i] = ctrv.ctrv(dt, X_gt[:, i - 1, None], True).squeeze()
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