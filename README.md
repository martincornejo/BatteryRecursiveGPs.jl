# Online-gp

### Kalman filter

A Kalman filter in discrete time solves the problem of estimating an state $x$ of a discrete-time controlled process governed by a linear Stochastic difference equation:

$$
x_t \sim \mathcal{N}(\hat{x}_t, P_t)
$$

$$
x_t = A \cdot x_{t -1} + B \cdot u_{t -1} + w_{t -1}, \quad w \sim \mathcal{N}(0, Q)
$$

On the next steps $u_{t-1}$ is set as $0$ for simplicity
And the relation with the measurement $y_k$ as:

$$
y_t = H \cdot x_t + v_t, \quad v \sim \mathcal{N}(0, R)
$$

With this set up, the Kalman filter is solved by two steps:

- Predict step:

$$
\hat{x}_{t|t-1} = A \cdot \hat{x}\_{t|t-1} +  w_t
$$

$$
P_{t | t-1} = A P_{t - 1| t-1} A^T + Q
$$

- Update step:

$$
K_t = P_{t | t-1} H^T(H P_{t | t-1} H^T + R)^{-1}
$$

$$
\hat{x}_{t | t} = \hat{x}\_{t | t-1}+ K_t(y_t - H\hat{x}\_{t | t-1})
$$

$$
P_{t | t} = P_{t | t-1} (I - K_t H)
$$


