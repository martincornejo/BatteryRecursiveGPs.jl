# Online-gp

## Setting

### Kalman filter

A Kalman filter in discrete time solves the problem of estimating an state $x$ of a discrte-time controlled process govenrend by a linear Stochastic difference equation:

$$
x_k \sim \mathcal{N}(\hat{x}_k, P_k)
$$

$$
x_k = A \cdot x_{k -1} + B \cdot u_{k -1} + w_{k -1}, \quad w \sim \mathcal{N}(0, Q)
$$

On the next steps $u_{k-1}$ is set as $0$ for simplicity
And the relation with the measurement $y_k$ as:

$$
y_k = H \cdot x_k + v_k, \quad v \sim \mathcal{N}(0, R)
$$

With this set up, the Kalman filter is solved by two steps:

- Predict step:

$$
\hat{x}_{k|k-1} = A \cdot \hat{x}\_{k|k-1} +  w_k
$$

$$
P_{k | k-1} = A P_{k - 1| k-1} A^T + Q
$$

- Update step:

$$
K_k = P_{k | k-1} H^T(H P_{k | k-1} H^T + R)^{-1}
$$

$$
\hat{x}_{k | k} = \hat{x}\_{k | k-1}+ K_k(y_k - H\hat{x}\_{k | k-1})
$$

$$
P_{k | k} = P_{k | k-1} (I - K_k H)
$$

### Problem setting and notation

In this project a Gaussian Process (GP) is modeled using a set of basis vectors $\{X_b\}$ and updated using a set of data set $\{X_d;Y_d\} = \{x_0, x_1,..., x_k; y_0, y_1,..., x_k\}$.

We define the following GP analysis:

- $g^b_0 = g(X_b) \sim p(g^b)$ prior of $g^b$ as $g(X_b)$.
- $g^t_0 = g(X_t) \sim p(g^t)$ prior of $g^t$ as $g(X_t)$.
- $g^b_k \sim p(g^b | y_{1:k})$ posterior of $g^b$.
- $g^t_k \sim p(g^t | y_{1:k})$ posterior of $g^t$.

With the following distributions:

- $p(g^b)$:

$$
g^b_0 \sim \mathcal{N}(\hat{g}^b_0, P^b_0),\quad \hat{g}^b_0 =m(X_b), P^b_0 = k(X_d, X_d)
$$

- $p(g^b| y_{1:k})$:

$$
g^b_k \sim \mathcal{N}(\hat{g}^b_k, P^b_k)
$$

- $p(g^t| y_{1:k})$:

$$
g^t_k \sim \mathcal{N}(\hat{g}^t_k, P^t_k)
$$

Finally, for the observed values of the GP and the measurement $y_k$ is defined as:

$$
y^t = g^t_k + v, \quad v \sim \mathcal{N}(0, \sigma^2)
$$

With this setting, a Kalman filter update is derived for $g$ with the intermediate variable $g_k$.

## Kalman Filter: Derivation state-observe matrix

For the predict step of the Kalman filter, the posterior $p(g^b, g^t | y_{1:k -1})$ is separated on the joint distribution:

$$
p(g^b, g^t | y_{1:k -1}) = p(g^t | g^b) \cdot p(g^b | y_{1:k-1})
$$

On which $p(g^b |  y_{1:k-1}) = \mathcal{N}(\hat{g}^b_{k-1}, P^b_{k-1})$ by eq. (10).

And $p(g^t| g^b)$ can be obtained by the derivation of the posterior conditioning on $\hat{g}^b_{k-1}$.

$$
p(g^t | g^b) = \frac{p(g^t, g^b)}{p(g^b)}
$$

Which yields:

$$
p(g^t | g^b) = \mathcal{N}(\hat{g}^t_{k - 1}, B_t)
$$

$$
\hat{g}^t_{k - 1} = \hat{g}^k_{0} + H_t \cdot (\hat{g}^b_{k - 1} - \hat{g}^b_{0})
$$

$$
B_t = k(X_t, X_t) - H_t \cdot k(X_b, X_t)
$$

$$
H_t = k(X_t, X_b) \cdot k(X_b, X_b)^{-1}
$$

Substituting both $p(g^b, g^t | y_{1:t -1}) = p(g^t | g^b) \cdot p(g^b|y_{1:k-1})$:

$$
 p(g^t, g^b | y_{1:t -1}) = \mathcal{N}(
          \begin{bmatrix} \hat{g}^b_{k - 1} \\ \hat{g}^k_{k - 1}\end{bmatrix},
          \begin{bmatrix} P^b_{k-1} && P^b_{k-1} H^T_t; \\ \hat H_tP^b_{k-1}  &&  P^t_{k-1} \end{bmatrix}
          )
$$

$$
P^t_{k-1} = B_t + H_t P^b_{k-1}  H^T_t
$$

### Kalman predictor of g

From the derivation above one can infere the prediction step. On which $g^{t}\_{k-1}$ is the equivalent of $x\_{k | k-1}$.

$$
g^{t}_{k-1} = H_t \cdot g^{b}\_{k-1} + w_k
$$

$$
\hat{b_t} = \hat{g}^{t}\_{0} - H_k \cdot \hat{g}^{t}_{0}
$$

Substituting:

$$
y^t = H_k \cdot g^b + w_t + v = H_t \cdot g^b + v_t ; \quad v_t \sim \mathcal{N}(\hat{v_t}, R)
$$

$$
R = k(X_t, X_t) - H_t\cdot k(X, X_t) + I \sigma^2
$$

$$
\hat{v_t} = \hat{b_t}
$$

Which is a Kalman predictor with state-observation matrix $H_k$, and a biased measurement noise $v_t$.

## Kalman filter: Update step

Now, once the problem has been set up as a Kalman filter step we can consider 2 cases:

**Update respect $g_t$ predicted value:**

- Predict step/Motion model as:

$$
g^t_{k-1} = H_t \cdot g^b_{k-1} + w_t, \quad w_t \sim \mathcal{N}(\hat{b_t}, B_t)
$$
    
- Measurement model

$$
y^t = g^t_{k-1} + v
$$

**Update respect $g$ following**
For $g^b$, there is no predict step since is the lattent function, and we can define the measurement  model for the update step as:

$$
y_k = H_k \cdot g^b_{k-1} + v_t
$$

### Kalman filter: Update step respect $g^t$

Following above, for $g^t$ one can observe that its observation matrix $H$ is the identity and measurement noise $R^t = I\sigma^2$. Thus:

- Gain Matrix $G^t_k$

$$
G^t_k = P^t_{k - 1} I  ( I P^b_{k - 1}  I^T + I\sigma^2)^{-1} = P^{k - 1} ( P^t_{k - 1}  + I\sigma^2)^{-1}
$$

- Update step:

$$
  \hat{g}^{t}_{k} = \hat{g}^{t}\_{k - 1} - G^t_k \cdot (y^t - \hat{g}^{t}\_{k - 1})
$$

$$
  P^t_{k} = P^t_{k - 1} (I - G^t_k)
$$

### Kalman filter: Update step respect $g^b$

For $g^b$, its observation matrix $H$ equals $H_t$ and the measurement noise $R = k(X_t, X_t) - H_t \cdot k(X, X_t) + I \sigma^2$. Thus:

- Gain Matrix $G^b_k$

$$
G^b_k = P^b_{k - 1} H_t  ( H_t P^b_{k - 1}  H^T_t + k(X_t, X_t) - H_t \cdot k(X, X_t) + I \sigma^2)^{-1}
$$

- Now, one can observe that $P^t{k - 1} = H_t \Sigma^b_{k - 1}  H^T_t + k(X_t, X_t) - H_t \cdot k(X, X_t)$:

$$
G^b_k = P^b_{k-1} H_t \cdot (P^t_{k-1} + I \sigma^2)^{-1}
$$

- Update step, note that we must include the bias correction of the noise biased mean $\hat{b}_t$:

$$
\hat{g}^b_k = \hat{g}^b_{k - 1} - G^b_k \cdot (y_k - H_t \hat{g}^{b}_{k - 1} - \hat{b}_t)
$$

$$
P^b_{k} = P^b_{k} \cdot (I - G^b_k H_t) = P^b_{k - 1}  - P^b_{k - 1}G^b_k H_t
$$

- Finally, one can observe that $\hat{g}^{t}_{k - 1} = H_t \hat{g}^{b}\_{k - 1} - \hat{b}\_t$, thus:

$$
\hat{g}^{b}_{k} = \hat{b}\_t - G^b_k \cdot (y_k - \hat{g}^{t}\_{k - 1}  )
$$
