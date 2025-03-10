# Online-gp


### Problem setting and notation

In this project a Gaussian Process (GP) is modeled using a set of basis vectors $\{X_b\}$ and updated using a set of data set $\{X_d;Y_d\} = \{x_0, x_1,..., x_k; y_0, y_1,..., x_k\}$.

We define the following GP analysis:

- $g^b_0 = g(X_b) \sim p(g^b)$ prior of $g^b$ as $g(X_b)$.
- $g^t_0 = g(X_t) \sim p(g^t)$ prior of $g^t$ as $g(X_t)$.
- $g^b_t \sim p(g^b | y_{1:t})$ posterior of $g^b$.
- $g^t_t \sim p(g^t | y_{1:t})$ posterior of $g^t$.

With the following distributions:

- $p(g^b)$:

$$
g^b_0 \sim \mathcal{N}(\hat{g}^b_0, P^b_0),\quad \hat{g}^b_0 =m(X_b), P^b_0 = k(X_d, X_d)
$$

- $p(g^b| y_{1:t})$:

$$
g^b_k \sim \mathcal{N}(\hat{g}^b_t, P^b_t)
$$

- $p(g^t| y_{1:t})$:

$$
g^t_k \sim \mathcal{N}(\hat{g}^t_t, P^t_t)
$$

Finally, for the observed values of the GP and the measurement $y_k$ is defined as:

$$
y^t = g^t_t + v, \quad v \sim \mathcal{N}(0, \sigma^2)
$$

With this setting, a Kalman filter update is derived for $g$ with the intermediate variable $g_k$.

## Kalman Filter: Derivation state-observe matrix

For the predict step of the Kalman filter, the posterior $p(g^b, g^t | y_{1:t -1})$ is separated on the joint distribution:

$$
p(g^b, g^t | y_{1:t-1}) = p(g^t | g^b) \cdot p(g^b | y_{1:t-1})
$$

On which $p(g^b |  y_{1:t-1}) = \mathcal{N}(\hat{g}^b_{t-1}, P^b_{t-1})$ by eq. (10).

And $p(g^t| g^b)$ can be obtained by the derivation of the posterior conditioning on $\hat{g}^b_{t-1}$.

$$
p(g^t | g^b) = \frac{p(g^t, g^b)}{p(g^b)}
$$

Which yields:

$$
p(g^t | g^b) = \mathcal{N}(\hat{g}^t_{t - 1}, B_t)
$$

$$
\hat{g}^t_{t - 1} = \hat{g}^t_{0} + H_t \cdot (\hat{g}^b_{t - 1} - \hat{g}^b_{0})
$$

$$
B_t = k(X_t, X_t) - H_t \cdot k(X_b, X_t)
$$

$$
H_t = k(X_t, X_b) \cdot k(X_b, X_b)^{-1}
$$

Substituting both $p(g^b, g^t | y_{1:t -1}) = p(g^t | g^b) \cdot p(g^b|y_{1:t-1})$:

$$
 p(g^t, g^b | y_{1:t -1}) = \mathcal{N}(
          \begin{bmatrix} \hat{g}^b_{t - 1} \\ \hat{g}^k_{t - 1}\end{bmatrix},
          \begin{bmatrix} P^b_{t-1} && P^b_{t-1} H^T_t; \\ \hat H_tP^b_{t-1}  &&  P^t_{t-1} \end{bmatrix}
          )
$$

$$
P^t_{t-1} = B_t + H_t P^b_{t-1}  H^T_t
$$

### Kalman predictor of g

From the derivation above one can infer the prediction step. On which $g^{t}\_{t-1}$ is the equivalent of $x\_{t | t-1}$.

$$
g^{t}_{t-1} = H_t \cdot g^{b}\_{t-1} + w_k
$$

$$
\hat{b_t} = \hat{g}^{t}\_{0} - H_t \cdot \hat{g}^{t}_{0}
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
g^t_{t-1} = H_t \cdot g^b_{t - 1} + w_t, \quad w_t \sim \mathcal{N}(\hat{b_t}, B_t)
$$
    
- Measurement model

$$
y^t = g^t_{t-1} + v
$$

**Update respect $g$ following**
For $g^b$, there is no predict step since is the latent function, and we can define the measurement  model for the update step as:

$$
y_t = H_t \cdot g^b_{t-1} + v_t
$$

### Kalman filter: Update step respect $g^t$

Following above, for $g^t$ one can observe that its observation matrix $H$ is the identity and measurement noise $R^t = I\sigma^2$. Thus:

- Gain Matrix $G^t_k$

$$
G^t_k = P^t_{t - 1} I  ( I P^b_{t - 1}  I^T + I\sigma^2)^{-1} = P^{t - 1} ( P^t_{t - 1}  + I\sigma^2)^{-1}
$$

- Update step:

$$
  \hat{g}^{t}_{t} = \hat{g}^{t}\_{t - 1} + G^t_t \cdot (y^t - \hat{g}^{t}\_{t - 1})
$$

$$
  P^t_{t} = P^t_{t - 1} (I - G^t_t)
$$
### Kalman filter: Update step by Kalman filter set-up of $g^b$ 

For $g^b$, its observation matrix $H$ equals $H_t$ and the measurement noise $R = k(X_t, X_t) - H_t \cdot k(X, X_t) + I \sigma^2$. Thus:

- Gain Matrix $G^b_k$

$$
G^b_t = P^b_{t - 1} H_t  ( H_t P^b_{t - 1}  H^T_t + k(X_t, X_t) - H_t \cdot k(X, X_t) + I \sigma^2)^{-1}
$$

- Now, one can observe that $P^t{k - 1} = H_t P^b_{t - 1}  H^T_t + k(X_t, X_t) - H_t \cdot k(X, X_t)$:

$$
G^b_t = P^b_{t-1} H_t \cdot (P^t_{t-1} + I \sigma^2)^{-1}
$$

- Update step, note that we must include the bias correction of the noise biased mean $\hat{b}_t$:

$$
\hat{g}^b_k = \hat{g}^b_{t - 1} + G^b_k \cdot (y_k - H_t \hat{g}^{b}_{t - 1} - \hat{b}_t)
$$

$$
P^b_{t} = P^b_{t-1} \cdot (I - G^b_t H_t) = P^b_{t - 1}  - P^b_{t - 1}G^b_t H_t
$$

- Finally, one can observe that $\hat{g}^{t}_{t - 1} = H_t \hat{g}^{b}\_{t - 1} - \hat{b}\_t$, thus:

$$
\hat{g}^{b}_{t} = \hat{g}^b\_{t-1} + G^b_t \cdot (y_t - \hat{g}^{t}\_{t - 1}  )
$$




