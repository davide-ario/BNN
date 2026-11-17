"""Bayesian regression models used in the notebooks of this repository.

All models are written in PyMC and fitted with mean-field ADVI
(automatic differentiation variational inference). Every builder has the same signature::

    build_xxx(X, y, batch_size=None, seed=None) -> pm.Model

* ``batch_size=None``  -> data are wrapped in ``pm.Data`` and the model exposes a
  deterministic ``mu`` (the conditional mean). Use this to predict.
* ``batch_size=int``   -> data are fed through ``pm.Minibatch`` and the
  likelihood is rescaled with ``total_size``. Use this to fit.

Because parameter names are identical in both versions, a posterior obtained
from the mini-batch model can be plugged into the full-data model to compute
predictions on any new ``X`` (see :func:`predict_mu`).
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _data(X, y, batch_size):
    """Return (X, y, total_size), either as mini-batch tensors or pm.Data."""
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if batch_size is not None:
        X_mb, y_mb = pm.Minibatch(X, y, batch_size=batch_size)
        return X_mb, y_mb, X.shape[0]
    X_d = pm.Data("X", X)
    y_d = pm.Data("y", y)
    return X_d, y_d, None


def _likelihood(mu, y, total_size):
    sigma = pm.HalfNormal("sigma", 1.0)
    if total_size is None:
        mu = pm.Deterministic("mu", mu)
    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, total_size=total_size,
              shape=mu.shape)


def _dense(name, inputs, n_in, n_out, rng, activation=None):
    """Fully connected layer with Gaussian priors on weights and biases.

    The prior scale of the weights is ``1 / sqrt(n_in)``. With unit-scale
    priors the variance of the network output would grow by a factor ``n_in``
    at every layer, so that a priori the network predicts values orders of
    magnitude larger than the (standardised) outcome. Weights are initialised
    at random draws from the prior to break the symmetry between hidden units.
    """
    scale = 1.0 / np.sqrt(n_in)
    w = pm.Normal(f"w_{name}", 0.0, scale, shape=(n_in, n_out),
                  initval=scale * rng.standard_normal((n_in, n_out)))
    b = pm.Normal(f"b_{name}", 0.0, 1.0, shape=(n_out,),
                  initval=0.1 * rng.standard_normal(n_out))
    z = pt.dot(inputs, w) + b
    return activation(z) if activation is not None else z


def _mlp(prefix, inputs, n_in, n_hidden, n_layers, activation, rng):
    """Stack of ``n_layers`` hidden layers followed by a linear output unit."""
    h, width = inputs, n_in
    for i in range(1, n_layers + 1):
        h = _dense(f"{prefix}{i}", h, width, n_hidden, rng, activation)
        width = n_hidden
    scale = 1.0 / np.sqrt(n_hidden)
    w_out = pm.Normal(f"w_{prefix}out", 0.0, scale, shape=(n_hidden,),
                      initval=scale * rng.standard_normal(n_hidden))
    return pt.dot(h, w_out)


relu = pt.nnet.relu if hasattr(pt, "nnet") else (lambda z: pt.maximum(z, 0.0))

# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #


def build_linear(X, y, batch_size=None, seed=None):
    """Bayesian linear regression: y = alpha + X @ beta + e."""
    k = np.shape(X)[1]
    with pm.Model() as model:
        X_, y_, n = _data(X, y, batch_size)
        alpha = pm.Normal("alpha", 0.0, 10.0)
        beta = pm.Normal("beta", 0.0, 10.0, shape=k)
        _likelihood(alpha + pt.dot(X_, beta), y_, n)
    return model


def build_bnn(X, y, batch_size=None, seed=None, n_hidden=30, n_layers=3):
    """Bayesian neural network with ReLU hidden layers and a linear output."""
    rng = np.random.default_rng(seed)
    k = np.shape(X)[1]
    with pm.Model() as model:
        X_, y_, n = _data(X, y, batch_size)
        alpha = pm.Normal("alpha", 0.0, 10.0)
        f = _mlp("h", X_, k, n_hidden, n_layers, relu, rng)
        _likelihood(alpha + f, y_, n)
    return model


def build_partially_linear_bnn(X, y, batch_size=None, seed=None, linear_cols=(0,),
                               n_hidden=30, n_layers=3):
    """Partially linear BNN: y = alpha + X_a @ beta + f(X_b) + e.

    ``X_a`` are the columns in ``linear_cols`` and ``X_b`` all the others. The
    network ``f`` only sees ``X_b``: if it also received ``X_a`` it could
    absorb their effect, and ``beta`` would not be identified.
    """
    rng = np.random.default_rng(seed)
    k = np.shape(X)[1]
    lin = list(linear_cols)
    rest = [j for j in range(k) if j not in lin]
    with pm.Model() as model:
        X_, y_, n = _data(X, y, batch_size)
        alpha = pm.Normal("alpha", 0.0, 10.0)
        beta = pm.Normal("beta", 0.0, 10.0, shape=len(lin))
        f = _mlp("h", X_[:, rest], len(rest), n_hidden, n_layers, relu, rng)
        _likelihood(alpha + pt.dot(X_[:, lin], beta) + f, y_, n)
    return model


def build_nested_bnn(X, y, batch_size=None, seed=None, n_hidden=10, n_layers=3):
    """Nested BNN with soft, per-variable model selection.

    Each regressor j is split between two sub-networks by a weight
    ``model_sel[j]`` in (0, 1):

    * a ReLU network receives ``X * (1 - model_sel)``;
    * a tanh network with a sigmoid output receives ``X * model_sel``.

    The posterior of ``model_sel`` shows which sub-network each variable is
    routed to.
    """
    rng = np.random.default_rng(seed)
    k = np.shape(X)[1]
    with pm.Model() as model:
        X_, y_, n = _data(X, y, batch_size)
        model_sel = pm.Uniform("model_sel", 0.0, 1.0, shape=k)
        alpha = pm.Normal("alpha", 0.0, 10.0)
        f_relu = _mlp("r", X_ * (1.0 - model_sel), k, n_hidden, n_layers, relu, rng)
        f_tanh = pm.math.sigmoid(
            _mlp("t", X_ * model_sel, k, n_hidden, n_layers, pt.tanh, rng))
        _likelihood(alpha + f_relu + f_tanh, y_, n)
    return model


# --------------------------------------------------------------------------- #
# Fitting and prediction                                                      #
# --------------------------------------------------------------------------- #


def fit_advi(builder, X, y, n_iter, batch_size=100, learning_rate=0.01,
             start_sigma=0.05, draws=2000, seed=0, **kwargs):
    """Fit ``builder`` by mini-batch ADVI with the Adam optimiser.

    At every iteration the gradient is computed on a random batch of
    ``batch_size`` observations, with the likelihood rescaled to the full
    sample size.

    ``start_sigma`` is the initial standard deviation of every variational
    factor (on the unconstrained scale). With PyMC's default (about 0.7) the
    random weights inject so much noise into the forward pass that the
    networks, and in particular the nested model, where ``model_sel``
    multiplies the inputs, get stuck predicting a constant.

    Returns ``(idata, loss_history)``, where ``idata`` holds ``draws`` samples
    from the fitted variational posterior.
    """
    model = builder(X, y, batch_size=batch_size, seed=seed, **kwargs)
    with model:
        sigma0 = {v.name: start_sigma for v in model.value_vars}
        approx = pm.fit(n_iter, method="advi", start_sigma=sigma0,
                        obj_optimizer=pm.adam(learning_rate=learning_rate),
                        random_seed=seed, progressbar=False)
    return approx.sample(draws, random_seed=seed), approx.hist


def predict_mu(builder, idata, X, seed=0, **kwargs):
    """Posterior draws of the conditional mean E[y | X], shape (draws, n)."""
    X = np.asarray(X, dtype="float64")
    model = builder(X, np.zeros(X.shape[0]), batch_size=None, seed=seed, **kwargs)
    with model:
        pp = pm.sample_posterior_predictive(idata, var_names=["mu"],
                                            random_seed=seed, progressbar=False)
    mu = pp.posterior_predictive["mu"]
    return mu.stack(sample=("chain", "draw")).transpose("sample", ...).values


def partial_dependence(builder, idata, X_ref, j, grid, center=0.0, transform=None,
                       **kwargs):
    """Posterior draws of the partial dependence of E[y | X] on column ``j``.

    For each value ``g`` in ``grid`` the column is set to ``g`` for every row
    of ``X_ref`` (other columns unchanged) and predictions are averaged over
    rows. The curve is centred so that it equals zero at ``X_j = center``.

    ``transform`` maps raw regressors to the scale the model was fitted on
    (e.g. ``StandardScaler().transform``). Returns an array of shape
    (draws, len(grid)) on the scale of the model's outcome.
    """
    X_ref = np.asarray(X_ref, dtype="float64")
    values = np.append(grid, center)
    X_stack = np.repeat(X_ref[None], len(values), axis=0)
    X_stack[:, :, j] = values[:, None]
    X_flat = X_stack.reshape(-1, X_ref.shape[1])
    if transform is not None:
        X_flat = transform(X_flat)
    mu = predict_mu(builder, idata, X_flat, **kwargs)
    curve = mu.reshape(mu.shape[0], len(values), -1).mean(axis=2)
    return curve[:, :-1] - curve[:, -1:]
