# Bayesian Neural Networks for economic regression

Bayesian neural networks (BNNs) predict well when the functional form is
unknown, but economists also need interpretable effects with uncertainty.
This repository explores two ways to get both, first on simulated data where
the true model is known and then on real wage data:

* a **partially linear BNN**, $Y = \alpha + \beta X_0 + f(X_{-0}) + \varepsilon$,
  which keeps a regression coefficient for the variable of interest and lets
  a network absorb non-linear controls;
* a **nested BNN**, which combines a ReLU and a tanh network and learns, for
  each regressor, which of the two should model it.

All models are written in [PyMC](https://www.pymc.io) and fitted by
mini-batch variational inference (ADVI).

## Notebooks

| Notebook | Content |
|---|---|
| [`Bayesian_NN.ipynb`](Bayesian_NN.ipynb) | Linear regression vs BNN vs partially linear BNN; recovery of a coefficient with non-linear controls; learned partial-dependence curves. |
| [`Nested-BayesianNN.ipynb`](Nested-BayesianNN.ipynb) | Nested BNN with per-variable model selection on a strongly non-linear, confounded process; comparison with OLS and a standard BNN; marginal effects with 90% credible bands. |
| [`BNN_returns_to_education.ipynb`](BNN_returns_to_education.ipynb) | **Real data** (CPS 2004, 61,395 full-time workers): is the return to a year of schooling constant? Marginal return to each year, with heterogeneity by gender and age, compared with the Mincer equation and an OLS regression with one dummy per education level; nested-BNN and multi-seed robustness check. |
| [`BNN_returns_to_education_card.ipynb`](BNN_returns_to_education_card.ipynb) | **Real data** (Card 1995, 3,010 young men, with family background, IQ and an instrument): sheepskin effects, ability bias, IV, and a double-machine-learning cross-check of the BNN estimates. |
| `BNN_gender_gap*.ipynb` | Application of the nested BNN to the gender wage gap (data not included in the repository). |

## Main results (simulated data, out of sample)

| | `Bayesian_NN` R² | `Nested-BayesianNN` R² |
|---|---|---|
| OLS | 0.840 | 0.631 |
| BNN | 0.991 | 0.989 |
| Partially linear BNN | 0.993 | – |
| Nested BNN | – | 0.997 |

* With non-linear controls handled by the network, the partially linear BNN
  estimates the coefficient of interest at 3.75 (90% interval 3.51–3.98),
  against a true predictive slope of 3.69. OLS, which is misspecified,
  gives 4.29.
* The nested BNN routes the cubic regressor that dominates the outcome to
  the unbounded ReLU network and the others to the tanh network, and halves
  the RMSE of a standard BNN.
* Both notebooks also show the limits: neither model removes omitted-variable
  bias, and effects that are small relative to the outcome are shrunk towards
  zero.

## Returns to schooling (real data)

* The return per year of schooling is far from constant. It peaks at the
  completion of high school and the first year of college (13–20% per
  year) and falls to about 5–8% per year for graduate degrees. The Mincer
  equation estimates a constant 9.3%.
* Below 12 years the profile is step-shaped (a sheepskin effect at the
  diploma); above 13 years, returns diminish with each further degree.
* The BNN finds, without being told to look for it, that college pays more
  for women (12.1% vs 8.8% per year between 12 and 16 years of schooling).
* With flexible controls, the average return (9.5%) is close to Mincer's.
* The nested BNN fits no better than the standard BNN on these data. Refitting
  with five seeds shows that single-run ADVI intervals are about half as
  wide as they should be; the main findings survive.

## Returns to schooling with rich controls (Card data)

* With schooling measured in actual years, the payoff is concentrated in
  degree years (12th, 16th, 18th): clear sheepskin effects.
* IQ and test scores explain about a fifth of the OLS return (ability bias),
  a result that holds with OLS, double machine learning and the BNN.
* On this small sample the BNNs under-fit and bias the return downwards
  (5.6% against 7.1–7.4% for OLS and double ML). The nested BNN is unstable
  across seeds. The notebook documents both: it is a useful counter-example
  to the CPS results.

## Code

[`bnn_models.py`](bnn_models.py) contains the model builders (linear, BNN,
partially linear BNN, nested BNN) and helpers for fitting, prediction and
partial dependence. Two choices matter in practice:

* **Prior scale.** Weight priors are $N(0, 1/n_{\text{in}})$. With
  unit-variance priors the output scale grows with every layer, and the
  networks fail to fit.
* **Initial variational scale.** ADVI starts from a small posterior standard
  deviation (0.05). With PyMC's default, the nested network never starts
  learning.

## Running

```bash
pip install -r requirements.txt
jupyter lab
```

Each notebook also has an *Open in Colab* badge. On Colab, the first cell
installs PyMC and downloads `bnn_models.py`. A full run takes a few minutes
on a laptop CPU.
