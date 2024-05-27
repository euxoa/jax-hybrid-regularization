
# A script that demonstrates JAX proximal gradients on hybrid L1-L2 regularization, 
# and also hyperparameter optimization with bayes_opt.
#
# Sparse matrix support is missing but should be relatively easy to add once the 
# native sparse support of JAX is mature enough.

import os
import numpy as np
# Set this to cpu/gpu if needed. Note that JAX Metal (for Macs) is incomplete and may be unstable.
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
import jaxopt
import bayes_opt # for hyperparameters; only one of available packages


# The logistic model itself. 
# Coefficients beta is assumed to be a triplet of vectors, (X1_coefs, X2_coefs, intercept)
def predict(beta, X1, X2):
    return jax.nn.sigmoid(X1 @ beta[0] + X2 @ beta[1] + beta[2])


# Cost _without_ regularizers, and using a fast and robust implementation of logreg.
# (Customary and quaranteedly consistent would be to call predict() here.)
def cost(beta, X1, X2, y):
    X = jnp.hstack((X1, X2, jnp.ones((X1.shape[0], 1)))) # covariates; intercept as the last
    betac = jnp.concatenate(beta)
    return jaxopt.objective.binary_logreg(betac, (X, y))

# This is a proximity operator, used in optimization to impose regularizers: 
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
# beta is a triplet vector of coefficients, coefficienet sets to be 
# L1-regularized, L2-regularized, and non-regularized, respectively.
def prox_hybrid(beta, l1l2, scaling = 1.0):
    l1reg, l2reg = l1l2
    beta1, beta2, icept = beta
    return (jaxopt.prox.prox_lasso(beta1, l1reg, scaling),
            jaxopt.prox.prox_ridge(beta2, l2reg, scaling),
            icept)

# Proximal gradient optimizer, combines the model and the regularization terms.
pg = jaxopt.ProximalGradient(fun=cost, prox=prox_hybrid)

# Calculates cross-validation cost (globals 'pg' and 'cost'), given two regularization coefficients, 
# covariate matrices (X1 and X2), binary outcomes y, initial params triplet beta0, 
# and number of folds. 
def cv_cost(log_l1, log_l2, X1, X2, y, beta0 = None, n_folds = 5):
    key = jax.random.PRNGKey(57)
    i_folds = jax.random.randint(key, (y.shape[0],), 0, n_folds + 1)
    def slct(x, mask): return jnp.compress(mask, x, 0)
    total_cost = 0.0
    for j in range(n_folds):
        in_mask, out_mask = (i_folds != j), (i_folds == j)
        j_beta = pg.run(beta0, (jnp.exp(log_l1), jnp.exp(log_l2)), 
                        slct(X1, in_mask), slct(X2, in_mask), slct(y, in_mask))[0]
        j_cost = cost(j_beta, 
                        slct(X1, out_mask), slct(X2, out_mask), slct(y, out_mask))
        total_cost += j_cost
    return total_cost

# Ok, then a test. Note that as X1 is sparse and binary, samples carry relatively
# little information there and the optimization process easily sets them all zero
# unless there is enough data. 

# Generate data. X1 is sparse count-like, X2 embedding-like.
M1, M2, N = 20, 20, 3000
X1 = np.floor(np.exp(np.random.normal(-2, 1, (N, M1))))
X2 = np.random.normal(0, 1, (N, M2))

# True solution.
beta1_true = np.concatenate((np.ones(5), np.zeros(M1-5)))
beta2_true = np.concatenate((np.ones(5), np.zeros(M2-5)))
icept_true = - 0.3


# Simulated target data (0/1) with true model coefs.
y = 0 + (np.random.random(N) < predict((beta1_true, beta2_true, icept_true), X1, X2))

# Common initial value for all iterations.
beta0 = (jnp.zeros(M1), jnp.zeros(M2), jnp.zeros((1,)))   

# test run with artificial regularizers
pg.run(beta0, (.01, .01), X1, X2, y)

# Invoke an optimizer to find hyperparameters efficiently. 
# This will be used with cross validation. 
#
# Bayesian optimization in general is recommendable in smooth low-dimensional
# cases like this, with no gradient information available. 
# There are several packages available, and note that one can tune the process
# more robust or efficient with hyperparameters. 
#
# Note that we treat hyperparameters on the log scale.
l12_opt = bayes_opt.BayesianOptimization(
    f = lambda ll1, ll2: -cv_cost(ll1, ll2, X1, X2, y, beta0=beta0),
    pbounds = {'ll1': (-10, 0), 'll2': (-10, 0)},
    random_state = 1)

# This is not exact science and maxima are typically shallow, so 
# just a few rounds is enough.
l12_opt.maximize(init_points=3, n_iter=10) 

hyperparams = l12_opt.max['params']
l1reg, l2reg = jnp.exp(hyperparams['ll1']), jnp.exp(hyperparams['ll2'])
print("Regularizers:", l1reg, l2reg)

# Final model
res = pg.run(beta0, (l1reg, l2reg), X1, X2, y)
beta, state = res
print("Number of iterations:", state.iter_num)
print(beta[0][:10], beta[1][:10], beta[-1])



# A way to simulate sparse matrices. JIT and tracer values in jax.opt.segment_sum(),
# in argument num_segments, became an issue. The official, currently experimental sparse
# matrix support in JAX may work better. 
if False:
    # THis is not a practical function, in that it creates several arrays 
    # of size X. Looping would be more efficient in terms of space at least.
    # Supposedly your big real data is already sparse, so this is not useful.
    def sparsify(X):
        non_z = (X != 0)
        i_covar = jnp.outer(jnp.ones(X.shape[0], dtype=int), jnp.arange(X.shape[1], dtype=int))
        i_sample = jnp.outer(jnp.arange(X.shape[0], dtype=int), jnp.ones(X.shape[1], dtype=int))
        return (X[non_z], i_covar[non_z], i_sample[non_z], X.shape[0]) # value, column, row, n_rows

    # Matrix multiplication for a sparse representation of matrices.
    def mult_sparse(X_sparse, beta):
        vals, i_covar, i_sample, n_rows = X_sparse
        beta_long = beta[i_covar]
        return jax.ops.segment_sum(beta_long * vals, i_sample, 
                                num_segments=n_rows, 
                                indices_are_sorted=True)

    def linspace_sparse1(beta, X1, X2):
        return mult_sparse(X1, beta[0]) + X2 @ beta[1] + beta[2]

    def predict_sparse1(beta, X1, X2):
        return jax.nn.sigmoid(linspace_sparse1(beta, X1, X2))

    def cost_sparse1(beta, X1, X2, y):
        return jnp.sum(jaxopt.loss.binary_logistic_loss(y, linspace_sparse1(beta, X1, X2)))
