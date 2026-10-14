# Theta and multiple linear regression

For p numeric inputs, the predicted score is:

    y_hat = theta_0 + theta_1 z_1 + ... + theta_p z_p

The engine fills each missing input with the training median, then standardizes:

    z_j = (x_j - mean_j) / scale_j

The scale is training standard deviation (or 1 for a constant feature). The model
saves medians, means and scales so inference uses exactly the same transformation.
For temporal validation, those quantities are fitted only on the earlier training
subset. The production refit computes them on all available historical labels.

Let A be the standardized design matrix with a leading column of ones, y the
vector of actual scores, m the training row count and D = diag(0,1,...,1). Minimize:

    J(theta) = ||A theta - y||^2 / (2m) + lambda * theta^T D theta / 2

Differentiating and setting the gradient to zero:

    gradient J = A^T(A theta - y)/m + lambda D theta
    (A^T A + m lambda D) theta = A^T y

The intercept is not penalized. With lambda=0 this is ordinary least squares.
An explicit inverse is unnecessary and unreliable with collinear inputs. The
default solver uses NumPy least squares on the augmented problem:

    [       A          ] theta ~= [ y ]
    [ sqrt(m*lambda) D ]          [ 0 ]

This solves ridge regression even when the ordinary normal matrix is singular.
The optional gradient descent solver instead repeats:

    theta <- theta - alpha * (A^T(A theta - y)/m + lambda D theta)

The engine checks alpha against the largest Hessian eigenvalue and tracks
convergence. An unconverged candidate is rejected by the MySQL workflow.

Weights in the original units are:

    weight_j = theta_j / scale_j
    intercept = theta_0 - sum(mean_j * weight_j)

They are exported in the model metadata. They describe fitted associations, not
causal effects. Scores are clipped to the configured minimum/maximum for the
displayed prediction, while raw values and clipping warnings are retained.

MAE is mean absolute error in score points. RMSE weights large errors more heavily.
R-squared can be negative; it is undefined when all validation targets are equal,
in which case the engine stores JSON null. Validation also includes a prediction
of the training mean as a simple baseline.
