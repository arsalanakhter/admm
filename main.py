import numpy as np

QUIET = 0
MAX_ITER = 1000
ABSTOL = 1e-4
RELTOL = 1e-2


def objective(c, x):
    return np.transpose(c) * x


def linprog(c, A, b, rho, alpha):
    """To solve standard LP problem"""

    # Compute the size of the input matrix
    m, n = A.shape

    # ADMM Solver
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    history = {}
    history['objval'] = []
    history['r_norm'] = []
    history['s_norm'] = []
    history['eps_pri'] = []
    history['eps_dual'] = []

    if not QUIET:
        print(f'iter\tnorm\teps pri\ts norm\teps dual\tobjective')

    for k in range(MAX_ITER):
        # x-update
        tmp1 = np.concatenate((rho * np.identity(n), np.transpose(A)), axis=0)
        tmp2 = np.concatenate((A, np.zeros(m)), axis=0)
        temp_neum = np.concatenate((tmp1, tmp2), axis=1)
        tmp3 = np.block([rho*(z-u)-c], [b])
        tmp = temp_neum / tmp3
        x = tmp[1:n]

        # z-update with relaxation
        z_old = z
        x_hat = alpha * x + (1 - alpha)*z_old
        z = x_hat + u
        u = u + (x_hat - z)

        # Diagnostics, reporting, termination
        history.objval[k] = objective(c, x)
        history.r_norm[k] = np.norm(x - z)
        history.s_norm[k] = np.norm(-rho * (z - z_old))

        history.eps_pri[k] = np.sqrt(n) * ABSTOL + RELTOL * max(np.norm(x), np.norm(-z))
        history.eps_dual[k] = np.sqrt(n) * ABSTOL + RELTOL * np.norm(rho * u)

        if ~QUIET:
            print(f'{k}\t{history.r_norm[k]}\t{history.eps_pri[k]}\t'
                  f'{history.s_norm[k]}\t{history.eps_dual[k]}\t'
                  f'{history.objval[k]}')

        if history.r_norm[k] < history.eps_pri[k] and \
                history.s_norm[k] < history.eps_dual[k]:
            break
    return x, history


if __name__ == '__main__':
    seed = 12345
    rng = np.random.default_rng(seed)

    n = 500  # dimension of x
    m = 400  # number of equality constraints

    # create non-negative price vector with mean 1
    c = rng.normal(n, 1) + 0.5
    x0 = abs(rng.random(n))  # create random solution vector

    A = abs(rng.random((m, n)))  # create random, non-negative matrix A
    b = A * x0

    x, history = linprog(c, A, b, 1.0, 1.0)