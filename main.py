import numpy as np
from types import SimpleNamespace

QUIET = 0
MAX_ITER = 10
ABSTOL = 1e-4
RELTOL = 1e-2


def objective(c, x):
    return np.transpose(c) @ x


def linprog(c, A, b, rho, alpha):
    """To solve standard LP problem"""

    # Compute the size of the input matrix
    m, n = A.shape

    # ADMM Solver
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    history = {}
    history = SimpleNamespace(**history)
    history.objval = [0 for _ in range(MAX_ITER)]
    history.r_norm = [0 for _ in range(MAX_ITER)]
    history.s_norm = [0 for _ in range(MAX_ITER)]
    history.eps_pri = [0 for _ in range(MAX_ITER)]
    history.eps_dual = [0 for _ in range(MAX_ITER)]

    if not QUIET:
        print(f'iter\tnorm\teps pri\ts norm\teps dual\tobjective')

    for k in range(MAX_ITER):
        # x-update
        tmp1 = np.hstack((rho * np.identity(n), np.transpose(A)))
        tmp2 = np.block([A, np.zeros((m, m))])
        temp_neum = np.block([[tmp1], [tmp2]])
        tmp3 = np.block([[rho*(z-u)-c], [b]])
        tmp = np.linalg.solve(temp_neum, tmp3)
        x = tmp[0:n]

        # z-update with relaxation
        z_old = z
        x_hat = alpha * x + (1 - alpha)*z_old
        z = x_hat + u
        u = u + (x_hat - z)

        # Diagnostics, reporting, termination
        history.objval[k] = objective(c, x)
        history.r_norm[k] = np.linalg.norm(x - z)
        history.s_norm[k] = np.linalg.norm(-rho * (z - z_old))

        history.eps_pri[k] = np.sqrt(n) * ABSTOL + RELTOL * max(np.linalg.norm(x), np.linalg.norm(-z))
        history.eps_dual[k] = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u)

        if not QUIET:
            print(f'{k}\t\t{history.r_norm[k]:.3f}\t{history.eps_pri[k]:.3f}\t'
                  f'{history.s_norm[k]:.3f}\t{history.eps_dual[k]:.3f}\t'
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
    c = rng.normal(loc=1, size=(n, 1)) + 0.5
    x0 = abs(rng.random(size=(n, 1)))  # create random solution vector

    A = abs(rng.random((m, n)))  # create random, non-negative matrix A
    b = np.matmul(A, x0)

    x, history = linprog(c, A, b, 1.0, 1.0)