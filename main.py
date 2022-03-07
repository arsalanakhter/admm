import numpy as np


def linprog(c, A, b, rho, alpha):
    'To solve standard LP problem'
    QUIET = 0
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    # Compute the size of the input matrix
    m, n = A.size

    # ADMM Solver
    x = np.zeros(n, 1)
    z = np.zeros(n, 1)
    u = np.zeros(n, 1)

    # if not QUIET:
    #     print(f'{iter}    norm', 'eps pri', 's norm', 'eps dual', 'objective')

    for k in range(MAX_ITER):
        # x-update
        tmp1 = np.concatenate((rho * np.identity(n), np.transpose(A)), axis=0)
        tmp2 = np.concatenate((A, np.zeros(m)), axis=0)
        temp_neum = np.concatenate((tmp1, tmp2), axis=1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.random('state', 0);
    np.random('state', 0);

    n = 500 # dimension of x
    m = 400 # number of equality constraints

    c = np.random(n, 1) + 0.5  # create nonnegative price vector with mean 1
    x0 = abs(np.random(n, 1)) # create random solution vector

    A = abs(np.random(m, n)) # create random, nonnegative matrix A
    b = A * x0

    x, history = linprog(c, A, b, 1.0, 1.0)