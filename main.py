import numpy as np


def linprog(c, A, b, rho, alpha):
    'To solve standard LP problem'
    QUIET = 0
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    # Compute the size of the input matrix
    m, n = A.size


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
