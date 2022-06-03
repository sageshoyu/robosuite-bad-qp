import osqp
import numpy as np
import scipy as sp
from scipy import sparse


# Generate problem data
# sp.random.seed(1)
# m = 30
# n = 20
# Ad = sparse.random(m, n, density=0.7, format='csc')
# b = np.random.randn(m)

# problem setup
# OSQP data

class JointLimitCBFSolver():

    def __init__(self, lower_limits, upper_limits, gamma=1.0):
        self.ll = lower_limits
        self.ul = upper_limits
        self.gamma = gamma
        self.prob = osqp.OSQP()
        self.init_problem = False

    def compute_safe_osc_output(self, q, qd, qdd, g, u_pi):
        f = np.concatenate([qd, qdd])
        g = sparse.vstack([sparse.csc_matrix((7, 7)), g])

        # compute CBF h(x) and \partial h / \partial x
        h = -(q - self.ll) * (q - self.ul)
        dh_dx = np.concatenate([-(2 * q - (self.ul + self.ll)), -2 * np.ones(7)])

        # compute lie derivatives (but since this is R^n, they are directional derivs)
        Lf_h = np.dot(dh_dx, f)  # since the velocity of the system is f (dynamics)
        Lg_h = np.dot(dh_dx, g)  # affine control transformation

        P = sparse.block_diag([sparse.csc_matrix((6, 6)), sparse.eye(6)], format='csc')
        c = np.zeros(6 + 6)
        A = sparse.vstack([
            sparse.hstack([sparse.eye(6), -sparse.eye(6)]),
            sparse.hstack([
                sparse.vstack([Lg_h, sparse.csc_matrix((5, 6))]),
                sparse.csc_matrix((6, 6))])], format='csc')
        l = np.hstack([u_pi, -self.gamma * h - Lf_h])
        u = np.hstack([u_pi, np.inf])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, c, A, l, u)

        # Solve problem
        return prob.solve()
