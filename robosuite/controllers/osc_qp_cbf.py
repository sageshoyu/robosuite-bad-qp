import osqp
import numpy as np
import scipy
from scipy import sparse

# NOTE: this code does not work due to improper modeling of torques in contact with the robot arm

# To get this code working without contact torques, we need to extend the CBF formalism to accomodate
# systems where the control input is two derivatives away from the state variable. We use
# exponential barrier functions from this CBF review paper: https://arxiv.org/abs/1903.11199

# This code uses OSQP to solve a quadratic program to compute the safe torque and was based off
# of an example script in the docs: https://osqp.org/docs/examples/least-squares.html
# For ease of debugging, I converted all the sparse linear algebra operations into normal explicit
# numpy operations. We are solving a problem of dimensionality-14, which, for these solvers, is pretty
# small, but in case you need a speed up, you can use the original script as a guide of where
# to re-substitute the scipy.linalg.sparse operations.





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
        self.ul = upper_limits # (both of shape (7,)
        self.gamma = gamma  # for proportional class K function (see Ames above)
        self.prob = osqp.OSQP()
        self.init_problem = False

    def compute_safe_osc_output(self, q, qd, qdd, g, u_pi):
        f = np.concatenate([qd, qdd])

        # this is a slight variant, since joint space are 'generalized' coordinates for the system
        #
        g = np.vstack([np.zeros((7, 6)), g])

        # compute CBF h(x) and \partial h / \partial x
        h = -(q - self.ll) * (q - self.ul)
        dh_dx = np.diag(-(2 * q - (self.ul + self.ll)))
        # compute lie derivatives (but since this is R^n, they are directional derivs)
        Lf_h = np.dot(dh_dx, f)  # since the velocity of the system is f (dynamics)
        Lg_h = np.dot(dh_dx, g)  # affine control transformation

        # in addition to these, we need to compute the following terms
        # Lf^2_h (lie derivative of h twice in the direction of f)
        # LgLf (lie derivative of h first in direction of f then in direction of g)
        # Here are the expressions written out in basic calculus:
        
        # Lf^2_h
        # \sum_j f_j (\sum_i \frac{del f_i}{del x_j} x \frac{del h}{\del x_i} + f_i \frac{del h}{x_j x_i}
        # we have {del h}{del x_i} in dh_dx. we need{del f_i}{del x_j} (derivative of ith coord of f with respect to joint variable j}{
        # LgLf_h
        # \sum_j g_j (same summation mess indexed by i for Lf^2_h)
       
    
        
        P = scipy.linalg.block_diag(np.zeros((6, 6)), np.eye(6))
        c = np.zeros(6 + 6)
        A = np.vstack([
            np.hstack([np.eye(6), -np.eye(6)]),
            np.hstack([Lg_h, np.zeros((14, 6))])])
        l = np.hstack([u_pi, -self.gamma * h - Lf_h])
        u = np.hstack([u_pi, np.inf * np.ones(7)])
        # then, modify this constraint here to respect the exponential ZBF:
        # currently, we respect the constraint Lf_h + Lg_h u + \gamma h(x) \geq 0
        # now we need to replace that with Lf^2_h + LfLg_h u + (gamma, gamma) \dot (h(x), Lfh) \geq 0.

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, c, A, l, u)

        # Solve problem
        return prob.solve()
