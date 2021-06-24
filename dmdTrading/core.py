import numpy as np
from scipy import linalg as la
from cmath import exp
import matplotlib.pyplot as plt

from dmdTrading.utils import Manipulate


class DMD:

    '''
    This class contains the functions needed for performing a full DMD
    on any given matrix. Depending on functions being used, different
    outputs can be achieved.
    This class also contains functions useful to the analysis of DMD
    results and intermediates.
    '''

    @staticmethod
    def decomp(Xf, time, verbose=False, rank_cut=True, esp=1e-2, svd_cut=False,
               num_svd=1, do_SVD=True, given_svd=False):
        '''
        This function performs the basic DMD on a given matrix A.
        The general outline of the algorithm is as follows...
        1)  Break up the input X matrix into time series for 1 to n-1 (X)
            and 2 to n (X) where n is the number of time intervals (X_p)
            (columns). This uses the manipulate class's function "split".
        2)  Compute the Singular Value Decomposition of X. X = (U)(S)(Vh)
        3)  Compute the A_t matrix. This is related to the A matrix which
            gives A = X * Xp. However, the At matrix has been projected
            onto the POD modes of X. Therefore At = U'*A*U. (' detonates
            a matrix transpose)
        4)  Compute the eigen decomposition of the At matrix. At*W=W*L
        5)  Compute the DMD modes of A by SVD reconstruction. Finally, the
            DMD modes are given by the columns of Phi.
            Phi = (Xp)(V)(S^-1)(W)
        6)  Compute the discrete and continuous time eigenvalues
            lam (discrete) is the diagonal matrix of eigenvalues of At.
            omg (continuous) = ln(lam)/dt
        7) 	Compute the amplitude of each DMD mode (b). This is a vector
            which applies to this system: Phi(b)=X_1 Where X_1 is the first
            column of the input vector X. This requires a linear equation
            solver via scipy.
        8)  Reconstruct the matrix X from the DMD modes (Xdmd).

        inputs:
            * X - (mxn) Spacial Temporal Matrix
            * time - (nx1) Time vector

        outputs:
            1. Phi - DMD modes
            2. omg - discrete time eigenvalues
            3. lam - continuous time eigenvalues
            4. b - amplitudes of DMD modes
            5. Xdmd - reconstructed X matrix from DMD modes
            6. rank - the rank used in calculations
            ** all contained in a class see ### (10) ### below **

        options:
            * verbose - boolean for more information
            * svd_cut - boolean for truncation of SVD values of X
            * esp - value to truncate singular values lower than
            * rank_cut - truncate the SVD of X to the rank of X
            * num_svd - number of singular values to use
            * do_SVD - tells the program if the svd is provided to it or not
        '''

        if verbose:
            print('Entering Dynamic Mode Decomposition:\n')

        # --- (1) --- #
        # split the Xf matrix
        X, Xp = Manipulate.split(Xf)
        if verbose:
            print('X = \n', X, '\n')
            print('X` = \n', Xp, '\n')

        ### (2) ###  # noqa:
        # perform a singular value decompostion on X
        if do_SVD:
            if verbose:
                'Performing singular value decompostion...\n'
            U, S, Vh = la.svd(X)
        else:
            if verbose:
                'Singular value decompostion provided...\n'
            U, S, Vh = given_svd

        if verbose:
            print('Singular value decomposition:')
            print('U: \n', U)
            print('S: \n', S)
            print('Vh: \n', Vh)
            print('Reconstruction:')
            S_m = np.zeros(np.shape(X))
            for i in range(len(list(S))):
                S_m[i, i] = S[i]
            recon = np.dot(np.dot(U, S_m), Vh)
            print('X =\n', recon)

        # perform desired truncations of X
        if svd_cut:
            rank_cut = False
        if rank_cut:  # this is the default truncation
            rank = 0
            for i in S:
                if i > esp:
                    rank += 1
            if verbose:
                print('Singular Values of X:', '\n', S, '\n')
                print('Reducing Rank of System...\n')
            Ur = U[:, 0:rank]
            Sr = S[0:rank]
            Vhr = Vh[0:rank, :]
            if verbose:
                recon = np.dot(np.dot(Ur, np.diag(Sr)), Vhr)
                print('Rank Reduced reconstruction:\n', 'X =\n', recon)
        elif svd_cut:
            rank = num_svd
            if verbose:
                print('Singular Values of X:', '\n', S, '\n')
                print('Reducing Rank of System to n =', num_svd, '...\n')
            Ur = U[:, 0:rank]
            Sr = S[0:rank]
            Vhr = Vh[0:rank, :]
            if verbose:
                recon = np.dot(np.dot(Ur, np.diag(Sr)), Vhr)
                print('Rank Reduced reconstruction:\n', 'X =\n', recon)

        # return the condition number to view singularity
        condition = max(Sr) / min(Sr)
        smallest_svd = min(Sr)
        svd_used = np.size(Sr)
        if verbose:
            condition = max(Sr) / min(Sr)
            print('Condition of Rank Converted Matrix X:', '\nK =', condition, '\n')

        # make the singular values a matrix and take the inverse
        Sr_inv = np.diag([i ** -1 for i in Sr])
        Sr = np.diag(Sr)

        ### (3) ###  # noqa:
        # now compute the A_t matrix
        Vr = Vhr.conj().T
        At = Ur.conj().T.dot(Xp)
        At = At.dot(Vr)
        At = At.dot(la.inv(Sr))
        if verbose:
            print('A~ = \n', At, '\n')

        ### (4) ###  # noqa:
        # perform the eigen decomposition of At
        L, W = la.eig(At)
        # also determine the number of positive eigenvalues
        pos_eigs = np.count_nonzero((L > 0))

        ### (5) ###  # noqa:
        # compute the DMD modes
        # phi = Xp @ Vhr.conj().T @ Sr_inv @ W
        phi = np.dot(Xp, Vhr.conj().T)
        phi = np.dot(phi, Sr_inv)
        phi = np.dot(phi, W)

        if verbose:
            print('DMD Mode Matrix:', '\nPhi =\n', phi, '\n')

        ### (6) ###   # noqa:
        # compute the continuous and discrete eigenvalues
        dt = time[1] - time[0]
        lam = L
        omg = np.log(lam) / dt
        if verbose:
            print('Discrete time eigenvalues:\n', 'Lambda =', L, '\n')
            print('Continuous time eigenvalues:\n', 'Omega =', np.log(L) / dt, '\n')
            print('Number of positive eigenvalues: ', pos_eigs, '\n')

        ### (7) ###  # noqa:
        # compute the amplitude vector b by solving the linear system described.
        # note that a least squares solver has to be used in order to approximate
        # the solution to the overdefined problem
        x1 = X[:, 0]
        b = la.lstsq(phi, x1)
        b = b[0]
        if verbose:
            print('b =\n', b, '\n')

        ### (8) ###  # noqa:
        # finally reconstruct the data matrix from the DMD modes
        length = np.size(time)  # number of time measurements
        # initialize the time dynamics
        dynamics = np.zeros((rank, length), dtype=np.complex_)
        for t in range(length):
            omg_p = np.array([exp(i * time[t]) for i in omg])
            dynamics[:, t] = b * omg_p

        if verbose:
            print('Time dynamics:\n', dynamics, '\n')

        # reconstruct the data
        Xdmd = np.dot(phi, dynamics)
        if verbose:
            print('Reconstruction:\n', np.real(Xdmd), '\n')
            print('Original:\n', np.real(Xf), '\n')

        ### (9) ###  # noqa:
        # calculate some residual value
        res = np.real(Xf - Xdmd)
        error = la.norm(res) / la.norm(Xf)
        if verbose:
            print('Reconstruction Error:', round(error * 100, 2), '%')

        ### (10) ###  # noqa:
        # returns a class with all of the results
        class results():
            def __init__(self):
                self.phi = phi
                self.omg = omg
                self.lam = lam
                self.b = b
                self.Xdmd = Xdmd
                self.error = error * 100
                self.rank = rank
                self.svd_used = svd_used
                self.condition = condition
                self.smallest_svd = smallest_svd
                self.pos_eigs = pos_eigs
                self.dynamics = dynamics
                self.svd_used = svd_used

        return results()

    @staticmethod
    def predict(dmd, t):
        '''
        This function will take a DMD decomposition output
        result and a desired time incremint prediction and
        produce a prediction of the system at the given time.

        inputs:
            * dmd - class that comes from the function "decomp"
            * t - future time for prediction

        outputs:
            * x - prediction vector (real part only)
        '''

        # finally reconstruct the data matrix from the DMD modes
        dynamics = np.zeros((DMD.rank, 1), dtype=np.complex_)
        omg_p = np.array([exp(i * t) for i in DMD.omg])
        dynamics = DMD.b * omg_p
        x = np.real(np.dot(DMD.phi, dynamics))

        return x

    @staticmethod
    def dmd_specific_svd(Xf):
        '''
        This is a helper function which will split the data and
        perform a singular value decomposition based on whatever the
        input data is and return the outputs for scipy.
        '''

        X, Xp = Manipulate.split(Xf)
        result = la.svd(X)

        return result

    @staticmethod
    def mode_analysis(data, dmd_results, N=np.arange(2), analyze=False, plot=True):
        '''
        This function will take the time dynamics and spacial dynamics and show
        a plot for the number of modes that have been specified.
        inputs:
        results - results class from DMD.decomp
        data - data used in the decomposition
        N - number of modes that you want to plot
        outputs:
        results - class with useful information
        fig - figure of the modes
        '''

        # make the time and space vectors
        time = np.arange(np.shape(data)[1])
        space = np.arange(np.shape(data)[0])

        # check feasibility
        if np.size(N) > dmd_results.svd_used:
            print('Too many singular values requested!')
            print('Reducing analysis to N =', dmd_results.svd_used)
            N = np.arange(dmd_results.svd_used)

        # do an analysis of the modes if ased for (default yes)
        if analyze:
            results = []
            True

        # make a plot if ased for (default no)
        if plot:

            # create the figure and the axes
            fig = plt.figure(figsize=(10, 2.3 * np.size(N)))
            time_axes = [True for i in N]
            space_axes = [True for i in N]

            # through the number of modes that are desired to be analyzed
            for ind, n in enumerate(N):

                # set up each axis
                time_axes[ind] = fig.add_subplot(np.size(N), 2, ind*2 + 1)
                time_axes[ind].set_xlabel('Time')
                time_axes[ind].set_ylabel('Price ($)')
                title = 'Time Mode #'+str(n+1)+' || eig = ' + \
                    str(round(dmd_results.omg[n], 2))
                time_axes[ind].set_title(title)
                time_axes[ind].plot(time, np.real(dmd_results.dynamics[n]))

                space_axes[ind] = fig.add_subplot(np.size(N), 2, ind*2 + 2)
                space_axes[ind].set_xlabel('Location Index')
                space_axes[ind].set_ylabel('Price ($)')
                space_axes[ind].set_title('Spacial Mode #'+str(n+1))
                space_axes[ind].plot(space, np.real(dmd_results.phi.T[n]))
            fig.tight_layout()

        # return the desired stuff
        if plot and analyze:
            return fig and results
        elif plot and not analyze:
            return fig
        elif analyze and not plot:
            return results
