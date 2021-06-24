import numpy as np
from scipy import linalg as la
from cmath import exp
import matplotlib.pyplot as plt
from matplotlib import cm


class Manipulate:

    '''
    This class contains helpful functions for the specific matrix
    manipulation needed for Dynamic Mode Decomposition.
    '''

    @staticmethod
    def split(Xf, verbose=False):
        '''
        This function will perform a crutical manipulation for DMD
        which is the splitting of a spacial-temporal matrix (Xf) into
        two matrices (X and Xp). The X matrix is the time series for
        1 to n-1 and Xp is the time series of 2 to n where n is the
        number of time intervals (columns of the original Xf).
        input:
        Xf - matix of full spacial-temporal data
        output:
        X - matix for times 1 to n-1
        Xp - matix for times 2 to n
        options:
        verbose - boolean for visualization of splitting
        '''

        if verbose:
            print('Entering the matrix splitting function:')

        if verbose:
            print('Xf =\n', Xf, '\n')

        X = Xf[:, :-1]
        Xp = Xf[:, 1:]

        if verbose:
            print('X =\n', X, '\n')
            print('Xp =\n', Xp, '\n')
        return X, Xp


class Examples:

    '''
    This class will hold functions that will give very simple examples of
    how DMD works in this library of functions. There will be theoretical
    examples as well as data driven examples in this class once the class
    is out of development
    '''

    @staticmethod
    def kutz():
        '''
        This is a simple example of how DMD can be used to reconstruct
        a complex example from Kutz's book on DMD.
        '''

        print('To show how DMD can be performed using the class given')
        print('let us take a look at an example from Kutz\'s book on DMD\n')

        print('We will look at a complex, periodic function given below:\n')

        print('f(x,t) = sech(x+3)exp(2.3it) + 2sech(x)tanh(x)exp(2.8it)\n')

        print('Now, the 3D function will be plotted on a surface plot as well as its')
        print('DMD reconstruction based on rank reduction at 1,2, and 3 singular values.\n')

        print('It can be shown that this function only has rank = 2, so notice how the DMD')
        print('reconstruction at rank = 3 is pretty much identical to the rank = 2 surface.\n')

        # testing function from book
        im = 0+1j

        def sech(x):
            return 1/np.cosh(x)

        def f(x, t):
            return sech(x + 3)*exp(2.3*im*t) + 2*sech(x)*np.tanh(x)*exp(im*2.8*t)

        points = 100
        x = np.linspace(-10, 10, points)
        t = np.linspace(0, 4*np.pi, points)

        # test decomposition of the function given above
        F = np.zeros((np.size(x), np.size(t)), dtype=np.complex_)
        for i, x_val in enumerate(x):
            for j, t_val in enumerate(t):
                F[i, j] = f(x_val, t_val)
        results1 = DMD.decomp(F, t, verbose=False, num_svd=1, svd_cut=True)
        results2 = DMD.decomp(F, t, verbose=False, num_svd=2, svd_cut=True)
        results3 = DMD.decomp(F, t, verbose=False, num_svd=3, svd_cut=True)

        # plotting

        # make the figure
        fig = plt.figure(figsize=(10, 7))
        surf_real_ax = fig.add_subplot(2, 2, 1, projection='3d')
        surf1_ax = fig.add_subplot(2, 2, 2, projection='3d')
        surf2_ax = fig.add_subplot(2, 2, 3, projection='3d')
        surf3_ax = fig.add_subplot(2, 2, 4, projection='3d')

        surf_real_ax = Visualize.surface_data(
            np.real(F), t, x, provide_axis=True, axis=surf_real_ax)
        surf1_ax = Visualize.surface_data(
            np.real(results1.Xdmd), t, x, provide_axis=True, axis=surf1_ax)
        surf2_ax = Visualize.surface_data(
            np.real(results2.Xdmd), t, x, provide_axis=True, axis=surf2_ax)
        surf3_ax = Visualize.surface_data(
            np.real(results3.Xdmd), t, x, provide_axis=True, axis=surf3_ax)

        surf_real_ax.set_xlabel('t')
        surf_real_ax.set_ylabel('x')
        surf_real_ax.set_zlabel('f(x,t)')
        surf_real_ax.set_title('Original function')

        surf1_ax.set_xlabel('t')
        surf1_ax.set_ylabel('x')
        surf1_ax.set_zlabel('f(x,t)')
        surf1_ax.set_title('1 Singular Value')

        surf2_ax.set_xlabel('t')
        surf2_ax.set_ylabel('x')
        surf2_ax.set_zlabel('f(x,t)')
        surf2_ax.set_title('2 Singular Values')

        surf3_ax.set_xlabel('t')
        surf3_ax.set_ylabel('x')
        surf3_ax.set_zlabel('f(x,t)')
        surf3_ax.set_title('3 Singular Values')

        # now make a plot for the normal mode analysis

        # make the figure
        fig_2 = plt.figure(figsize=(10, 7))
        time_1_ax = fig_2.add_subplot(3, 2, 1)
        time_2_ax = fig_2.add_subplot(3, 2, 3)
        time_3_ax = fig_2.add_subplot(3, 2, 5)
        space_1_ax = fig_2.add_subplot(3, 2, 2)
        space_2_ax = fig_2.add_subplot(3, 2, 4)
        space_3_ax = fig_2.add_subplot(3, 2, 6)

        space_1_ax.plot(x, np.real(results3.phi.T[0]))
        space_2_ax.plot(x, np.real(results3.phi.T[1]))
        space_3_ax.plot(x, np.real(results3.phi.T[2]))

        time_1_ax.plot(x, np.real(results3.dynamics[0]))
        time_2_ax.plot(x, np.real(results3.dynamics[1]))
        time_3_ax.plot(x, np.real(results3.dynamics[2]))

        time_1_ax.set_xlabel('t')
        time_1_ax.set_ylabel('f')
        time_1_ax.set_title('First Time Mode')

        time_2_ax.set_xlabel('t')
        time_2_ax.set_ylabel('f')
        time_2_ax.set_title('Second Time Mode')

        time_3_ax.set_xlabel('t')
        time_3_ax.set_ylabel('f')
        time_3_ax.set_title('Third Time Mode')

        space_1_ax.set_xlabel('x')
        space_1_ax.set_ylabel('f')
        space_1_ax.set_title('First Spacial Mode')

        space_2_ax.set_xlabel('x')
        space_2_ax.set_ylabel('f')
        space_2_ax.set_title('Second Spacial Mode')

        space_3_ax.set_xlabel('x')
        space_3_ax.set_ylabel('f')
        space_3_ax.set_title('Third Spacial Mode')

        fig_2.tight_layout()

        plt.show()

        return fig, fig_2


class Energy:
    '''
    This class will hold all of the necessary function for manipulating
    the energy price data in this project along with the DMD results
    '''

    def calc_end_error(end_val, error, data, verbose=False):
        '''
        This function calculates the 2-norm error of a matrix of
        residual values based on the last "end_val" times.
        The input is a list of residual matrices like the
        "sv_dependence" function returns.

        inputs:
            * end_val - the last "N" number of time steps to calculate error on
            * error - matrix of error measurements
            * data - original data input into dmd
            * verbose - verbose argument
        output:
            * end_error - % error of the last "end_val" time steps
        '''

        # determine the 2-norm of the data in the last time measurements
        time_len = np.size(data[0])
        data = data.T
        data = data[time_len - end_val:]
        data = data.T
        data_norm = la.norm(data)

        # initalize a list for the error values
        end_error = []

        # loop through to find the error for each sv
        for test_ind, res_matrix in enumerate(error):

            # grab the last end_vals
            res_matrix = res_matrix.T
            res_matrix = res_matrix[time_len - end_val:]
            res_matrix = res_matrix.T

            # calculate the error
            error = la.norm(res_matrix) / data_norm * 100

            # append the error
            end_error.append(error)

            if verbose:
                print('------------------------------------')
                print('Test #'+str(test_ind))
                print()
                print(res_matrix)
                print(la.norm(res_matrix))
                print()
                print(data)
                print(data_norm)
                print('Error:', error)
                print('------------------------------------')
                print()

        return end_error

    def calc_opt_sv_cut(results, data, N=24, verbose=False):
        '''
        From a singular value sensitivity test class, this will calculate
        the optimal rank reduction given a time period to test on.
        inputs:
        results - class returned from sv_sensitivity
        N - time period on which to test
        outputs:
        opt_sv - optimal singular value to cut on
        end_error - array that has the error for each rank reduction
        '''

        # determine the optimal number of singular values
        end_error = Energy.calc_end_error(N, results.res_matrix, data, verbose=False)
        opt_sv = end_error.index(min(end_error)) + 1
        if verbose:
            print('For a time period of', N, 'hours...')
            print('\nOptimal Singular Value Reduction Identified:', opt_sv)
            print('Percentage:', opt_sv/np.size(results.num_svd)*100, '%')

        return opt_sv, end_error


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


class Augment:

    '''
    This class hold the information on how to perform augmented DMD.
    '''

    def augment_matrix(x, n):
        '''
        Function to take a vector x and returns an augmented matrix X which
        has n rows.
        '''

        # length of full time series
        num_elements = x.shape[0]

        # length of each row in the X matrix
        len_row = num_elements - n + 1

        # initalize the matrix
        X = []

        # loop over each row
        for row_num in range(n):

            # grab the smaller vector
            small_vec = x[row_num:row_num + len_row]

            # append the vector
            X.append(small_vec)

        return np.array(X)

    def make_forecast(data, train_start, train_end, num_predict=48, rank=8, verbose=False,
                      give_recon=False):
        '''
        This function will make a 48 hour forecast using augmented DMD.
        '''

        # get the time measurements desired
        data = data.T[train_start:train_end].T
        if verbose:
            print('start:', train_start)
            print('end:', train_end)

        # loop through each row of data and make a forecast
        forecast = []
        error_vec = []
        for x in data:

            # determine how many rows and the length of each row
            num_rows = int((train_end - train_start) / 2)
            if verbose:
                print('Rows:', num_rows)

            # make the augmented matrix
            X = Augment.augment_matrix(x, num_rows)
            # DMD
            time = np.arange(X[0].shape[0])
            if verbose:
                print('augmented shape:', X.shape)
            dmd_results = DMD.decomp(X, time, svd_cut=True, num_svd=rank)

            # predict the future measurements
            x_dmd_future = []
            time_future = np.arange(num_predict) + x.shape[0]
            for t in time_future:
                x_dmd_future.append(DMD.predict(dmd_results, t)[0])
            x_dmd_future = np.array(x_dmd_future)
            forecast.append(x_dmd_future)
            error_vec.append(dmd_results.error)

        # return the forecast
        if give_recon:
            return np.array(forecast), error_vec
        else:
            return np.array(forecast)

class Visualize:

    '''
    This class holds all of the functions needed for visualizing of DMD
    results and the input data into DMD.
    '''

    def surface_data(F, x, t, bounds_on=False, provide_axis=False, axis=False,
                     bounds=[[0, 1], [0, 1], [0, 1]]):
        '''
        This function will create a surface plot of given a set of data
        for f(x),x,t. f(x) must be given in matrix format with evenly
        spaced x and t corresponding the A matrix.
        inputs:
        f - spacial-temporal data
        x - spacial vector
        t - time vector
        outputs:
        surf - object of the 3D plot
        options:
        bounds_on - boolean to indicate bounds wanted
        bounds - Optional array that contains the bounds desired to put on
                 the axes. Sample input: [[0,1],[0,1],[0,1]] for f(x),x,t.
        '''

        # first make a meshgrid with the t and x vector.
        # we first define the x values as the rows and t as the columns
        # in order to be consistent with general DMD structure.
        X, T = np.meshgrid(x, t)

        # Create 3D figure if you are not providing the axis
        if provide_axis:
            ax = axis
        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        # Plot f(x)
        ax.plot_surface(X, T, F, linewidth=0, cmap=cm.coolwarm, antialiased=True)

        # give the two options for what to provide
        if provide_axis:
            return ax
        else:
            return ax, fig
