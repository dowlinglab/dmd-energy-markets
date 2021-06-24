import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from dmdTrading.core import DMD
from cmath import exp

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
