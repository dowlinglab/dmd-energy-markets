import numpy as np
from scipy import linalg as la
from dmdTrading.core import DMD


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

        # initialize a list for the error values
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