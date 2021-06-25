from pyomo.environ import *
import numpy as np
from matplotlib import pyplot as plt


def schedule_battery(forecast, real_price, plot_ex=False, e0=0):
    '''
    This function will implement a simple battery example based on a forecast
    that is given by the user.
    '''

    # start looping through each location making and solving a model
    models = []
    for ind, single_pred in enumerate(forecast):
        models.append(build_model(single_pred, e0=e0))
        models[ind] = solve(models[ind])

    # now extract the results and store profit
    all_profits = []
    for ind, model in enumerate(models):

        # extract the charge, discharge, and energy
        c_control, d_control, E_control = extract_results(model)

        # now calculate profit
        profit = calc_profit(c_control, d_control, real_price[ind], N=24)
        all_profits.append(profit)

    if plot_ex:
        t = np.arange(forecast[0].size)
        fig, ax = plt.subplots(3, 1, figsize=(7, 7))

        ax[0].plot(t, E_control, 'b.-')
        ax[0].set_xlabel('Time (hr)')
        ax[0].set_ylabel('SOC (MWh)')
        ax[0].set_xticks(range(0, t.size, 3))
        ax[0].set_title('Energy in Battery')

        ax[1].plot(t, c_control, 'r.-')
        ax[1].plot(t, d_control, 'g.-')
        ax[1].set_xlabel('Time (hr)')
        ax[1].set_ylabel('Power from Grid (MW)')
        ax[1].set_xticks(range(0, t.size, 3))
        ax[1].set_title('Charging and Disharging')

        ax[2].plot(t, forecast[-1], 'k', label='Forecast')
        ax[2].plot(t, real_price[-1], 'r.-', label='Market')
        ax[2].set_xlabel('Time (hr)')
        ax[2].set_ylabel('Price ($)')
        ax[2].set_xticks(range(0, t.size, 3))
        ax[2].set_title('Market Prices')
        ax[2].legend()

    # return the profit vector
    return all_profits

def schedule_battery_single(forecast, price, plot_ex=False, e0=0):

    # build model
    model = build_model(forecast, e0=e0)

    # solve model
    model = solve(model)

    # extract the charge, discharge, and energy
    c_control, d_control, E_control = extract_results(model)
    new_e0 = E_control[24]

    # now calculate profit
    profit = calc_profit(c_control, d_control, price, N=24)

    if plot_ex:
        t = np.arange(forecast.shape[0])
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))

        ax[2].plot(t, E_control, 'b.-', linewidth=3, markersize=12)
        ax[2].set_xlabel('Time [hr]')
        ax[2].set_ylabel('Stored Energy\n[MWh]')
        ax[2].set_xticks(range(0, t.size, 3))

        ax[1].step(t, c_control, 'ro-', linewidth=3)
        ax[1].step(t, d_control, 'gs-', linewidth=3)
        ax[1].legend(['Charging', 'Disharging'], ncol=2)
        ax[1].set_ylim(-1.05, 1.48)
        ax[1].set_xlabel('Time [hr]')
        ax[1].set_ylabel('Power from Grid [MW]')
        ax[1].set_xticks(range(0, t.size, 3))

        ax[0].plot(t, forecast, 'k', label='Forecast', linewidth=3, markersize=12)
        ax[0].plot(t, price, 'r.-', label='Actual', linewidth=3, markersize=12)
        ax[0].set_xlabel('Time [hr]')
        ax[0].set_ylabel('Price [$ / MWh]')
        ax[0].set_xticks(range(0, t.size, 3))
        ax[0].legend()

        # return the information and figure and axes
        return profit, new_e0, fig, ax

    else:
        # return only the information
        return profit, new_e0


def build_model(price, e0=0):
    '''
    Create optimization model for MPC

    Inputs:
        price: NumPy array with energy price timeseries
        e0: initial value for energy storage level

    Output:
        model: Pyomo optimization model
    '''

    # Create a concrete Pyomo model. We'll learn more about this in a few weeks
    model = ConcreteModel()

    ## Define Sets

    # Number of timesteps in planning horizon
    # print(price)
    model.HORIZON = Set(initialize=np.arange(price.shape[0]))

    ## Define Parameters

    # Square root of round trip efficiency
    model.sqrteta = Param(initialize=sqrt(0.88))

    # Energy in battery at t=0
    model.E0 = Param(initialize=e0, mutable=True)

    # Charging rate [MW]
    model.c = Var(model.HORIZON, initialize=0.0, bounds=(0, 1))

    # Discharging rate [MW]
    model.d = Var(model.HORIZON, initialize=0.0, bounds=(0, 1))

    # Energy (state-of-charge) [MWh]
    model.E = Var(model.HORIZON, initialize=0.0, bounds=(0, 4))

    ## Define constraints

    # Define Energy Balance constraints. [MWh] = [MW]*[1 hr]
    # Note: this model assumes 1-hour timestep in price data and control actions.
    def EnergyBalance(model, t):
        # First timestep
        if t == 0:
            return model.E[t] == model.E0 + model.c[t]*model.sqrteta-model.d[t]/model.sqrteta

        # Subsequent timesteps
        else:
            return model.E[t] == model.E[t-1]+model.c[t]*model.sqrteta-model.d[t]/model.sqrteta

    model.EnergyBalance_Con = Constraint(model.HORIZON, rule=EnergyBalance)

    ## Define the objective function (profit)
    # Receding horizon
    def objfun(model):
        return sum((-model.c[t] + model.d[t]) * price[t] for t in model.HORIZON)
    model.OBJ = Objective(rule=objfun, sense=maximize)

    return model


def solve(model, solver_name = 'ipopt'):
    '''
    This will create an instance of the model then return the pyomo results class.
    '''

    # Build the model
    # instance = self.build_model(self,price_data,0.0)

    # Specify the solver
    solver = SolverFactory(solver_name)

    # Solve the model
    results = solver.solve(model) # noqa

    return model


def extract_results(instance):
    '''
    param: instance: pyomo object that has been solved
    type: pyomo object
    '''

    # Declare empty lists
    c_control = []
    d_control = []
    E_control = []

    # Loop over elements of HORIZON set.
    for i in instance.HORIZON:
        # Use value( ) function to extract the solution for each varliable and append
        c_control.append(value(instance.c[i]))

        # Adding negative sign to discharge for plotting
        d_control.append(-value(instance.d[i]))
        E_control.append(value(instance.E[i]))

    return c_control, d_control, E_control


def calc_profit(c_control, d_control, price, N=24):
    '''
    From a given list of control actions on the battery, this function
    will calculate profit made from a give price forecast.

    param: c_control: charging battery list
    param: d_control: discharging battery list
    param: real_price: real price from the market
    '''

    c_control = np.array(c_control)[0:N]
    d_control = np.array(d_control)[0:N]
    new_price = np.array(price)[0:N]

    profit = - new_price * d_control - new_price * c_control

    return np.sum(profit)
