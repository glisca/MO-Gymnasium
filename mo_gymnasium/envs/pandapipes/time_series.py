import numpy as np
import pandas as pd

from pandapipes.timeseries import run_timeseries

class GasNetworkTimeSeries:

    def __init__(self, net, output_writer, log_variables, verbose = True):
        self._net = net
        self.output_writer = output_writer
        self.log_variables = log_variables
        self.verbose = verbose

        # set up output dictionary
        self.output = {f"{table}.{variable}": None for table, variable in self.log_variables}

    def run_timestep(self, time_step):

        run_timeseries(
            self._net, time_steps=range(time_step, time_step+1), verbose=self.verbose)

        # add the recent data frame to the previous ones
        for table, variable in self.log_variables: 
            key = f"{table}.{variable}"
            if self.output[key] is None: # not been set yet
                self.output[key] = self.output_writer.output[key]
            else: 
                self.output[key] = pd.concat( [self.output[key], self.output_writer.output[key]], ignore_index=True)

    def get_multigaussian_flow(self, num_values, modes, max_flow):

        # Generate an array with 10 values between 0 and 10
        x = np.linspace(0, num_values, num_values)

        # Create Gaussian modes around the 3rd and 7th entries
        mode_functions = [np.exp(-(x - mode_center)**2 / (2 * 1**2)) for mode_center in modes]

        # Combine the modes and scale to get the final array with the highest value of 0.02
        combined = np.sum(mode_functions, axis=0)
        final_array = max_flow * (combined) / (np.max(combined))
        return x, final_array
