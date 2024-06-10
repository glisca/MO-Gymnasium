import numpy as np 
import os
import pandapipes as pp
import pandapower.timeseries as ts
import pandapower.control as control
import pandas as pd

from pandapower.control.basic_controller import Controller
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter

from envs.pandapipes.time_series import GasNetworkTimeSeries

class GasNetworkController:

    def __init__(self, gas_network, verbose=False):
        """ sets up everything we need to run pandapipes time series simulations
        """

        self._gas_network = gas_network

        self._verbose = verbose

        # Time series of the source, mass_storage and sink.
        path_profile_source = os.path.join(
            os.path.dirname(__file__), 'assets', 'source_profiles.csv')
        print(f'Loading source flow profile from {path_profile_source}')
        self._source_csv = pd.read_csv(path_profile_source, index_col=0)

        # self._mass_storage_pd = pd.DataFrame([0.] * 25, columns=['mdot_storage'])

        path_profile_sink = os.path.join(
            os.path.dirname(__file__), 'assets','sink_profiles.csv')
        print(f'Loading source flow profile from {path_profile_sink}')
        self._sink_csv = pd.read_csv(path_profile_sink, index_col=0)

        self._controller_source = control.ConstControl(
            self._gas_network, element='source', variable='mdot_kg_per_s',
            element_index=self._gas_network.source.index.values,
            data_source=DFData(self._source_csv),
            profile_name=self._gas_network.source.index.values.astype(str))

        self._controller_storage = StorageController(
            net=self._gas_network, sid=0,
            # data_source=DFData(self._mass_storage_pd), mdot_profile='mdot_storage')
            data_source=None, mdot_profile='mdot_storage')

        self._controller_sink = control.ConstControl(
            self._gas_network, element='sink', variable='mdot_kg_per_s',
            data_source=DFData(self._sink_csv),
            element_index=self._gas_network.sink.index.values,
            profile_name=self._gas_network.sink.index.values.astype(str))

        self._log_variables = [
            ('source', 'mdot_kg_per_s'),
            ('res_source', 'mdot_kg_per_s'),

            ('mass_storage', 'mdot_kg_per_s'),
            ('mass_storage', 'm_stored_kg'),
            ('mass_storage', 'filling_level_percent'),
            ('res_mass_storage', 'mdot_kg_per_s'),

            ('sink', 'mdot_kg_per_s'),
            ('res_sink', 'mdot_kg_per_s'),

            # ('ext_grid', 'mdot_kg_per_s'),
            ('res_ext_grid', 'mdot_kg_per_s'),

            ('res_junction', 'p_bar'),
            ('res_pipe', 'v_mean_m_per_s'),
            ('res_pipe', 'reynolds'),
            ('res_pipe', 'lambda')]

        self._output_writer = OutputWriter(
            self._gas_network, log_variables=self._log_variables)

    @property
    def gas_network(self):
        """
        """
        return self._gas_network

class StorageController(Controller):
    """
        Example class of a Storage-Controller. Models an abstract mass storage.
    """
    def __init__(self, net, sid, data_source=None, mdot_profile=None, in_service=True,
                 recycle=False, order=0, level=0, duration_timestep_h=1, **kwargs):
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                    initial_pipeflow = True, **kwargs)

        # read storage attributes from net
        self.sid = sid  # index of the controlled storage

        self.gas_network = net

        # profile attributes
        self.data_source     = data_source
        self.mdot_profile    = mdot_profile
        self.duration_ts_sec = duration_timestep_h * 3600

        self._initialize_controller_state()

    def _initialize_controller_state(self):

        self.junction      = self.gas_network.mass_storage.at[self.sid, "junction"]
        self.mdot_kg_per_s = self.gas_network.mass_storage.at[self.sid, "mdot_kg_per_s"]
        self.name          = self.gas_network.mass_storage.at[self.sid, "name"]
        self.storage_type  = self.gas_network.mass_storage.at[self.sid, "type"]
        self.in_service    = self.gas_network.mass_storage.at[self.sid, "in_service"]
        self.scaling       = self.gas_network.mass_storage.at[self.sid, "scaling"]

        # specific attributes
        self.max_m_kg    = self.gas_network.mass_storage.at[self.sid, "max_m_stored_kg"]
        self.min_m_kg    = self.gas_network.mass_storage.at[self.sid, "min_m_stored_kg"]
        self.m_stored_kg = self.gas_network.mass_storage.at[self.sid, "init_m_stored_kg"]

        self.last_time_step  = 0
        self.applied = False

    def reset(self):

        self._initialize_controller_state()

    # In a time-series simulation the mass storage should read new flow values from a profile and keep track
    # of its amount of stored mass as depicted below.
    def time_step(self, net, time):

        # print(f'\u001b[31m{self.last_time_step}\u001b[0m')

        # keep track of the stored mass (the duration of one time step is given as input to the controller)
        if self.last_time_step is not None:
            # The amount of mass that flowed into or out of the storage in the last timestep is added
            # requested change of mass:
            self.delta_m_kg_req = (self.mdot_kg_per_s * (time - self.last_time_step) * self.duration_ts_sec)

            # limit by available mass and free capacity in the storage:
            if self.delta_m_kg_req > 0:  # "charging"
                self.delta_m_kg_real = min(self.delta_m_kg_req, self.max_m_kg - self.m_stored_kg)
            else:  # "discharging", delta < 0
                self.delta_m_kg_real = max(self.delta_m_kg_req, self.min_m_kg - self.m_stored_kg)

            self.m_stored_kg += self.delta_m_kg_real

            # if time - self.last_time_step == 0:
            #     self.mdot_kg_per_s = 0
            # else:
            #     self.mdot_kg_per_s = self.delta_m_kg_real / ((time - self.last_time_step) * self.duration_ts_sec)

        self.last_time_step = time

        # # read new values from a profile
        # if self.data_source:
        #     if self.mdot_profile is not None:
        #         self.mdot_kg_per_s = self.data_source.get_time_step_value(
        #             time_step=time, profile_name=self.mdot_profile)
        #         self.m_stored_kg *= self.scaling * self.in_service
        # else:
        #     self.mdot_kg_per_s =  -0.05 + np.random.random() * 0.1

        self.applied = False  # reset applied variable

    # Some convenience methods to calculate indicators for the state of charge:
    def get_stored_mass(self):
        # return the absolute stored mass
        return self.m_stored_kg

    def get_free_stored_mass(self):
        # return the stored mass excl. minimum filling level
        return self.m_stored_kg - self.min_m_kg

    def get_filling_level_percent(self):
        # return the ratio of absolute stored mass and total maximum storable mass in Percent
        return 100 * self.get_stored_mass() / self.max_m_kg

    def get_free_filling_level_percent(self):
        # return the ratio of available stored mass (i.e. excl. min_m_stored_kg) and difference between max and min in Percent
        return 100 * self.get_free_stored_mass() / (self.max_m_kg - self.min_m_kg)

    # Define which values in the net shall be updated
    def write_to_net(self, net):
        # write mdot_kg_per_s, m_stored_kg to the table in the net
        net.mass_storage.at[self.sid, "mdot_kg_per_s"]         = self.mdot_kg_per_s
        net.mass_storage.at[self.sid, "m_stored_kg"]           = self.m_stored_kg
        net.mass_storage.at[self.sid, "filling_level_percent"] = \
            self.get_free_filling_level_percent()
        # Note: a pipeflow will automatically be conducted in the run_timeseries / run_control procedure.
        # This will then update the result table (net.res_mass_storage).
        # If something was written to net.res_mass_storage in this method here, the pipeflow would overwrite it.

    # In case the controller is not yet converged (i.e. in the first iteration,
    # maybe also more iterations for more complex controllers), the control step is executed.
    # In the example it simply adopts a new value according to the previously calculated target
    # and writes back to the net.
    def control_step(self, net):
        # Call write_to_net and set the applied variable True
        self.write_to_net(net)
        self.applied = True

    # convergence check
    def is_converged(self, net):
        # check if controller already was applied
        return self.applied
