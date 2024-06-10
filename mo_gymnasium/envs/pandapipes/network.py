import pandapipes as pp

from pandapipes.pandapipes_net import add_default_components
from pandapipes.pandapipes_net import get_basic_net_entries
from pandapipes.pandapipes_net import pandapipesNet
from pandapipes.properties.fluids import Fluid
from pandapipes.std_types.std_types import add_basic_std_types

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

class GasNetwork(pandapipesNet):

    '''
    Note: Mirrors the implementation of the pandapipes.create_empty_network() function!
    '''

    def __init__(self, name='', fluid='hydrogen', add_stdtypes=True):
        super().__init__(get_basic_net_entries())

        add_default_components(self, True)

        self['name'] = name

        if add_stdtypes:
            add_basic_std_types(self)

        if fluid is not None:
            if isinstance(fluid, Fluid):
                self["fluid"] = fluid
            elif isinstance(fluid, str):
                pp.create_fluid_from_lib(self, fluid)
            else:
                logger.warning("The fluid %s cannot be added to the net. Only fluids of type Fluid or "
                            "strings can be used." % fluid)

        pn_bar = 30
        norm_temp = 293.15 # in K, would be 20° C

        # create the coordinates of 4 locations
        geodata = [(0,0), (1,-1), (2,0), (1, 1)]

        # create junctions
        j = dict()
        for i in range(0, 4):
            j[i] = pp.create_junction(
                self, pn_bar=pn_bar, tfluid_k=norm_temp, name=f"Junction {i}", geodata=geodata[i])

        # create junction elements
        pp.create_ext_grid(
            self, junction=j[0], name="ExternalGrid",
            p_bar=pn_bar, t_k=norm_temp)

        # attach components to junctions
        pp.create_source(
            self, junction=j[1], name="Source",
            mdot_kg_per_s=0.2)

        pp.create_mass_storage(
            self, junction=j[3], name = "Storage", type="Classical mass storage",
            mdot_kg_per_s=0.00, init_m_stored_kg=2, min_m_stored_kg=0, max_m_stored_kg=1000)

        pp.create_sink(
            self, junction=j[2], name="Sink",
            mdot_kg_per_s=0.01)

        # connect the junctions with pipes
        pp.create_pipe_from_parameters(
            self, from_junction=j[0], to_junction=j[1],
            length_km = 10, diameter_m=0.4, name="Pipe 0", geodata=[geodata[0], geodata[1]])

        pp.create_pipe_from_parameters(
            self, from_junction=j[1], to_junction=j[2],
            length_km = 10, diameter_m=0.4, name="Pipe 1", geodata=[geodata[1], geodata[2]])

        pp.create_pipe_from_parameters(
            self, from_junction=j[2], to_junction=j[3], 
            length_km = 20, diameter_m=0.4, name="Pipe 2", geodata=[geodata[2], geodata[3]])
