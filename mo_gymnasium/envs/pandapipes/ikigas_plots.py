# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from pandapower.plotting import draw_collections

from pandapipes.component_models.circulation_pump_mass_component import CirculationPumpMass
from pandapipes.component_models.circulation_pump_pressure_component import CirculationPumpPressure
from pandapipes.component_models.pump_component import Pump
from pandapipes.plotting.collections import create_junction_collection, create_pipe_collection, \
    create_valve_collection, create_source_collection, create_pressure_control_collection, \
    create_heat_exchanger_collection, create_sink_collection, create_pump_collection, \
    create_compressor_collection, create_flow_control_collection
from pandapipes.plotting.generic_geodata import create_generic_coordinates
from pandapipes.plotting.plotting_toolbox import get_collection_sizes
import numpy as np
from matplotlib.patches import RegularPolygon, Rectangle, Circle, PathPatch, FancyArrow
from matplotlib.path import Path
from pandapower.plotting.plotting_toolbox import get_color_list, _rotate_dim2, get_angle_list, \
    get_list

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def storage_patches(node_coords, size, angles, **kwargs):
    """
    Creation function of patches for storages.

    :param node_coords: coordinates of the nodes that the sources belong to.
    :type node_coords: iterable
    :param size: size of the patch
    :type size: float
    :param angles: angles by which to rotate the patches (in radians)
    :type angles: iterable(float), float
    :param kwargs: additional keyword arguments (might contain parameter "offset")
    :type kwargs:
    :return: Return values are: \
        - lines (list) - list of coordinates for lines leading to source patches and diagonals\
        - polys (list of RegularPolygon) - list containing the load patches\
        - keywords (set) - set of keywords removed from kwargs
    """
    offset = kwargs.get("offset", size * 2)
    all_angles = get_angle_list(angles, len(node_coords))
    facecolor = kwargs.get("patch_faceolor", "w")
    edgecolor = kwargs.get("patch_edgecolor", "k")
    filling_percentages =  kwargs.get("filling_percentages", [0.5] * len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    edgecolors = get_color_list(edgecolor, len(node_coords))
    polys, lines = list(), list()
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, offset]), all_angles[i])
        #p_edge_left = p2 + _rotate_dim2(np.array([size, size]), all_angles[i])
        #p_edge_right = p2 + _rotate_dim2(np.array([- size, size]), all_angles[i])
        p_ll = p2 + _rotate_dim2(np.array([-size, 0]), all_angles[i])
        polys.append(Rectangle(p_ll, 2 * size, 2 * size, angle=(- all_angles[i] / np.pi * 180),
                               fc=facecolors[i], ec=edgecolors[i]))
        polys.append(Rectangle(p_ll, 2 * size, filling_percentages[i] * 2*size, angle=(- all_angles[i] / np.pi * 180),
                               fc=edgecolors[i], ec=edgecolors[i]))
        x, y = p_ll
        x += size
        y += 2*size + 0.1

        plt.text(x, y, f"Fill percentage: {np.round(filling_percentages[i], 2)}%", fontsize=15, horizontalalignment='center')
        lines.append((node_geo, p2))
        #lines.append((p2, p_edge_left))
        #lines.append((p2, p_edge_right))

    return lines, polys, {"offset", "filling_percentages"}

from pandapipes.plotting.collections import _create_node_element_collection
def create_mass_storage_collection(net, mass_storages=None, size=1., infofunc=None, picker=False,
                             orientation=(np.pi*7/6), cmap=None, norm=None, z=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes sources.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param sources: The sources for which the collections are created. If None, all sources
                    connected to junctions that have junction_geodata entries are considered.
    :type sources: list, default None
    :param size: Patch size
    :type size: float, default 1.
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: Picker argument passed to the patch collection
    :type picker: bool, default False
    :param orientation: Orientation of source collection. pi is directed downwards, increasing\
        values lead to clockwise direction changes.
    :type orientation: float, default np.pi*(7/6)
    :param cmap: colormap for the source colors
    :type cmap: matplotlib norm object, default None
    :param norm: matplotlib norm object to normalize the values of z
    :type norm: matplotlib norm object, default None
    :param z: Array of source result magnitudes for colormap. Used in case of given cmap. If None,\
        net.res_source.mdot_kg_per_s is used.
    :type z: array, default None
    :param kwargs: Keyword arguments are passed to the patch function
    :return: source_pc - patch collection, source_lc - line collection
    """
    mass_storages = net.mass_storage.index
    if len(mass_storages) == 0:
        return None
    infos = [infofunc(i) for i in range(len(mass_storages))] if infofunc is not None else []
    filling_percentages = net.mass_storage.loc[mass_storages, "filling_level_percent"].values / 100
    node_coords = net.junction_geodata.loc[net.mass_storage.loc[mass_storages, "junction"].values,
                                           ["x", "y"]].values
    
    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    if cmap is not None:
        if z is None:
            z = net.res_source.mdot_kg_per_s
        colors = [cmap(norm(z.at[idx])) for idx in mass_storages]
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)
    
    source_pc, source_lc = _create_node_element_collection(
        node_coords, storage_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, patch_edgecolor=patch_edgecolor, line_color=line_color,
        linewidths=linewidths, repeat_infos=(1, 3), filling_percentages=filling_percentages, **kwargs)
    return source_pc, source_lc

def simple_plot(net, respect_valves=False, respect_in_service=True, pipe_width=2.0,
                junction_size=1.0, ext_grid_size=1.0, plot_sinks=False, plot_sources=False,
                sink_size=1.0, source_size=1.0, valve_size=1.0, pump_size=1.0,
                heat_exchanger_size=1.0, pressure_control_size=1.0, compressor_size=1.0, flow_control_size=1.0,
                scale_size=True, junction_color="r", pipe_color='silver', ext_grid_color='orange',
                valve_color='silver', pump_color='silver', heat_exchanger_color='silver',
                pressure_control_color='silver', compressor_color='silver', flow_control_color='silver',
                library="igraph", show_plot=True, ax=None, **kwargs):
    """
    Plots a pandapipes network as simple as possible. If no geodata is available, artificial
    geodata is generated. For advanced plotting see
    the `tutorial <https://github.com/e2nIEE/pandapipes/blob/master/tutorials/simple_plot.ipynb>`_.

    :param net: The pandapipes format network.
    :type net: pandapipesNet
    :param respect_valves: Respect valves if artificial geodata is created. \
            Note: This Flag is ignored if plot_line_switches is True
    :type respect_valves: bool default False
    :param respect_in_service: Respect only components which are in service.
    :type respect_in_service: bool default True
    :param pipe_width: Width of pipes
    :type pipe_width: float, default 5.0
    :param junction_size: Relative size of junctions to plot. The value junction_size is multiplied\
            with mean_distance_between_buses, which equals the distance between the max geoocord\
            and the min divided by 200
    :type junction_size: float, default 1.0
    :param ext_grid_size: Relative size of ext_grids to plot. See bus sizes for details. Note: \
            ext_grids are plottet as rectangles
    :type ext_grid_size: float, default 1.0
    :param plot_sinks: Flag to decide whether sink symbols should be drawn.
    :type plot_sinks: bool, default False
    :param plot_sources: Flag to decide whether source symbols should be drawn.
    :type plot_sources: bool, default False
    :param sink_size: Relative size of sinks to plot.
    :type sink_size: float, default 1.0
    :param source_size: Relative size of sources to plot.
    :type source_size: float, default 1.0
    :param valve_size: Relative size of valves to plot.
    :type valve_size: float, default 1.0
    :param pump_size: Relative size of pumps to plot.
    :type pump_size: float, default 1.0
    :param heat_exchanger_size: Relative size of heat_exchanger to plot.
    :type heat_exchanger_size: float, default 1.0
    :param pressure_control_size: Relative size of pres_control to plot.
    :type pressure_control_size: float, default 1.0
    :param compressor_size: Relative size of compressor to plot.
    :type compressor_size: float, default 1.0
    :param flow_control_size: Relative size of flow_control to plot.
    :type flow_control_size: float, default 1.0
    :param scale_size: Flag if junction_size, ext_grid_size, valve_size- and distance will be \
            scaled with respect to grid mean distances
    :type scale_size: bool, default True
    :param junction_color: Junction Color. See also matplotlib or seaborn documentation on how to\
            choose colors.
    :type junction_color: str, tuple, default "r"
    :param pipe_color: Pipe color
    :type pipe_color: str, tuple, default "silver"
    :param ext_grid_color: External grid color
    :type ext_grid_color: str, tuple, default "orange"
    :param valve_color: Valve Color.
    :type valve_color: str, tuple, default "silver"
    :param pump_color: Pump Color.
    :type pump_color: str, tuple, default "silver"
    :param heat_exchanger_color: Heat Exchanger Color.
    :type heat_exchanger_color: str, tuple, default "silver"
    :param pressure_control_color: Pressure Control Color.
    :type pressure_control_color: str, tuple, default "silver"
    :param compressor_color: Compressor Color.
    :type compressor_color: str, tuple, default "silver"
    :param flow_control_color: Flow Control Color.
    :type flow_control_color: str, tuple, default "silver"
    :param library: Library name to create generic coordinates (case of missing geodata). Choose\
            "igraph" to use igraph package or "networkx" to use networkx package.
    :type library: str, default "igraph"
    :param show_plot: If True, show plot at the end of plotting
    :type show_plot: bool, default True
    :param ax: matplotlib axis to plot to
    :type ax: object, default None
    :return: ax - Axes of figure

    """
    collections = create_simple_collections(net,
                                            respect_valves=respect_valves,
                                            respect_in_service=respect_in_service,
                                            pipe_width=pipe_width,
                                            junction_size=junction_size,
                                            ext_grid_size=ext_grid_size,
                                            plot_sinks=plot_sinks,
                                            plot_sources=plot_sources,
                                            sink_size=sink_size,
                                            source_size=source_size,
                                            valve_size=valve_size,
                                            pump_size=pump_size,
                                            heat_exchanger_size=heat_exchanger_size,
                                            pressure_control_size=pressure_control_size,
                                            compressor_size=compressor_size,
                                            flow_control_size=flow_control_size,
                                            scale_size=scale_size,
                                            junction_color=junction_color,
                                            pipe_color=pipe_color,
                                            ext_grid_color=ext_grid_color,
                                            valve_color=valve_color,
                                            pump_color=pump_color,
                                            heat_exchanger_color=heat_exchanger_color,
                                            pressure_control_color=pressure_control_color,
                                            compressor_color=compressor_color,
                                            flow_control_color=flow_control_color,
                                            library=library,
                                            as_dict=False, **kwargs)
    ax = draw_collections(collections, ax=ax)

    if show_plot:
        plt.show()
    return ax


def create_simple_collections(net, respect_valves=False, respect_in_service=True, pipe_width=5.0,
                              junction_size=1.0, ext_grid_size=1.0, plot_sinks=False,
                              plot_sources=False, sink_size=1.0, source_size=1.0, valve_size=1.0,
                              pump_size=1.0, heat_exchanger_size=1.0, pressure_control_size=1.0,
                              compressor_size=1.0, flow_control_size=1.0,
                              scale_size=True, junction_color="r", pipe_color='silver',
                              ext_grid_color='orange', valve_color='silver', pump_color='silver',
                              heat_exchanger_color='silver', pressure_control_color='silver',
                              compressor_color='silver', flow_control_color='silver',
                              library="igraph", as_dict=True, **kwargs):
    """
    Plots a pandapipes network as simple as possible.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the
    tutorial

    :param net: The pandapipes format network.
    :type net: pandapipesNet
    :param respect_valves: Respect valves if artificial geodata is created. \
            .. note:: This Flag is ignored if plot_line_switches is True
    :type respect_valves: bool default False
    :param respect_in_service: Respect only components which are in service.
    :type respect_in_service: bool default True
    :param pipe_width: width of pipes
    :type pipe_width: float, default 5.0
    :param junction_size: Relative size of junctions to plot. The value junction_size is multiplied\
            with mean_distance_between_buses, which equals the distance between the max geoocord\
            and the min divided by 200: \
            mean_distance_between_buses = sum((net['bus_geodata'].max() \
                                               - net['bus_geodata'].min()) / 200)
    :type junction_size: float, default 1.0
    :param ext_grid_size: Relative size of ext_grids to plot. See bus sizes for details. Note: \
            ext_grids are plottet as rectangles
    :type ext_grid_size: float, default 1.0
    :param plot_sinks: Flag to decide whether sink symbols should be drawn.
    :type plot_sinks: bool, default False
    :param plot_sources: Flag to decide whether source symbols should be drawn.
    :type plot_sources: bool, default False
    :param sink_size: Relative size of sinks to plot.
    :type sink_size: float, default 1.0
    :param source_size: Relative size of sources to plot.
    :type source_size: float, default 1.0
    :param valve_size: Relative size of valves to plot.
    :type valve_size: float, default 1.0
    :param pump_size: Relative size of pumps to plot.
    :type pump_size: float, default 1.0
    :param heat_exchanger_size: Relative size of heat_exchanger to plot.
    :type heat_exchanger_size: float, default 1.0
    :param pressure_control_size: Relative size of pres_control to plot.
    :type pressure_control_size: float, default 1.0
    :param compressor_size: Relative size of compressor to plot.
    :type compressor_size: float, default 1.0
    :param flow_control_size: Relative size of flow_control to plot.
    :type flow_control_size: float, default 1.0
    :param scale_size: Flag if junction_size, ext_grid_size, valve_size- and distance will be \
            scaled with respect to grid mean distances
    :type scale_size: bool, default True
    :param junction_color: Junction Color. See also matplotlib or seaborn documentation on how to\
            choose colors.
    :type junction_color: str, tuple, default "r"
    :param pipe_color: Pipe Color.
    :type pipe_color: str, tuple, default "silver"
    :param ext_grid_color: External Grid Color.
    :type ext_grid_color: str, tuple, default "orange"
    :param valve_color: Valve Color.
    :type valve_color: str, tuple, default "silver"
    :param pump_color: Pump Color.
    :type pump_color: str, tuple, default "silver"
    :param heat_exchanger_color: Heat Exchanger Color.
    :type heat_exchanger_color: str, tuple, default "silver"
    :param pressure_control_color: Pressure Control Color.
    :type pressure_control_color: str, tuple, default "silver"
    :param compressor_color: Compressor Color.
    :type compressor_color: str, tuple, default "silver"
    :param flow_control_color: Flow Control Color.
    :type flow_control_color: str, tuple, default "silver"
    :param library: library name to create generic coordinates (case of missing geodata). Choose\
            "igraph" to use igraph package or "networkx" to use networkx package. **NOTE**: \
            Currently the networkx implementation is not working!
    :type library: str, default "igraph"
    :param as_dict: flag whether to return dictionary for network components or just a list
    :type as_dict: bool, default True
    :return: collections - list of simple collections for the given network
    """
    # don't hide lines if switches are plotted

    # create geocoord if none are available
    if len(net.junction_geodata) == 0 and len(net.pipe_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, library=library)

    if scale_size:
        # if scale_size -> calc size from distance between min and max geocoord
        sizes = get_collection_sizes(
            net, junction_size, ext_grid_size, sink_size, source_size, valve_size, pump_size,
            heat_exchanger_size, pressure_control_size, compressor_size, flow_control_size)
        junction_size = sizes["junction"]
        ext_grid_size = sizes["ext_grid"]
        source_size = sizes["source"]

        sink_size = sizes["sink"]
        valve_size = sizes["valve"]
        pump_size = sizes["pump"]
        heat_exchanger_size = sizes["heat_exchanger"]
        pressure_control_size = sizes["pressure_control"]
        compressor_size = sizes["compressor"]
        flow_control_size = sizes["flow_control"]

    # create junction collections to plot
    junc_idx = net.junction[net.junction.in_service].index if respect_in_service \
        else net.junction.index
    junction_coll = create_junction_collection(net, junc_idx, size=junction_size,
                                               color=junction_color, zorder=10)

    # if bus geodata is available, but no line geodata
    use_junction_geodata = len(net.pipe_geodata) == 0

    if respect_in_service:
        plot_lines = net.pipe[net.pipe.in_service].index
    else:
        plot_lines = net.pipe.index

    # create line collections
    pipe_coll = create_pipe_collection(net, plot_lines, color=pipe_color, linewidths=pipe_width,
                                       use_junction_geodata=use_junction_geodata)
    collections = {"junction": junction_coll, "pipe": pipe_coll}

    # create ext_grid collections
    if respect_in_service:
        eg_junctions_with_geo_coordinates = \
            set(net.ext_grid[net.ext_grid.in_service].junction.values) \
            & set(net.junction_geodata.index)
    else:
        eg_junctions_with_geo_coordinates = set(net.ext_grid.junction.values) \
                                            & set(net.junction_geodata.index)
    if len(eg_junctions_with_geo_coordinates) > 0:
        eg_coll = create_junction_collection(
            net, eg_junctions_with_geo_coordinates, patch_type="rect", size=ext_grid_size,
            color=ext_grid_color, zorder=11)
        collections["ext_grid"] = eg_coll

    if 'source' in net and plot_sources and len(net.source) > 0:
        idx = net.source[net.source.in_service].index if respect_in_service else net.source.index
        source_colls = create_source_collection(net, sources=idx, size=source_size,
                                                patch_edgecolor='black', line_color='black',
                                                linewidths=pipe_width, orientation=0)
        collections["source"] = source_colls

    if 'sink' in net and plot_sinks and len(net.sink) > 0:
        idx = net.sink[net.sink.in_service].index if respect_in_service else net.sink.index
        sink_colls = create_sink_collection(net, sinks=idx, size=sink_size,
                                            patch_edgecolor='black', line_color='black',
                                            linewidths=pipe_width, orientation=np.pi)
        collections["sink"] = sink_colls

    if 'valve' in net:
        valve_colls = create_valve_collection(net, size=valve_size, linewidths=pipe_width,
                                              color=valve_color, respect_valves=respect_valves)
        collections["valve"] = valve_colls

    for pump_comp in [Pump, CirculationPumpPressure, CirculationPumpMass]:
        pump_tbl = pump_comp.table_name()
        if pump_tbl in net:
            fjc, tjc = pump_comp.from_to_node_cols()
            idx = net[pump_tbl][net[pump_tbl].in_service].index if respect_in_service else net[pump_tbl].index
            pump_colls = create_pump_collection(net, idx, table_name=pump_tbl,
                                                size=pump_size, linewidths=pipe_width,
                                                color=pump_color, fj_col=fjc, tj_col=tjc)
            collections[pump_tbl] = pump_colls

    if 'flow_control' in net:
        idx = net.flow_control[net.flow_control.in_service].index if respect_in_service \
            else net.flow_control.index
        flow_control_colls = create_flow_control_collection(net, flow_controllers=idx,
                                                            size=flow_control_size,
                                                            linewidths=pipe_width,
                                                            color=flow_control_color)
        collections["flow_control"] = flow_control_colls

    if 'heat_exchanger' in net:
        idx = net.heat_exchanger[net.heat_exchanger.in_service].index if respect_in_service \
            else net.heat_exchanger.index
        hxc = create_heat_exchanger_collection(net, heat_ex=idx, size=heat_exchanger_size,
                                               linewidths=pipe_width, color=heat_exchanger_color)
        collections["heat_exchanger"] = hxc

    if 'press_control' in net:
        idx = net.press_control[net.press_control.in_service].index if respect_in_service \
            else net.press_control.index
        pc = create_pressure_control_collection(net, pcs=idx, size=pressure_control_size,
                                                linewidths=pipe_width, color=pressure_control_color)
        collections["press_control"] = pc

    if 'compressor' in net:
        idx = net.compressor[net.compressor.in_service].index if respect_in_service \
            else net.compressor.index
        compr_colls = create_compressor_collection(net, idx, size=compressor_size,
                                                   linewidths=pipe_width, color=compressor_color)
        collections["compressor"] = compr_colls

    if 'additional_collections' in kwargs:
        collections["additional"] = list()
        for collection in kwargs.pop('additional_collections'):
            collections["additional"].extend(collection)

    if as_dict:
        return collections
    return list(chain.from_iterable([list(c) if hasattr(c, "__iter__") else [c]
                                     for c in collections.values()]))
