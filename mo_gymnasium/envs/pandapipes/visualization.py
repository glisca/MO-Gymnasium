import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandapipes as pp
import pandas as pd

import envs.pandapipes.ikigas_plots as plot

from PIL import Image

from envs.pandapipes.ikigas_plots import create_mass_storage_collection

plt.rcParams['image.cmap']='coolwarm'

class GasNetworkVisualizer:

    def __init__(self, gas_network=None, verbose=False):

        self._verbose = verbose
        self._gas_network = gas_network

    def get_network_image(self, gas_network):

        self._gas_network = gas_network

        old_backend = matplotlib.rcParams['backend']
        matplotlib.use('Agg')
        fig, ax = plt.subplots(figsize=(16,7))
        self._plot_gas_network(ax)
        img = self._fig2img(fig)
        matplotlib.use(old_backend)
        return img

    def _plot_gas_network(self, ax, draw_cb=True):
        # plot.simple_plot(line, plot_sinks=True, plot_sources=True, junction_color="blue", pipe_color="black")
        # do it step instead

        # create a list of simple collections
        simple_collections = plot.create_simple_collections(
            self._gas_network, as_dict=True, plot_sinks=True, plot_sources=True,
            pipe_width=2.0, pipe_color="black", junction_color="silver", sink_size=2.0, source_size=2.0)

        # convert dict to values
        pipe_collection = simple_collections["pipe"]
        pipe_collection.set_colors(None)
        pipe_collection.set_array(self._gas_network.res_pipe["mdot_to_kg_per_s"])
        pipe_collection.set_linewidths(5.)
        simple_collections = list(simple_collections.values())

        # add additional collections to the list
        junction_mass_storage_collection = plot.create_junction_collection(
            self._gas_network, junctions=[2], patch_type="rect", size=0.05, color="green", zorder=200)

        mass_storage_collection = self._get_mass_storage_collections()

        #source_colls = create_source_collection(net, sources=idx, size=source_size,
        #                                                patch_edgecolor='black', line_color='black',
        #                                                linewidths=pipe_width, orientation=0)
        simple_collections.append(mass_storage_collection)
        simple_collections.append([junction_mass_storage_collection])

        plot.draw_collections(simple_collections, ax = ax)

        self._write_pipe_labels()
        self._write_inflow_labels()
        self._write_outflow_labels()

        if draw_cb:
            axcb = plt.colorbar(pipe_collection, ax = ax, boundaries = np.linspace(-0.1,0.1,1000))
        else:
            axcb = None
        return ax, axcb

    # write the mass_flow as a label
    # maybe the length below
    def _write_pipe_labels(self):
        for i in self._gas_network.pipe.index:
            geodata = self._gas_network.pipe_geodata.loc[i].coords
            x, y = center_gravity = np.mean(geodata, axis=0)
            mass_flow = self._gas_network.res_pipe.loc[i]["mdot_to_kg_per_s"]
            plt.text(x, y+.05, "$ \dot{m} = $"+ f"{np.round(mass_flow, 2)} kg/s", fontsize=15, horizontalalignment='center')

    def _write_inflow_labels(self):
        for i in self._gas_network.source.index:
            geodata = self._gas_network.junction_geodata.loc[self._gas_network.source.loc[0].junction]
            x, y = geodata
            y -= 0.25
            mass_flow = self._gas_network.res_source.loc[i]["mdot_kg_per_s"]
            plt.text(x, y, "Inflow:\n $\dot{m} = $"+ f"{np.round(mass_flow, 2)} kg/s", fontsize=15, horizontalalignment='center')

    def _write_outflow_labels(self):
        for i in self._gas_network.sink.index:
            geodata = self._gas_network.junction_geodata.loc[self._gas_network.sink.loc[0].junction]
            x, y = geodata
            y += 0.15
            mass_flow = self._gas_network.res_sink.loc[i]["mdot_kg_per_s"]
            plt.text(x, y, "Outflow:\n $\dot{m} = $"+ f"{np.round(mass_flow, 2)} kg/s", fontsize=15, horizontalalignment='center')

    def _get_mass_storage_collections(self, respect_in_service = False):
        if len(self._gas_network.mass_storage) > 0:
            idx = self._gas_network.mass_storage[self._gas_network.mass_storage.in_service].index if respect_in_service else self._gas_network.mass_storage.index

            storage_colls = create_mass_storage_collection(
                self._gas_network, mass_storages=idx, size=0.08, patch_edgecolor='black', line_color='black', linewidths=2.0, orientation=0)
        return storage_colls

    def _fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
