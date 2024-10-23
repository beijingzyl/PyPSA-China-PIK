# coding: utf-8

import logging
logger = logging.getLogger(__name__)

import pandas as pd
idx = pd.IndexSlice

import pypsa
import yaml
import numpy as np
import xarray as xr

from add_existing_baseyear import add_build_year_to_new_assets
from _helpers import override_component_attrs
from functions import pro_names, offwind_nodes


def basename(x):
    return x.split("-2")[0]

def add_brownfield(n, n_p, year):

    print("adding brownfield")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    # dc_i = n.links[n.links.carrier=="DC"].index
    # n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]
    # update links
    n.links.loc[(n.links.length>0) & (n.links.lifetime==np.inf),"p_nom"] = n_p.links.loc[(n_p.links.carrier=='AC') & (n_p.links.build_year==0),"p_nom_opt"]
    n.links.loc[(n.links.length>0) & (n.links.lifetime==np.inf),"p_nom_min"] = n_p.links.loc[(n_p.links.carrier=='AC') & (n_p.links.build_year==0),"p_nom_opt"]

    if year == 2025:
        add_build_year_to_new_assets(n_p, 2020)

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):

        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(
            c.name,
            c.df.index[c.df.lifetime==np.inf]
        )
        # remove assets whose build_year + lifetime < year
        n_p.mremove(
            c.name,
            c.df.index[c.df.build_year + c.df.lifetime < year]
        )
        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, make sure threshold is compatible
        chp_heat = c.df.index[(
            c.df[attr + "_nom_extendable"]
            & c.df.index.str.contains("CHP")
        )]

        threshold = snakemake.config['existing_capacities']['threshold_capacity']

        if not chp_heat.empty:
            threshold_chp_heat = threshold*c.df.loc[chp_heat].efficiency2/c.df.loc[chp_heat].efficiency
            n_p.mremove(
                c.name,
                chp_heat[c.df.loc[chp_heat, attr + "_nom_opt"] < threshold_chp_heat]
            )

        n_p.mremove(
            c.name,
            c.df.index[c.df[attr + "_nom_extendable"] & ~c.df.index.isin(chp_heat) & (c.df[attr + "_nom_opt"] < threshold)]
        )

        # copy over assets but fix their capacity
        c.df[attr + "_nom"] = c.df[attr + "_nom_opt"]
        c.df[attr + "_nom_extendable"] = False
        c.df[attr + "_nom_max"] = np.inf

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = (
            n.component_attrs[c.name].type.str.contains("series")
            & n.component_attrs[c.name].status.str.contains("Input")
        )

        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr].set_index(n.snapshots), c.name, tattr)


    for tech in ['onwind', 'offwind', 'solar']:
        ds_tech = xr.open_dataset(snakemake.input['profile_' + tech])
        p_nom_max_initial = ds_tech['p_nom_max'].to_pandas()

        if tech == 'offwind':
            for node in offwind_nodes:
                n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == tech) & (n.generators.build_year == year),"p_nom_max"] = \
                p_nom_max_initial[node] - n_p.generators[(n_p.generators.bus == node) & (n_p.generators.carrier == tech)].p_nom_opt.sum()
        else:
            for node in pro_names:
                n.generators.loc[(n.generators.bus == node) & (n.generators.carrier == tech) & (n.generators.build_year == year),"p_nom_max"] = \
                p_nom_max_initial[node] - n_p.generators[(n_p.generators.bus == node) & (n_p.generators.carrier == tech)].p_nom_opt.sum()

    n.generators.loc[(n.generators.p_nom_max < 0), "p_nom_max"] = 0

    ## retrofit coal power plant with carbon capture
    n.generators.loc[n.generators.carrier == 'coal power plant', 'p_nom_extendable'] = True
    n.generators.loc[n.generators.index.str.contains("retrofit") & ~n.generators.index.str.contains(
        str(year)), "p_nom_extendable"] = False



#%%

if __name__ == '__main__':

    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('add_brownfield',
                                   opts='ll',
                                   topology='current+FCG',
                                   pathway='linear-275',
                                   co2_reduction='1.0',
                                   planning_horizons=2025)

    print(snakemake.input.network_p)
    logging.basicConfig(level=snakemake.config['logging_level'])

    year = int(snakemake.wildcards.planning_horizons)

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p, override_component_attrs=overrides)

    add_brownfield(n, n_p, year)

    n.export_to_netcdf(snakemake.output.network_name)
