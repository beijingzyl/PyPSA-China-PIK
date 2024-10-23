# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

"""
Plots energy and cost summaries for solved networks.

Relevant Settings
-----------------

Inputs
------

Outputs
-------

Description
-----------

"""

import os
import logging
from _helpers import configure_logging

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

logger = logging.getLogger(__name__)

# consolidate and rename
def rename_techs(label):
    prefix_to_remove = [
        "central ",
        "decentral ",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "H2": "H2",
        "coal cc": "CC"
    }

    rename_if_contains = [
        "gas",
        "coal"
    ]

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "hydro_inflow": "hydroelectricity",
        "stations": "hydroelectricity",
        "AC": "transmission lines",
        "CO2 capture": "biomass carbon capture",
        "CC": "coal carbon capture",
        "battery": "battery"
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename.items():
        if old == label:
            label = new
    return label


preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "nuclear",
        "coal",
        "coal carbon capture",
        "coal power plant",
        "coal power plant retrofit",
        "coal boiler",
        "CHP coal",
        "gas",
        "OCGT",
        "gas boiler",
        "CHP gas",
        "biomass",
        # "biomass carbon capture",
        "onshore wind",
        "offshore wind",
        "solar PV",
        "solar thermal",
        "heat pump",
        "resistive heater",
        "methanation",
        "H2",
        "H2 fuel cell",
        "H2 CHP",
        "battery",
        "battery storage",
        "hot water storage",
        "hydrogen storage",
    ]
)



def plot_costs(infn, config, fn=None):

    ## For now ignore the simpl header
    cost_df = pd.read_csv(infn,index_col=list(range(3)),header=[1])

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    #convert to billions
    df = df/1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < config['plotting']['costs_plots_threshold']]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    print(df.sum())

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    new_columns = df.sum().sort_values().index

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    df.loc[new_index,new_columns].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[config['plotting']['tech_colors'][i] for i in new_index],
    )


    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([0,config['plotting']['costs_max']])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,bbox_to_anchor=[1, 1],loc="upper left")


    fig.tight_layout()

    if fn is not None:
        fig.savefig(fn, transparent=True)


def plot_energy(infn, config, fn=None):

    energy_df = pd.read_csv(infn, index_col=list(range(2)),header=[1])

    df = energy_df.groupby(energy_df.index.get_level_values(1)).sum()

    #convert MWh to TWh
    df = df/1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.abs().max(axis=1) < config['plotting']['energy_threshold']]

    logger.info(
        f"Dropping all technology with energy consumption or production below {config['plotting']['energy_threshold']} TWh/a"
    )
    logger.debug(df.loc[to_drop])

    df = df.drop(to_drop)

    logger.info(f"Total energy of {round(df.sum()[0])} TWh/a")

    new_index = preferred_order.intersection(df.index).append(
        df.index.difference(preferred_order)
    )

    new_columns = df.columns.sort_values()

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    logger.debug(df.loc[new_index, new_columns])

    df.loc[new_index,new_columns].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[config['plotting']['tech_colors'][i] for i in new_index],
    )

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([config['plotting']['energy_min'], config['plotting']['energy_max']])

    ax.set_ylabel("Energy [TWh/a]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,bbox_to_anchor=[1, 1],loc="upper left")

    fig.tight_layout()

    if fn is not None:
        fig.savefig(fn, transparent=True)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_summary',
                                   opts='ll',
                                   topology ='current+Neighbor',
                                   pathway ='exponential175',
                                   planning_horizons="2020")
    configure_logging(snakemake)

    config = snakemake.config
    wildcards = snakemake.wildcards
    logs = snakemake.log
    out = snakemake.output
    paths = snakemake.input

    Summary = ['energy', 'costs']
    summary_i = 0
    for summary in Summary:
        try:
            func = globals()[f"plot_{summary}"]
        except KeyError:
            raise RuntimeError(f"plotting function for {summary} has not been defined")
        func(os.path.join(paths[0], f"{summary}.csv"), config, out[summary_i])
        summary_i = summary_i + 1
