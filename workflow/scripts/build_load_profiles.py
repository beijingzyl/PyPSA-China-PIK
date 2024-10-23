import logging
from _helpers import configure_logging

import atlite
import pytz

from functions import pro_names
import pandas as pd
import scipy as sp
import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

idx = pd.IndexSlice

nodes = pd.Index(pro_names)

def generate_periodic_profiles(dt_index=None,col_tzs=pd.Series(index=pro_names, data=len(pro_names)*['Shanghai']),weekly_profile=range(24*7)):
    """Give a 24*7 long list of weekly hourly profiles, generate this
    for each country for the period dt_index, taking account of time
    zones and Summer Time."""


    weekly_profile = pd.Series(weekly_profile,range(24*7))

    week_df = pd.DataFrame(index=dt_index,columns=col_tzs.index)
    for ct in col_tzs.index:
        week_df[ct] = [24*dt.weekday()+dt.hour for dt in dt_index.tz_convert(pytz.timezone("Asia/{}".format(col_tzs[ct])))]
        week_df[ct] = week_df[ct].map(weekly_profile)
    return week_df


def build_daily_heat_demand_profiles():
    with pd.HDFStore(snakemake.input.population_map, mode='r') as store:
        pop_map = store['population_gridcell_map']

    cutout = atlite.Cutout(snakemake.input.cutout)

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    hd = cutout.heat_demand(matrix=pop_matrix,
                            index=index,
                            threshold=15.,
                            a=1.,
                            constant=0.,
                            hour_shift=8.)

    daily_hd = hd.to_pandas().divide(pop_map.sum())
    daily_hd.loc['2020-04-01':'2020-09-30'] = 0

    return daily_hd

def build_hot_water_per_day(planning_horizons):
    with pd.HDFStore(snakemake.input.population, mode='r') as store:
        population_count = store['population']

    # In 2008 China 228.4 Twh for urban residential DHW
    # MWh/capita/year = 228.4 * 1e6 / 62403/1e4 = 0.366008
    unit_hot_water_2020 = 0.366008

    # We can consider that, on average,
    # the 52 M in developed countries is around 1000 kWh per person
    # http://www.estif.org/fileadmin/estif/content/publications/downloads/UNEP_2015/factsheet_single_family_houses_v05.pdf
    unit_hot_water_2060 = 1.0

    if snakemake.wildcards.heating_demand == 'positive':
        def func(x, a, b):
            return a * x + b

        x = np.array([2020, 2060])
        y = np.array([unit_hot_water_2020, unit_hot_water_2060])
        popt, pcov = curve_fit(func, x, y)

        unit_hot_water = func(int(planning_horizons), *popt)

    if snakemake.wildcards.heating_demand == 'constant':
        unit_hot_water = unit_hot_water_2020

    if snakemake.wildcards.heating_demand == 'mean':
        def func(x, a, b):
            return a * x + b

        x = np.array([2020, 2060])
        y = np.array([unit_hot_water_2020, unit_hot_water_2060])
        popt, pcov = curve_fit(func, x, y)

        unit_hot_water = (func(int(planning_horizons), *popt) + unit_hot_water_2020)/2

        # MWh per day
    hot_water_per_day = unit_hot_water * population_count / 365.

    return hot_water_per_day


def build_heat_demand_profile(daily_hd,hot_water_per_day,date_range,planning_horizons):
    h = daily_hd
    h_n = h[~h.index.duplicated(keep='first')].iloc[:-1, :]
    h_n.index = h_n.index.tz_localize('Asia/shanghai')
    date_range_2020 = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq="1H",tz='Asia/shanghai')
    date_range_2020 = date_range_2020.map(lambda t: t.replace(year=int(2020)))
    heat_demand_hdh = h_n.reindex(index=date_range_2020, method="ffill")
    heat_demand_hdh.index = date_range

    intraday_profiles = pd.read_csv(snakemake.input.intraday_profiles,index_col=0)
    intraday_year_profiles = generate_periodic_profiles(dt_index=heat_demand_hdh.index, weekly_profile=(list(intraday_profiles["weekday"])*5 + list(intraday_profiles["weekend"])*2))

    space_heat_demand_total = pd.read_csv(snakemake.input.space_heat_demand, index_col=0)
    space_heat_demand_total = space_heat_demand_total * 1e6
    space_heat_demand_total = space_heat_demand_total.squeeze()

    if snakemake.wildcards.heating_demand == 'positive':
        def func(x, a, b, c, d):
            return a * x ** 3 + b * x ** 2 + c * x + d

        # heated area in China
        # 2060: 6.02 * 36.52 * 1e4 north population * floor_space_per_capita in city
        x = np.array([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2060])
        y = np.array([2742, 21263, 64645, 110766, 252056, 435668, 672205, 988209, 2198504]) # 10000 m2

        # Perform curve fitting
        popt, pcov = curve_fit(func, x, y)
        factor = func(int(planning_horizons), *popt)/func(2020, *popt)

    if snakemake.wildcards.heating_demand == 'constant':
        factor = 1.0

    if snakemake.wildcards.heating_demand == 'mean':
        def func(x, a, b, c, d):
            return a * x ** 3 + b * x ** 2 + c * x + d

        # heated area in China
        # 2060: 6.02 * 36.52 * 1e4 north population * floor_space_per_capita in city
        x = np.array([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2060])
        y = np.array([2742, 21263, 64645, 110766, 252056, 435668, 672205, 988209, 2198504]) # 10000 m2

        # Perform curve fitting
        popt, pcov = curve_fit(func, x, y)
        factor = (func(int(planning_horizons), *popt)/func(2020, *popt) + 1.0)/2

    space_heating_per_hdd = (space_heat_demand_total * factor)  / (heat_demand_hdh.sum() * snakemake.config['frequency'])

    space_heat_demand = intraday_year_profiles.mul(heat_demand_hdh).mul(space_heating_per_hdd)
    water_heat_demand = intraday_year_profiles.mul(hot_water_per_day/24.)

    heat_demand = space_heat_demand + water_heat_demand

    return heat_demand, space_heat_demand, water_heat_demand


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_load_profiles',
                                   heating_demand='positive',
                                   planning_horizons="2020")
    configure_logging(snakemake)

    daily_hd = build_daily_heat_demand_profiles()

    config = snakemake.config
    planning_horizons = snakemake.wildcards['planning_horizons']
    hot_water_per_day = build_hot_water_per_day(planning_horizons)

    date_range = pd.date_range('2025-01-01 00:00', '2025-12-31 23:00', freq="1H",tz='Asia/shanghai')
    date_range = date_range.map(lambda t: t.replace(year=int(planning_horizons)))

    heat_demand, space_heat_demand, water_heat_demand = build_heat_demand_profile(daily_hd,hot_water_per_day,date_range,planning_horizons)

    with pd.HDFStore(snakemake.output.heat_demand_profile, mode='w', complevel=4) as store:
        store['heat_demand_profiles'] = heat_demand


