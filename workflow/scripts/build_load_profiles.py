"""Functions for the rules to build the hourly heat and load demand profiles
- electricity load profiles are based on scaling an hourly base year profile to yearly future
    projections
- daily heating demand is based on the degree day approx (from atlite) & upscaled hourly based
  on an intraday profile (for Denmark by default, see snakefile)
"""

import logging
import atlite
import pandas as pd
import scipy as sp
import numpy as np
from collections.abc import Iterable
from scipy.optimize import curve_fit
import os
from _helpers import configure_logging, mock_snakemake, get_cutout_params
from _pypsa_helpers import (
    make_periodic_snapshots,
    calc_atlite_heating_timeshift,
    shift_profile_to_planning_year,
)
from readers import read_yearly_load_projections

# TODO switch from hardocded REF_YEAR to a base year?
from constants import (
    PROV_NAMES,
    UNIT_HOT_WATER_START_YEAR,
    UNIT_HOT_WATER_END_YEAR,
    START_YEAR,
    END_YEAR,
    REF_YEAR,
    REGIONAL_GEO_TIMEZONES,
    TIMEZONE,
)

logger = logging.getLogger(__name__)
idx = pd.IndexSlice
nodes = pd.Index(PROV_NAMES)


# TODO this is really stupid since there are 5 hours shifts on the heating demand due to sun time
def downscale_time_data(
    dt_index: pd.DatetimeIndex,
    weekly_profile: Iterable,
    regional_tzs: pd.Series,
) -> pd.DataFrame:
    """Make hourly resolved data profiles based on exogenous weekdays and weekend profiles.
    This fn takes into account that the profiles are in local time and that regions may
     have different timezones.

    Args:
        dt_index (DatetimeIndex): the snapshots (in network local naive time) but hourly res.
        weekly_profile (Iterable): the weekly profile as a list of 7*24 entries.
        regional_tzs (pd.Series, optional): regional geographical timezones for profiles.
            Defaults to pd.Series(index=PROV_NAMES, data=list(REGIONAL_GEO_TIMEZONES.values())).

    Returns:
        pd.DataFrame: Regionally resolved profiles for each snapshot hour rperesented by dt_index
    """
    weekly_profile = pd.Series(weekly_profile, range(24 * 7))
    # make a dataframe with timestamps localized to the network TIMEZONE timestamps
    all_times = pd.DataFrame(
        dict(zip(PROV_NAMES, [dt_index.tz_localize(TIMEZONE)] * len(PROV_NAMES))),
        index=dt_index.tz_localize(TIMEZONE),
        columns=PROV_NAMES,
    )
    # then localize to regional time. _dt ensures index is not changed
    week_hours = all_times.apply(
        lambda col: col.dt.tz_convert(regional_tzs[col.name]).tz_localize(None)
    )
    # then convert into week hour & map to the intraday heat demand profile (based on local time)
    return week_hours.apply(lambda col: col.dt.weekday * 24 + col.dt.hour).apply(
        lambda col: col.map(weekly_profile)
    )


def make_heat_demand_projections(
    planning_year: int, projection_name: str, ref_year=REF_YEAR
) -> float:
    """Make projections for heating demand

    Args:
        projection_name (str): name of projection
        planning_year (int): year to project to
        ref_year (int, optional): reference year. Defaults to REF_YEAR.

    Returns:
        float: scaling factor relative to base year for heating demand
    """
    years = np.array([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2060])

    if projection_name == "positive":

        def func(x, a, b, c, d):
            """Cubic polynomial fit to proj"""
            return a * x**3 + b * x**2 + c * x + d

        # TODO soft code
        # heated area projection in China
        # 2060: 6.02 * 36.52 * 1e4 north population * floor_space_per_capita in city
        heated_area = np.array(
            [2742, 21263, 64645, 110766, 252056, 435668, 672205, 988209, 2198504]
        )  # 10000 m2

        # Perform curve fitting
        popt, pcov = curve_fit(func, years, heated_area)
        factor = func(int(planning_year), *popt) / func(REF_YEAR, *popt)

    elif projection_name == "constant":
        factor = 1.0

    else:
        raise ValueError(f"Invalid heating demand projection {projection_name}")
    return factor


def build_daily_heat_demand_profiles(
    heat_demand_config: dict,
    atlite_heating_hr_shift: int,
    switch_month_day: bool = True,
) -> pd.DataFrame:
    """build the heat demand profile according to forecast demans

    Args:
        heat_demand_config (dict): the heat demand configuration
        atlite_heating_hr_shift (int): the hour shift for heating demand, needed due to imperfect
            timezone handling in atlite
        switch_month_day (bool, optional): whether to switch month & day from heat_demand_config.
            Defaults to True.
    Returns:
        pd.DataFrame: regional daily heating demand with April to Sept forced to 0
    """
    with pd.HDFStore(snakemake.input.population_map, mode="r") as store:
        pop_map = store["population_gridcell_map"]

    cutout = atlite.Cutout(snakemake.input.cutout)
    atlite_year = get_cutout_params(snakemake.config)["weather_year"]

    pop_matrix = sp.sparse.csr_matrix(pop_map.T)
    index = pop_map.columns
    index.name = "provinces"

    # TODO clarify a bit here, maybe the po_matrix should be normalised earlier?
    # unclear whether it's per cap or not
    total_hd = cutout.heat_demand(
        matrix=pop_matrix,
        index=index,
        threshold=heat_demand_config["heating_start_temp"],
        a=heat_demand_config["heating_lin_slope"],
        constant=heat_demand_config["heating_offet"],
        # hack to bring it back to local from UTC
        hour_shift=atlite_heating_hr_shift,
    )

    regonal_daily_hd = total_hd.to_pandas().divide(pop_map.sum())
    # input given as dd-mm but loc as yyyy-mm-dd
    if switch_month_day:
        start_day = "{}-{}".format(*heat_demand_config["start_day"].split("-")[::-1])
        end_day = "{}-{}".format(*heat_demand_config["end_day"].split("-")[::-1])
    else:
        start_day = heat_demand_config["start_day"]
        end_day = heat_demand_config["end_day"]
    regonal_daily_hd.loc[f"{atlite_year}-{start_day}":f"{atlite_year}-{end_day}"] = 0

    return regonal_daily_hd


# TODO rename to make_hot_water_projections
# TODO FIX VALUES -> ref value is 2008 not 2020 :|
def build_hot_water_per_day(planning_horizons: int | str) -> pd.Series:
    """Make projections for the hot water demand increase and scale the ref year value
    NB: the ref year value is for 2008 not 2020 -> incorrect

    Args:
        planning_horizons (int | str): the planning year to which demand will be projected

    Raises:
        ValueError: if the config projection type is not supported

    Returns:
        pd.Series: regional hot water demand per day
    """

    with pd.HDFStore(snakemake.input.population, mode="r") as store:
        population_count = store["population"]

    unit_hot_water_start_yr = UNIT_HOT_WATER_START_YEAR
    unit_hot_water_end_yr = UNIT_HOT_WATER_END_YEAR

    if snakemake.wildcards["heating_demand"] == "positive":

        def func(x, a, b):
            return a * x + b

        x = np.array([START_YEAR, END_YEAR])
        y = np.array([unit_hot_water_start_yr, unit_hot_water_end_yr])
        popt, pcov = curve_fit(func, x, y)

        unit_hot_water = func(int(planning_horizons), *popt)

    elif snakemake.wildcards["heating_demand"] == "constant":
        unit_hot_water = unit_hot_water_start_yr

    elif snakemake.wildcards["heating_demand"] == "mean":

        def lin_func(x: np.array, a: float, b: float) -> np.array:
            return a * x + b

        x = np.array([START_YEAR, END_YEAR])
        y = np.array([unit_hot_water_start_yr, unit_hot_water_end_yr])
        popt, pcov = curve_fit(lin_func, x, y)
        popt, pcov = curve_fit(lin_func, x, y)

        unit_hot_water = (lin_func(int(planning_horizons), *popt) + UNIT_HOT_WATER_START_YEAR) / 2
    else:
        raise ValueError(f"Invalid heating demand type {snakemake.wildcards['heating_demand']}")
    # MWh per day per region
    hot_water_per_day = unit_hot_water * population_count / 365.0

    return hot_water_per_day


# TODO return dict would be nice
# TODO separate the projection and move it ot the main
def build_heat_demand_profile(
    daily_hd: pd.DataFrame,
    hot_water_per_day: pd.DataFrame,
    snapshots: pd.date_range,
    planning_horizons: int | str,
) -> tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame]:
    """Downscale the daily heat demand to hourly heat demand using pre-defined intraday profiles
    THIS FUNCTION ALSO MAKES PROJECTIONS FOR HEATING DEMAND - WHICH IS NOT THE CORRECT PLACE

    Args:
        daily_hd (DataFrame): the day resolved heat demand for each region (atlite time axis)
        hot_water_per_day (DataFrame): the day resolved hot water demand for each region
        snapshots (pd.date_range): the snapshots for the planning year
        planning_horizons (int | str): the planning year

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, object, pd.DataFrame]:
            heat, space_heat, space_heating_per_hdd, water_heat demands
    """
    # TODO - very strange, why would this be needed unless atlite is buggy
    daily_hd_uniq = daily_hd[~daily_hd.index.duplicated(keep="first")]
    # hourly resolution regional demand (but wrong data, it's just ffill)
    heat_demand_hourly = shift_profile_to_planning_year(
        daily_hd_uniq, planning_yr=planning_horizons
    ).reindex(index=snapshots, method="ffill")

    # ===== downscale to hourly =======
    intraday_profiles = pd.read_csv(snakemake.input.intraday_profiles, index_col=0)

    # TODO, does this work with variable frequency?
    intraday_year_profiles = downscale_time_data(
        dt_index=heat_demand_hourly.index,
        weekly_profile=(
            list(intraday_profiles["weekday"]) * 5 + list(intraday_profiles["weekend"]) * 2
        ),
        regional_tzs=pd.Series(index=PROV_NAMES, data=list(REGIONAL_GEO_TIMEZONES.values())),
    )

    # TWh -> MWh
    space_heat_demand_total = pd.read_csv(snakemake.input.space_heat_demand, index_col=0) * 1e6
    space_heat_demand_total = space_heat_demand_total.squeeze()

    # ==== SCALE TO FUTURE DEMAND ======
    # TODO soft-code ref/base year or find a better variable name
    # TODO remind coupling: fix this kind of stuff or make separate fn
    # Belongs outside of this function really and in main
    factor = make_heat_demand_projections(
        planning_horizons, snakemake.wildcards["heating_demand"], ref_year=REF_YEAR
    )
    # WOULD BE NICER TO SUM THAN TO WEIGH OR TO directly build the profile with the freq
    # TODO, does this work with variable frequency?
    space_heating_per_hdd = (space_heat_demand_total * factor) / (
        heat_demand_hourly.sum() * snakemake.config["snapshots"]["frequency"]
    )

    space_heat_demand = intraday_year_profiles.mul(heat_demand_hourly).mul(space_heating_per_hdd)
    water_heat_demand = intraday_year_profiles.mul(hot_water_per_day / 24.0)

    heat_demand = space_heat_demand + water_heat_demand

    return heat_demand, space_heat_demand, space_heating_per_hdd, water_heat_demand


def prepare_hourly_load_data(
    hourly_load_p: os.PathLike,
    prov_codes_p: os.PathLike,
) -> pd.DataFrame:
    """Read the hourly demand data and prepare it for use in the model

    Args:
        hourly_load_p (os.PathLike, optional): raw elec data from zenodo, see readme in data.
        prov_codes_p (os.PathLike, optional): province mapping for data.

    Returns:
        pd.DataFrame: the hourly demand data with the right province names, in TWh/hr
    """
    TO_TWh = 1e-6
    hourly = pd.read_csv(hourly_load_p)
    hourly_TWh = hourly.drop(columns=["Time Series"]) * TO_TWh
    prov_codes = pd.read_csv(prov_codes_p)
    prov_codes.set_index("Code", inplace=True)
    hourly_TWh.columns = hourly_TWh.columns.map(prov_codes["Full name"])
    return hourly_TWh


def project_elec_demand(
    hourly_demand_base_yr_MWh: pd.DataFrame,
    yearly_projections_MWh: pd.DataFrame,
    year=2020,
) -> pd.DataFrame:
    # 统一年份类型为字符串
    year = str(year)
    if yearly_projections_MWh.index.name == "province":
        yearly_projections_MWh.columns = yearly_projections_MWh.columns.astype(str)
    else:
        yearly_projections_MWh.index = yearly_projections_MWh.index.astype(str)
    # normalise the hourly load
    hourly_load_profile = hourly_demand_base_yr_MWh.loc[:, PROV_NAMES]
    hourly_load_profile /= hourly_load_profile.sum(axis=0)

    # 添加调试信息
    logger.info(f"yearly_projections_MWh 形状: {yearly_projections_MWh.shape}")
    logger.info(f"yearly_projections_MWh 索引: {list(yearly_projections_MWh.index)}")
    logger.info(f"yearly_projections_MWh 列: {list(yearly_projections_MWh.columns)}")
    logger.info(f"yearly_projections_MWh 列类型: {[type(c) for c in yearly_projections_MWh.columns]}")
    logger.info(f"目标年份: {year}")
    logger.info(f"目标年份类型: {type(year)}")

    if yearly_projections_MWh.index.name == "province":
        # 原始格式：省份作为索引，年份作为列
        logger.info("使用原始格式：省份作为索引，年份作为列")
        available_years = [str(y).strip() for y in yearly_projections_MWh.columns]
        target_year = str(year).strip()
        logger.info(f"target_year repr: {repr(target_year)}")
        logger.info(f"available_years repr: {[repr(y) for y in available_years]}")
        if target_year not in available_years:
            raise ValueError(f"无法找到合适的年份，目标年份 {target_year} 不在可用年份中: {available_years}")
        yearly_projections_MWh = yearly_projections_MWh.T.loc[target_year, PROV_NAMES]

    else:
        # 转置格式：年份作为索引，省份作为列
        logger.info("使用转置格式：年份作为索引，省份作为列")
        available_years = [str(y).strip() for y in yearly_projections_MWh.index]
        target_year = str(year).strip()
        logger.info(f"target_year repr: {repr(target_year)}")
        logger.info(f"available_years repr: {[repr(y) for y in available_years]}")
        if target_year not in available_years:
            raise ValueError(f"无法找到合适的年份，目标年份 {target_year} 不在可用年份中: {available_years}")
        yearly_projections_MWh = yearly_projections_MWh.loc[target_year, PROV_NAMES]
    
    hourly_load_projected = yearly_projections_MWh.multiply(hourly_load_profile)

    if len(hourly_load_projected) == 8784:
        hourly_load_projected.drop(hourly_load_projected.index[1416:1440], inplace=True)
    elif len(hourly_load_projected) != 8760:
        raise ValueError("The length of the hourly load is not 8760 or 8784 (leap year, dropped)")

    snapshots = make_periodic_snapshots(
        year=int(target_year),  # 这里用 int
        freq="1h",
        start_day_hour="01-01 00:00:00",
        end_day_hour="12-31 23:00",
        bounds="both",
    )

    hourly_load_projected.index = snapshots
    return hourly_load_projected


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "build_load_profiles",
            heating_demand="positive",
            planning_horizons="2040",
            # co2_pathway="remind_ssp2NPI",
            co2_pathway="exp175default",
            topology="Current+Neigbor",
        )

    configure_logging(snakemake, logger=logger)

    planning_horizons = snakemake.wildcards["planning_horizons"]
    config = snakemake.config

    date_range = make_periodic_snapshots(
        year=planning_horizons,
        freq="1h",
        start_day_hour=config["snapshots"]["start"],
        end_day_hour=config["snapshots"]["end"],
        bounds=config["snapshots"]["bounds"],
        tz=None,
        end_year=(None if not config["snapshots"]["end_year_plus1"] else planning_horizons + 1),
    )

    # project the electric load based on the demand
    conversion = snakemake.params.elec_load_conversion  # to MWHr
    hrly_MWh_load = prepare_hourly_load_data(
        snakemake.input.hrly_regional_ac_load, snakemake.input.province_codes
    )

    # 检查是否需要集成交通负荷
    integrate_transport_load = config["run"].get("integrate_transport_load", False)
    logger.info(f"集成交通负荷设置: {integrate_transport_load}")
    
    if integrate_transport_load:
        # 如果集成交通负荷，使用非EV负荷
        logger.info("使用非EV负荷进行小时级分解")
        try:
            non_ev_annual_load = pd.read_csv(snakemake.input.non_ev_annual_load, index_col=0)
            logger.info(f"非EV年度负荷数据形状: {non_ev_annual_load.shape}")
            logger.info(f"非EV年度负荷数据列: {list(non_ev_annual_load.columns)}")
            logger.info(f"非EV年度负荷数据索引: {list(non_ev_annual_load.index)}")
            
            # 检查是否有目标年份的数据
            # 统一 available_years 和 target_year 类型为字符串
            available_years = [str(y) for y in non_ev_annual_load.columns]
            target_year = str(planning_horizons)
            if target_year not in available_years:
                logger.error(f"非EV年度负荷数据中没有 {target_year} 年的数据")
                logger.error(f"可用年份: {available_years}")
                raise ValueError(f"缺少 {target_year} 年的数据")
            
            # 转换数据格式：将省份作为索引，年份作为列
            non_ev_annual_load_transposed = non_ev_annual_load.T
            logger.info(f"转置后数据形状: {non_ev_annual_load_transposed.shape}")
            logger.info(f"转置后数据索引: {list(non_ev_annual_load_transposed.index)}")
            logger.info(f"转置后数据列: {list(non_ev_annual_load_transposed.columns)}")
            
            # 检查PROV_NAMES是否匹配
            logger.info(f"PROV_NAMES: {PROV_NAMES}")
            missing_provinces = set(PROV_NAMES) - set(non_ev_annual_load_transposed.columns)
            extra_provinces = set(non_ev_annual_load_transposed.columns) - set(PROV_NAMES)
            logger.info(f"缺失的省份: {missing_provinces}")
            logger.info(f"多余的省份: {extra_provinces}")
            
            # 确保数据包含所有需要的省份
            if missing_provinces:
                logger.warning(f"添加缺失的省份，设置为0")
                for province in missing_provinces:
                    non_ev_annual_load_transposed[province] = 0
            
            # 只保留需要的省份
            non_ev_annual_load_transposed = non_ev_annual_load_transposed[PROV_NAMES]
            
            # 生成非EV负荷
            non_ev_projected_demand = project_elec_demand(hrly_MWh_load, non_ev_annual_load_transposed, planning_horizons)
            
            # 保存为非EV负荷文件
            with pd.HDFStore(snakemake.output.non_ev_load_hrly, mode="w", complevel=4) as store:
                store["load"] = non_ev_projected_demand
            logger.info("成功生成非EV小时级负荷")
            
            # 同时生成总负荷文件（使用非EV负荷作为总负荷，因为EV负荷将在后续步骤中单独生成并集成）
            with pd.HDFStore(snakemake.output.elec_load_hrly, mode="w", complevel=4) as store:
                store["load"] = non_ev_projected_demand
            logger.info("成功生成总负荷小时级负荷（非EV负荷）")
            
        except Exception as e:
            logger.error(f"生成非EV负荷失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    else:
        # 如果不集成交通负荷，使用原来的总负荷预测
        logger.info("使用总负荷预测进行小时级分解")
        yearly_projs = read_yearly_load_projections(snakemake.input.elec_load_projs, conversion)
        projected_demand = project_elec_demand(hrly_MWh_load, yearly_projs, planning_horizons)
        
        # 保存为总负荷文件
        with pd.HDFStore(snakemake.output.elec_load_hrly, mode="w", complevel=4) as store:
            store["load"] = projected_demand
        logger.info("成功生成总负荷小时级负荷")
        
        # 同时生成非EV负荷文件（与总负荷相同）
        with pd.HDFStore(snakemake.output.non_ev_load_hrly, mode="w", complevel=4) as store:
            store["load"] = projected_demand
        logger.info("成功生成非EV小时级负荷（与总负荷相同）")

    if config.get("heat_coupling", False):
        atlite_hour_shift = calc_atlite_heating_timeshift(date_range, use_last_ts=False)
        reg_daily_hd = build_daily_heat_demand_profiles(
            config["heat_demand"],
            atlite_heating_hr_shift=atlite_hour_shift,
            switch_month_day=True,
        )

        hot_water_per_day = build_hot_water_per_day(planning_horizons)

        heat_demand, space_heat_demand, space_heating_per_hdd, water_heat_demand = (
            build_heat_demand_profile(
                reg_daily_hd,
                hot_water_per_day,
                date_range,
                planning_horizons,
            )
        )

        with pd.HDFStore(snakemake.output.heat_demand_profile, mode="w", complevel=4) as store:
            store["heat_demand_profiles"] = heat_demand

        with pd.HDFStore(snakemake.output.energy_totals_name, mode="w") as store:
            store["space_heating_per_hdd"] = space_heating_per_hdd
            store["hot_water_per_day"] = hot_water_per_day

        logger.info("Heat demand profiles successfully built")
    else:
        with pd.HDFStore(snakemake.output.heat_demand_profile, mode="w") as store:
            store["skipped_not_heat_coupled"] = pd.Series(["Skipped, heat coupling is disabled"])
        with pd.HDFStore(snakemake.output.energy_totals_name, mode="w") as store:
            store["skipped_not_heat_coupled"] = pd.Series(["Skipped, heat coupling is disabled"])
