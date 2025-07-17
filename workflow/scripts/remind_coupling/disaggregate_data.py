"""generic disaggregation development
Split steps into:

- ETL
- disagg (also an ETL op)

to be rebalanced with the remind_coupling package"""

import pandas as pd
import logging
import os.path
from typing import Dict

from rpycpl.disagg import SpatialDisaggregator
from rpycpl.etl import ETL_REGISTRY, Transformation, register_etl
from generic_etl import ETLRunner

# import needed for the capacity method to be registered
from rpycpl import capacities_etl

import setup  # sets up paths
from readers import read_yearly_load_projections
from _helpers import configure_logging

import polars as pl

# 添加Gompertz模型导入
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from gompertz_transport_disaggregator import GompertzTransportSpatialDisaggregator

logger = logging.getLogger(__name__)


# 添加 EV 年度用电量提取函数
def extract_ev_annual_energy(remind_file: str, ev_types_file: str, target_year: int) -> float:
    """从REMIND数据中提取年度EV用电量，优先用最细分子项，避免重复计算"""
    
    logger.info(f"从REMIND数据中提取 {target_year} 年EV用电量")
    
    # 读取REMIND数据
    data = pl.read_csv(remind_file, separator=';')
    
    # 过滤交通电力数据
    filtered = data.filter(
        (pl.col("Region") == "CHA") &
        (pl.col("Variable").str.starts_with("FE|Transport")) &
        (pl.col("Variable").str.contains("Electricity"))
    )
    
    # 解析车辆类型
    def get_part(s: str, index: int):
        try: 
            return s.split('|')[index]
        except IndexError: 
            return None

    # 转为pandas便于处理
    df = filtered.to_pandas()
    
    # 打印所有Variable，便于调试
    logger.info(f"所有Variable字段:")
    for v in df['Variable']:
        logger.info(f"  {v}")
    
    # 通用的细分/汇总判定函数，用所有非汇总项，避免重复且不遗漏
    def sum_subtypes(df, main_type, year):
        """
        用所有非汇总项的策略：
        - 汇总项：以|main_type|Electricity结尾的
        - 非汇总项：除了汇总项之外的所有项
        - 这样既避免重复，又不会遗漏任何细分类型
        """
        mask_type = df['Variable'].str.contains(f'|{main_type}|', regex=False)
        df_type = df[mask_type].copy()
        if df_type.empty:
            logger.warning(f"没有找到包含 {main_type} 的数据")
            return 0.0
        
        # 找汇总项（以|main_type|Electricity结尾）
        mask_summary = df_type['Variable'].str.endswith(f'|{main_type}|Electricity')
        df_non_summary = df_type[~mask_summary]  # 所有非汇总项
        
        if not df_non_summary.empty:
            total_energy = df_non_summary[str(year)].sum()
            logger.info(f"{main_type} 用所有非汇总项，{len(df_non_summary)}条，总和: {total_energy:.6f} EJ")
            logger.info(f"非汇总项详情:")
            for _, row in df_non_summary.iterrows():
                logger.info(f"  {row['Variable']}: {row[str(year)]:.6f} EJ")
            return total_energy
        else:
            # 如果没有非汇总项，用汇总项
            total_energy = df_type[mask_summary][str(year)].sum()
            logger.info(f"{main_type} 只有汇总项，总和: {total_energy:.6f} EJ")
            return total_energy

    # Bus 直接用Bus类型
    bus_energy = sum_subtypes(df, 'Bus', target_year)
    logger.info(f"Bus能源: {bus_energy:.6f} EJ")
    # Heavy
    heavy_energy = sum_subtypes(df, 'Heavy', target_year)
    logger.info(f"Heavy能源: {heavy_energy:.6f} EJ")
    # Light
    light_energy = sum_subtypes(df, 'Light', target_year)
    logger.info(f"Light能源: {light_energy:.6f} EJ")
    spv_energy = heavy_energy + light_energy
    logger.info(f"SPV能源(Heavy+Light): {spv_energy:.6f} EJ")
    # LDV
    ldv_energy = sum_subtypes(df, 'LDV', target_year)
    logger.info(f"LDV能源: {ldv_energy:.6f} EJ")
    
    # 使用EV类型文件拆分LDV
    ev_type_ratios = pd.read_excel(ev_types_file, index_col='Vehicle type')
    ldv_types_to_split = ['Private car', 'Taxi', 'Official car', 'Rental car']
    total_ev_energy = 0
    for v_type in ldv_types_to_split:
        ratio = ev_type_ratios.loc[v_type, '%'] if v_type in ev_type_ratios.index else 0
        energy = ldv_energy * ratio
        logger.info(f"{v_type}能源: {energy:.6f} EJ (比例: {ratio})")
        total_ev_energy += energy
    # 添加其他类型
    total_ev_energy += bus_energy + spv_energy
    logger.info(f"EV能源总和(EJ): {total_ev_energy:.6f} EJ")
    # 转换为MWh
    total_ev_energy_mwh = total_ev_energy * 277.778 * 1000000  # EJ to TWh to MWh
    logger.info(f"EV能源总和(MWh): {total_ev_energy_mwh:.2f} MWh")
    logger.info(f"{target_year}年总EV用电量: {total_ev_energy_mwh:.2f} MWh")
    return total_ev_energy_mwh


def get_ev_provincial_shares(target_year: int) -> Dict[str, float]:
    """使用Gompertz模型获取EV的省份分配比例"""
    try:
        # 初始化Gompertz模型
        gompertz_config = {
            'saturation_level': 500,
            'alpha': -5.58,
            'start_year': 2020,
            'end_year': 2060
        }
        disaggregator = GompertzTransportSpatialDisaggregator(gompertz_config)
        
        # 加载历史数据并拟合模型
        historical_data = disaggregator.load_historical_data()
        if not historical_data.empty:
            success = disaggregator.gompertz_model.fit_model(historical_data)
            if success:
                # 创建未来预测
                disaggregator.create_future_projections(2020, 2060)
                disaggregator.predict_vehicle_ownership_timeline()
                
                # 获取省份分配比例
                shares = disaggregator.calculate_provincial_shares(target_year)
                logger.info(f"成功获取 {target_year} 年EV省份分配比例")
                return shares
        
        logger.warning("Gompertz模型拟合失败，使用均匀分配")
        
    except Exception as e:
        logger.error(f"获取EV省份分配比例失败: {e}")
    
    # 后备方案：均匀分配
    provinces = [
        'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'InnerMongolia', 'Liaoning', 
        'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 
        'Fujian', 'Jiangxi', 'Shandong', 'Henan', 'Hubei', 'Hunan', 
        'Guangdong', 'Guangxi', 'Hainan', 'Chongqing', 'Sichuan', 'Guizhou', 
        'Yunnan', 'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
    ]
    uniform_share = 1.0 / len(provinces)
    return {province: uniform_share for province in provinces}


@register_etl("disagg_acload_ref")
def disagg_ac_using_ref(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    reference_year: int | str,
) -> pd.DataFrame:
    """Spatially Disaggregate the load using regional/nodal reference data"""
    
    regional_reference = reference_data[int(reference_year)]
    regional_reference /= regional_reference.sum()
    electricity_demand = data["loads"].query("load == 'ac'")
    electricity_demand.set_index("year", inplace=True)
    
    # 提取EV年度用电量
    target_year = int(reference_year)

    # 获取文件路径（从snakemake.input中获取）
    remind_transport_file = snakemake.input.remind_transport_data
    ev_types_file = snakemake.input.ev_types_file

    ev_annual_energy = extract_ev_annual_energy(
        remind_transport_file, 
        ev_types_file, 
        target_year
    )

    # 从总负荷中减去EV用电量
    # 修正：根据目标年份选取原始负荷
    if target_year in electricity_demand.index:
        original_total = electricity_demand.loc[target_year, "value"]
    else:
        raise ValueError(f"负荷数据中没有年份 {target_year}，可用年份为：{electricity_demand.index.tolist()}")
    non_ev_load = original_total - ev_annual_energy

    logger.info(f"原始总负荷: {original_total:.2f} MWh")
    logger.info(f"EV负荷: {ev_annual_energy:.2f} MWh")
    logger.info(f"非EV负荷: {non_ev_load:.2f} MWh")
    
    # 分别分解非EV和EV负荷
    logger.info("分解非EV负荷到区域")
    disagg_non_ev = SpatialDisaggregator().use_static_reference(
        pd.Series(non_ev_load, index=electricity_demand.index), 
        regional_reference
    )
    
    logger.info("分解EV负荷到区域")
    
    # 使用Gompertz模型获取EV的省份分配比例
    ev_provincial_shares = get_ev_provincial_shares(target_year)
    
    # 添加调试信息
    logger.info(f"ev_provincial_shares包含 {len(ev_provincial_shares)} 个省份")
    logger.info(f"ev_provincial_shares前5个: {dict(list(ev_provincial_shares.items())[:5])}")
    logger.info(f"ev_provincial_shares总和: {sum(ev_provincial_shares.values()):.6f}")
    
    # 检查reference_data的格式
    logger.info(f"regional_reference包含 {len(regional_reference)} 个省份")
    logger.info(f"regional_reference前5个: {regional_reference.head()}")
    logger.info(f"regional_reference总和: {regional_reference.sum():.6f}")
    
    # 将省份名称映射到与reference_data相同的格式
    # 创建省份名称映射字典
    province_mapping = {
        'Innermongolia': 'InnerMongolia',  # Gompertz模型使用Innermongolia，但reference_data使用InnerMongolia
        'Beijing': 'Beijing',
        'Tianjin': 'Tianjin', 
        'Hebei': 'Hebei',
        'Shanxi': 'Shanxi',
        'Liaoning': 'Liaoning',
        'Jilin': 'Jilin',
        'Heilongjiang': 'Heilongjiang',
        'Shanghai': 'Shanghai',
        'Jiangsu': 'Jiangsu',
        'Zhejiang': 'Zhejiang',
        'Anhui': 'Anhui',
        'Fujian': 'Fujian',
        'Jiangxi': 'Jiangxi',
        'Shandong': 'Shandong',
        'Henan': 'Henan',
        'Hubei': 'Hubei',
        'Hunan': 'Hunan',
        'Guangdong': 'Guangdong',
        'Guangxi': 'Guangxi',
        'Hainan': 'Hainan',
        'Chongqing': 'Chongqing',
        'Sichuan': 'Sichuan',
        'Guizhou': 'Guizhou',
        'Yunnan': 'Yunnan',
        'Tibet': 'Tibet',
        'Shaanxi': 'Shaanxi',
        'Gansu': 'Gansu',
        'Qinghai': 'Qinghai',
        'Ningxia': 'Ningxia',
        'Xinjiang': 'Xinjiang'
    }
    
    ev_regional_reference = pd.Series(0.0, index=regional_reference.index)
    
    # 使用映射字典将Gompertz模型的省份名称转换为reference_data格式
    for province, share in ev_provincial_shares.items():
        # 使用映射字典转换省份名称
        mapped_province = province_mapping.get(province, province)
        if mapped_province in ev_regional_reference.index:
            ev_regional_reference[mapped_province] = share
        else:
            # 如果省份名称仍然不匹配，记录警告
            logger.warning(f"省份 {province} (映射为 {mapped_province}) 在reference_data中未找到")
    
    # 归一化
    logger.info(f"归一化前ev_regional_reference总和: {ev_regional_reference.sum():.6f}")
    logger.info(f"归一化前ev_regional_reference非零值数量: {(ev_regional_reference > 0).sum()}")
    
    if ev_regional_reference.sum() > 0:
        ev_regional_reference /= ev_regional_reference.sum()
        logger.info(f"归一化后ev_regional_reference总和: {ev_regional_reference.sum():.6f}")
    else:
        logger.warning("EV省份分配比例全为0，使用均匀分配")
        ev_regional_reference = pd.Series(1.0/len(ev_regional_reference), index=ev_regional_reference.index)
        logger.info(f"均匀分配后ev_regional_reference总和: {ev_regional_reference.sum():.6f}")
    
    # 添加调试信息
    logger.info(f"ev_regional_reference前5个值: {ev_regional_reference.head()}")
    logger.info(f"ev_regional_reference后5个值: {ev_regional_reference.tail()}")
    
    # 使用EV专用的区域分布进行分解
    # 确保使用正确的年度EV负荷总量
    logger.info(f"EV年度负荷总量: {ev_annual_energy:.2f} MWh")
    logger.info(f"EV省份分配比例总和: {ev_regional_reference.sum():.6f}")
    
    # 使用EV专用的区域分布进行分解
    disagg_ev = SpatialDisaggregator().use_static_reference(
        pd.Series(ev_annual_energy, index=electricity_demand.index), 
        ev_regional_reference
    )
    
    # 验证分解结果
    ev_sum = disagg_ev.sum().iloc[0] if hasattr(disagg_ev.sum(), 'iloc') else disagg_ev.sum()
    logger.info(f"EV分解结果总和: {ev_sum:.2f} MWh")
    logger.info(f"EV分解结果应该等于: {ev_annual_energy:.2f} MWh")
    
    # 如果分解结果不正确，进行修正
    if abs(ev_sum - ev_annual_energy) > 1e-6:
        logger.warning(f"EV分解结果总和 ({ev_sum:.2f}) 与预期 ({ev_annual_energy:.2f}) 不符，进行修正")
        # 按比例修正
        correction_factor = ev_annual_energy / ev_sum
        disagg_ev = disagg_ev * correction_factor
        corrected_sum = disagg_ev.sum().iloc[0] if hasattr(disagg_ev.sum(), 'iloc') else disagg_ev.sum()
        logger.info(f"修正后EV分解结果总和: {corrected_sum:.2f} MWh")
    
    # 分别保存非EV和EV负荷分解结果
    logger.info("保存分别的分解结果")
    
    # 将分解结果保存到全局变量，供后续使用
    global non_ev_disagg_result, ev_disagg_result
    non_ev_disagg_result = disagg_non_ev
    ev_disagg_result = disagg_ev
    
    # 返回合并结果（保持向后兼容）
    final_disagg_load = disagg_non_ev + disagg_ev
    return final_disagg_load


def add_possible_techs_to_paidoff(paidoff: pd.DataFrame, tech_groups: pd.Series) -> pd.DataFrame:
    """Add possible PyPSA technologies to the paid off capacities DataFrame.
    The paidoff capacities are grouped in case the Remind-PyPSA tecg mapping is not 1:1
    but the network needs to add PyPSA techs.
    A constraint is added so the paid off caps per group are not exceeded.

    Args:
        paidoff (pd.DataFrame): DataFrame with paid off capacities
    Returns:
        pd.DataFrame: paid off techs with list of PyPSA technologies
    Example:
        >> tech_groups
            PyPSA_tech, group
            coal CHP, coal
            coal, coal
        >> add_possible_techs_to_paidoff(paidoff, tech_groups)
        >> paidoff
            tech_group, paid_off_capacity, techs
            coal, 1000, ['coal CHP', 'coal']
    """
    df = tech_groups.reset_index()
    possibilities = df.groupby("group").PyPSA_tech.apply(lambda x: list(x.unique()))
    paidoff["techs"] = paidoff.tech_group.map(possibilities)
    return paidoff


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = setup._mock_snakemake(
            "disaggregate_data",
            co2_pathway="SSP2-PkBudg1000-freeze",
            topology="current+FCG",
            config_files="resources/tmp/remind_coupled.yaml",
            heating_demand="positive",
        )
    configure_logging(snakemake)
    logger.info("Running disaggregation script")
    logger.debug(f"Available ETL methods: {ETL_REGISTRY.keys()}")

    params = snakemake.params
    region = params.region
    config = params.etl_cfg
    if not config:
        raise ValueError("Aborting: No REMIND data ETL config provided")

    # ================ Load data ===============
    input_files = {k: v for k, v in snakemake.input.items() if not os.path.isdir(v)}
    readers = {
        "reference_load": read_yearly_load_projections, 
        "default": pd.read_csv
    }

    # read files (and not directories) with encoding handling
    data = {}
    for k, v in input_files.items():
        try:
            if k in readers:
                data[k] = readers[k](v)
            else:
                # 检查文件扩展名来决定读取方式
                file_ext = Path(v).suffix.lower()
                if file_ext in ['.xlsx', '.xls']:
                    # Excel文件使用pd.read_excel
                    data[k] = pd.read_excel(v)
                else:
                    # CSV文件尝试不同的编码方式
                    try:
                        data[k] = readers["default"](v)
                    except UnicodeDecodeError:
                        # 如果UTF-8失败，尝试其他编码
                        try:
                            data[k] = readers["default"](v, encoding='latin-1')
                        except:
                            data[k] = readers["default"](v, encoding='cp1252')
        except Exception as e:
            logger.error(f"Error reading file {v}: {e}")
            raise

    powerplant_data = [k for k in data if k.startswith("pypsa_powerplants_")]
    data["pypsa_capacities"] = {k.split("pypsa_powerplants_")[-1]: data[k] for k in powerplant_data}
    # group techs together for harmonization
    pypsa_tech_groups = (
        data["remind_tech_groups"].set_index("PyPSA_tech")["group"].drop_duplicates()
    )
    if not pypsa_tech_groups.index.is_unique:
        raise ValueError(
            "PyPSA tech groups are not unique. Check the remind_tech_groups.csv"
            " file for remind techs that appear in multiple pypsa techs"
        )
    for cap_df in data["pypsa_capacities"].values():
        cap_df["tech_group"] = cap_df.Tech.map(pypsa_tech_groups)
        cap_df.fillna({"tech_group": ""}, inplace=True)

    logger.info(f"Loaded data files {data.keys()}")
    missing = set(input_files.keys()) - set(data.keys())
    if missing:
        logger.warning(f"Warning: Missing data files {missing}")

    # ==== transform remind data =======
    steps = config.get("disagg", [])
    results = {}
    
    # 检查是否需要集成交通负荷
    integrate_transport_load = params.get("integrate_transport_load", False)
    logger.info(f"集成交通负荷设置: {integrate_transport_load}")
    
    for step_dict in steps:
        step = Transformation(**step_dict)
        logger.info(f"Running ETL step: {step.name} with method {step.method}")
        
        if step.method == "disagg_acload_ref":
            if integrate_transport_load:
                # 如果集成交通负荷，进行完整的EV负荷分解
                logger.info("进行完整的EV负荷分解")
                result = ETLRunner.run(
                    step,
                    data,
                    reference_data=data["reference_load"],
                    reference_year=params["reference_load_year"],
                )
            else:
                # 如果不集成交通负荷，跳过EV负荷分解，直接用REMIND原始负荷进行空间分解
                logger.info("跳过EV负荷分解，直接用REMIND原始负荷进行空间分解")
                
                # 获取REMIND原始负荷数据
                loads_data = data["loads"].copy()
                
                # 确保数据格式正确，包含省份列
                if "load" in loads_data.columns:
                    # 如果数据包含load列，按load类型过滤
                    ac_load = loads_data[loads_data["load"] == "ac"].copy()
                else:
                    # 如果没有load列，假设所有数据都是ac负荷
                    ac_load = loads_data.copy()
                
                # 确保有year列
                if "year" not in ac_load.columns:
                    logger.warning("负荷数据中没有year列，使用默认年份")
                    ac_load["year"] = params["reference_load_year"]
                
                # 设置year为索引
                ac_load.set_index("year", inplace=True)
                
                # 确保数据格式与build_load_profiles期望的格式一致
                # build_load_profiles期望的格式：省份作为列，年份作为索引
                if "value" in ac_load.columns:
                    # 如果只有value列，需要按省份分解
                    logger.info("按省份分解REMIND原始负荷数据")
                    regional_reference = data["reference_load"][params["reference_load_year"]]
                    regional_reference /= regional_reference.sum()
                    
                    # 使用空间分解器分解REMIND原始负荷（不分离EV和非EV）
                    disagg = SpatialDisaggregator().use_static_reference(
                        ac_load["value"], regional_reference
                    )
                    result = disagg
                else:
                    # 如果已经有省份列，直接使用
                    result = ac_load
        elif step.method == "harmonize_capacities":
            # TODO loop over years
            result = ETLRunner.run(
                step, data["pypsa_capacities"], remind_capacities=data["remind_caps"]
            )
        elif step.method == "calc_paid_off_capacity":
            result = ETLRunner.run(
                step, data["remind_caps"], harmonized_pypsa_caps=results["harmonize_model_caps"]
            )
        else:
            result = ETLRunner.run(step, data)

        results[step.name] = result

    # TODO export, fix index
    logger.info("\n\nExporting results")
    outp_files = dict(snakemake.output.items())
    logger.info(f"Output files: {outp_files}")
    if "disagg_load" in results:
        logger.info(f"Exporting disaggregated load to {outp_files['disagg_load']}")
        results["disagg_load"].to_csv(
            outp_files["disagg_load"],
        )
        
        # 保存分别的分解结果
        if integrate_transport_load and 'non_ev_disagg_result' in globals() and 'ev_disagg_result' in globals():
            # 如果集成交通负荷，保存分别的分解结果
            non_ev_file = outp_files["disagg_load"].replace(".csv", "_non_ev.csv")
            logger.info(f"Exporting non-EV disaggregated load to {non_ev_file}")
            non_ev_disagg_result.to_csv(non_ev_file)
            
            # 保存EV负荷分解结果（用于load_disaggregation_system.py）
            ev_file = outp_files["disagg_load"].replace("ac_load_disagg.csv", "ac_load_disagg_ev.csv")
            logger.info(f"Exporting EV disaggregated load to {ev_file}")
            ev_disagg_result.to_csv(ev_file)
            
            # 保存合并结果（保持向后兼容）
            combined_file = outp_files["disagg_load"]
            logger.info(f"Exporting combined disaggregated load to {combined_file}")
            (non_ev_disagg_result + ev_disagg_result).to_csv(combined_file)
        else:
            # 如果不集成交通负荷，生成空的EV负荷文件
            ev_file = outp_files["disagg_load"].replace("ac_load_disagg.csv", "ac_load_disagg_ev.csv")
            logger.info(f"Creating empty EV load file: {ev_file}")
            # 创建一个空的DataFrame，格式与正常EV负荷文件相同
            empty_ev_df = pd.DataFrame()
            empty_ev_df.to_csv(ev_file)
            logger.info(f"Empty EV load file created: {ev_file}")
    if "harmonize_model_caps" in results:
        logger.info("Exporting harmonized model capacities")
        for year, df in results["harmonize_model_caps"].items():
            logger.info(f"Exporting harmonized capacities for year {year}")
            df.to_csv(outp_files[f"caps_{year}"], index=False)

    if "available_cap" in results:
        logger.info("Exporting paid off capacities")
        paid_off = results["available_cap"].copy()
        paid_off = add_possible_techs_to_paidoff(paid_off, pypsa_tech_groups)
        paid_off.to_csv(outp_files["paid_off"], index=False)
