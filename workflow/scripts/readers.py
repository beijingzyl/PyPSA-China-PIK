"""file reading support functions"""

import os

import pandas as pd


import os
import pandas as pd

def load_h2_conversion_efficiency(remind_output_dir: str, region: str = "CHA") -> float:
    """从 REMIND 输出文件加载 H2 转换效率 (elh2)"""
    eta_file = os.path.join(remind_output_dir, "pm_eta_conv.csv")
    if not os.path.exists(eta_file):
        raise FileNotFoundError(f"效率文件不存在: {eta_file}")
    
    eta_df = pd.read_csv(eta_file)
    h2_eta = eta_df.query("all_te == 'elh2' and all_regi == @region")
    if h2_eta.empty:
        raise ValueError(f"未找到区域 {region} 的 H2 转换效率 (elh2)")
    return float(h2_eta["value"].iloc[0])


def merge_sectors_by_config(yearly_proj: pd.DataFrame, config: dict) -> pd.DataFrame:
    """按配置合并部门，并在关闭 H2 时转换为电力需求"""
    sectors_cfg = config.get("sectors", {})
    mapping = config.get("sector_mapping", {})
    
    if not sectors_cfg or not mapping:
        raise ValueError("缺少 sectors 或 sector_mapping 配置")

    # 确定要合并的部门
    # 只考虑在数据中实际存在的部门（检查数据中是否有对应的sector）
    data_sectors = set(yearly_proj["sector"].unique())
    available_sectors = []
    for k in sectors_cfg.keys():
        if k in mapping:
            # 检查mapping中定义的部门是否在数据中存在
            mapping_sectors = mapping.get(k, [])
            if any(s in data_sectors for s in mapping_sectors):
                available_sectors.append(k)
    
    available_values = [sectors_cfg[k] for k in available_sectors]
    
    print(f"数据中的部门: {sorted(data_sectors)}")
    print(f"实际可用部门: {available_sectors}, 值: {available_values}")
    
    if all(available_values):
        sectors = mapping.get("base", [])
        print(f"所有可用部门都打开，只使用基础部门: {sectors}")
    elif not any(available_values):
        sectors = sum(mapping.values(), [])
        print(f"所有可用部门都关闭，使用所有部门: {sectors}")
    else:
        sectors = mapping.get("base", [])
        for k, active in sectors_cfg.items():
            if active and k in mapping:
                sectors += mapping.get(k, [])
        print(f"部分部门打开，合并部门: {sectors}")

    merged = yearly_proj[yearly_proj["sector"].isin(sectors)].copy()
    if merged.empty:
        raise ValueError(f"未找到需要合并的部门数据: {sectors}")

    # H2 被关闭时，强制转化为电力需求
    h2_sectors = mapping.get("add_H2", [])
    if not sectors_cfg.get("add_H2", False) and any(s in merged["sector"].values for s in h2_sectors):
        print(f"H2被关闭，需要转换H2需求为电力需求")
        remind_dir = config["paths"]["remind_outpt_dir"]
        region = config["run"]["remind"]["region"]
        eta = load_h2_conversion_efficiency(remind_dir, region)
        print(f"使用H2转换效率: {eta}")
        year_cols = [c for c in merged.columns if c.isdigit()]
        for s in h2_sectors:
            if s in merged["sector"].values:
                print(f"转换部门 {s} 的数据")
                merged.loc[merged["sector"] == s, year_cols] /= eta
    else:
        print(f"H2被打开，保持H2需求原样")

    # 按省份分组求和，排除sector列
    year_cols = [c for c in merged.columns if c.isdigit()]
    result = merged.groupby("province")[year_cols].sum()
    return result


def read_yearly_load_projections(
    yearly_projections_p: os.PathLike = "resources/data/load/Province_Load_2020_2060.csv",
    conversion=1,
    config: dict = None,
) -> pd.DataFrame:
    """Prepare projections for model use

    Args:
        yearly_projections_p (os.PathLike, optional): the data path.
                Defaults to "resources/data/load/Province_Load_2020_2060.csv".
        conversion (int, optional): the conversion factor to MWh. Defaults to 1.
        config (dict, optional): configuration dictionary for sector merging. Defaults to None.

    Returns:
        pd.DataFrame: the formatted data, in MWh
    """
    yearly_proj = pd.read_csv(yearly_projections_p)
    yearly_proj.rename(columns={"Unnamed: 0": "province", "region": "province"}, inplace=True)
    
    if "province" not in yearly_proj.columns:
        raise ValueError(
            "The province (or region or unamed) column is missing in the yearly projections data"
            ". Index cannot be built"
        )
    
    # 检查是否有sector列（REMIND数据）
    if "sector" in yearly_proj.columns:
        if config is None:
            raise ValueError("Config is required when processing REMIND data with sector column")
        # 使用部门合并函数
        yearly_proj = merge_sectors_by_config(yearly_proj, config)
    else:
        # 传统数据，直接设置索引
        yearly_proj.set_index("province", inplace=True)
    
    # 重命名年份列为整数类型
    yearly_proj.rename(columns={c: int(c) for c in yearly_proj.columns if c.isdigit()}, inplace=True)

    # 打印各省加总的量
    print("=" * 50)
    print("各省加总的AC需求数据 (MWh):")
    print("=" * 50)
    print(yearly_proj)
    print("=" * 50)

    return yearly_proj * conversion
