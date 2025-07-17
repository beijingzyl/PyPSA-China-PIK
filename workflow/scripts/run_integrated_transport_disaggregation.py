#!/usr/bin/env python3
"""
集成Gompertz模型的交通部门负荷分解脚本

该脚本将Gompertz车辆拥有量预测模型与负荷分解系统集成，
生成基于真实省份分配比例的各省份电动汽车小时负荷曲线。

主要功能：
1. 使用Gompertz模型计算各省份车辆分配比例
2. 从REMIND模型读取国家年度能源消耗数据
3. 将国家数据分解到各省份
4. 生成各省份的小时负荷曲线
"""

import sys
import logging
from pathlib import Path

# 添加脚本目录到路径
sys.path.append(str(Path(__file__).parent))

from load_disaggregation_system import LoadDisaggregationSystem, TransportProcessor
from gompertz_transport_disaggregator import GompertzTransportSpatialDisaggregator
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_ev_hourly_load_from_annual(ev_annual_load, target_year, output_file):
    """从年度EV负荷生成小时负荷"""
    logger.info(f"从年度EV负荷生成 {target_year} 年小时负荷")
    
    # 读取交通负荷曲线
    transport_profiles = pd.read_csv("results/output/normalized_profiles.csv")
    logger.info(f"读取交通负荷曲线: {transport_profiles.shape}")
    
    # 配置
    config = {
        'weekdays_per_year': 260,
        'weekends_holidays_per_year': 105,
        'weekend_ratios': {
            'Private car': 1.0290,
            'Taxi': 0.9411,
            'Official car': 1.0373,
            'Rental car': 1.1713,
            'Bus': 0.8772,
            'SPV': 1.0948
        }
    }
    
    # 准备年度能源数据 - 直接使用已分解的省份负荷
    target_year_str = str(target_year)
    
    # 检查数据格式
    logger.info(f"年度EV负荷数据形状: {ev_annual_load.shape}")
    logger.info(f"年度EV负荷数据列: {ev_annual_load.columns.tolist()}")
    logger.info(f"年度EV负荷数据索引: {ev_annual_load.index.tolist()}")
    
    # 创建小时时间索引
    hourly_index = pd.to_datetime(
        pd.date_range(start=f'{target_year}-01-01', end=f'{target_year}-12-31 23:00', freq='h')
    )
    
    # 创建按省份分解的小时负荷
    all_provinces_load = pd.DataFrame(index=hourly_index)
    
    # 遍历每个省份
    for province in ev_annual_load.index:
        if target_year_str in ev_annual_load.columns:
            energy_mwh = ev_annual_load.loc[province, target_year_str]  # 已经是MWh
            if energy_mwh > 0:
                logger.info(f"处理 {province} 的EV负荷: {energy_mwh:.2f} MWh")
                
                # 假设所有EV负荷都是Private car类型
                v_type = 'Private car'
                
                # 获取负荷曲线
                workday_col_name = f'{v_type}_workday'
                weekend_col_name = f'{v_type}_weekend & holiday'
                
                if workday_col_name not in transport_profiles.columns or weekend_col_name not in transport_profiles.columns:
                    logger.warning(f"在CSV中找不到 '{v_type}' 的负荷曲线，将跳过 {province}。")
                    continue
                
                # 计算加权平均
                ratio = config['weekend_ratios'].get(v_type, 1.0)
                total_energy_mwh = energy_mwh  # 已经是MWh
                denominator = config['weekdays_per_year'] + config['weekends_holidays_per_year'] * ratio
                
                if denominator > 0:
                    daily_energy_mwh_workday = total_energy_mwh / denominator
                    daily_energy_mwh_weekend = daily_energy_mwh_workday * ratio
                else:
                    daily_energy_mwh_workday = 0
                    daily_energy_mwh_weekend = 0
                
                # 获取负荷曲线
                workday_profile_np = transport_profiles[workday_col_name].to_numpy()
                weekend_profile_np = transport_profiles[weekend_col_name].to_numpy()
                
                # 计算小时负荷
                hours_of_year = hourly_index.hour
                is_weekday_mask = (hourly_index.dayofweek < 5)
                
                hourly_weights = np.where(
                    is_weekday_mask, 
                    workday_profile_np[hours_of_year], 
                    weekend_profile_np[hours_of_year]
                )
                
                daily_energy_map = np.where(
                    is_weekday_mask,
                    daily_energy_mwh_workday,
                    daily_energy_mwh_weekend
                )
                
                hourly_loads_for_province = daily_energy_map * hourly_weights
                all_provinces_load[province] = hourly_loads_for_province
            else:
                logger.info(f"跳过 {province}，负荷为0")
        else:
            logger.warning(f"目标年份 {target_year_str} 不在数据列中")
    
    if all_provinces_load.empty or all_provinces_load.sum().sum() == 0:
        logger.warning("没有找到EV负荷数据")
        # 创建空的负荷文件
        empty_load = pd.DataFrame(index=hourly_index)
        empty_load['total_load_mw'] = 0
        total_load = empty_load
    else:
        # 计算全国总负荷
        all_provinces_load['total_load_mw'] = all_provinces_load.sum(axis=1)
        total_load = all_provinces_load
        
        # 确保输出包含所有省份列，而不仅仅是total_load_mw
        logger.info(f"生成的EV负荷包含 {len(all_provinces_load.columns)} 列")
        logger.info(f"省份列: {[col for col in all_provinces_load.columns if col != 'total_load_mw']}")
        logger.info(f"总负荷范围: {all_provinces_load['total_load_mw'].min():.2f} - {all_provinces_load['total_load_mw'].max():.2f} MW")
    
    # 保存结果
    with pd.HDFStore(output_file, mode="w", complevel=4) as store:
        store["load"] = total_load
    
    logger.info(f"EV小时负荷已保存到: {output_file}")

def main():
    """主函数"""
    # 检查是否在Snakemake环境中运行
    if "snakemake" in globals():
        # 在Snakemake环境中运行
        target_year = snakemake.params.target_year
        ev_annual_file = snakemake.input.ev_annual_load
        transport_profiles_file = snakemake.input.transport_profiles
        ev_types_file = snakemake.input.ev_types_file
        output_file = snakemake.output.ev_hourly_load
        
        logger.info(f"在Snakemake环境中运行，目标年份: {target_year}")
        logger.info(f"输入文件: {ev_annual_file}")
        logger.info(f"输出文件: {output_file}")
        
        # 直接生成EV小时负荷
        if Path(ev_annual_file).exists():
            ev_annual_load = pd.read_csv(ev_annual_file, index_col=0)
            logger.info(f"读取年度EV负荷: {ev_annual_load.shape}")
            
            # 生成EV小时负荷
            generate_ev_hourly_load_from_annual(ev_annual_load, target_year, output_file)
            logger.info(f"EV小时负荷已保存到: {output_file}")
        else:
            logger.error(f"年度EV负荷文件不存在: {ev_annual_file}")
            raise FileNotFoundError(f"年度EV负荷文件不存在: {ev_annual_file}")
    else:
        # 独立运行模式
        logger.info("=== 开始集成Gompertz模型的交通负荷分解 ===")
        
        try:
            # 1. 初始化负荷分解系统
            logger.info("步骤1: 初始化负荷分解系统")
            config_file = "config/load_disaggregation_config.yaml"
            system = LoadDisaggregationSystem(config_file)
            
            # 2. 验证Gompertz模型状态
            logger.info("步骤2: 验证Gompertz模型状态")
            if system.gompertz_disaggregator is None:
                logger.error("Gompertz模型初始化失败，无法继续")
                return
            
            # 3. 显示Gompertz模型信息
            logger.info("步骤3: 显示Gompertz模型信息")
            target_year = system.config.config['general']['target_year']
            shares = system.get_provincial_shares(target_year)
            
            print(f"\n=== {target_year}年各省份车辆分配比例 ===")
            for province, share in sorted(shares.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {province}: {share:.4f}")
            
            # 4. 运行负荷分解
            logger.info("步骤4: 运行负荷分解")
            system.run_disaggregation(departments=['transport'])
            
            # 5. 生成EV小时负荷文件（用于PyPSA）
            logger.info("步骤5: 生成EV小时负荷文件")
            output_dir = Path(system.config.config['general']['output_dir'])
            
            # 读取年度EV负荷分解结果
            ev_annual_file = "workflow/derived_data/remind/ac_load_disagg_ev.csv"
            ev_hourly_file = None
            
            if Path(ev_annual_file).exists():
                ev_annual_load = pd.read_csv(ev_annual_file, index_col=0)
                logger.info(f"读取年度EV负荷: {ev_annual_load.shape}")
                
                # 生成EV小时负荷
                ev_hourly_file = f"workflow/derived_data/load/ev_hourly_load_{target_year}.h5"
                generate_ev_hourly_load_from_annual(ev_annual_load, target_year, ev_hourly_file)
                logger.info(f"EV小时负荷已保存到: {ev_hourly_file}")
            else:
                logger.warning(f"未找到年度EV负荷文件: {ev_annual_file}")
                
            # 6. 显示EV负荷文件信息
            if ev_hourly_file and Path(ev_hourly_file).exists():
                logger.info(f"EV小时负荷文件已生成: {ev_hourly_file}")
                try:
                    with pd.HDFStore(ev_hourly_file, mode="r") as store:
                        ev_load_data = store["load"]
                        logger.info(f"EV负荷数据形状: {ev_load_data.shape}")
                        logger.info(f"EV负荷时间范围: {ev_load_data.index.min()} 到 {ev_load_data.index.max()}")
                        if 'total_load_mw' in ev_load_data.columns:
                            logger.info(f"EV总负荷范围: {ev_load_data['total_load_mw'].min():.2f} - {ev_load_data['total_load_mw'].max():.2f} MW")
                except Exception as e:
                    logger.warning(f"读取EV负荷文件失败: {e}")
            
            # 7. 显示结果摘要
            logger.info("步骤7: 显示结果摘要")
            
            print(f"\n=== 输出文件 ===")
            print(f"全国详细负荷: {output_dir / 'transport_hourly_load_detailed.csv'}")
            print(f"全国汇总负荷: {output_dir / 'transport_hourly_load_summary.csv'}")
            print(f"省份负荷目录: {output_dir / 'provincial_loads'}")
            
            # 检查省份文件
            provincial_dir = output_dir / "provincial_loads"
            if provincial_dir.exists():
                provincial_files = list(provincial_dir.glob("transport_*_hourly_load_summary.csv"))
                print(f"已生成 {len(provincial_files)} 个省份的负荷曲线")
                
                # 显示前几个省份的文件
                for file in provincial_files[:5]:
                    print(f"  {file.name}")
                if len(provincial_files) > 5:
                    print(f"  ... 还有 {len(provincial_files) - 5} 个省份")
            
            logger.info("=== 集成Gompertz模型的交通负荷分解完成 ===")
            
        except Exception as e:
            logger.error(f"运行过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 