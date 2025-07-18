#!/usr/bin/env python3
"""
交通负荷集成脚本

该脚本将交通部门分解得到的负荷曲线集成到现有的负荷系统中，
实现年份匹配、省份匹配和负荷相加的自动化处理。

主要功能：
1. 读取现有负荷数据
2. 读取交通分解负荷数据
3. 进行年份匹配和省份匹配
4. 将交通负荷添加到现有负荷中
5. 保存集成后的负荷数据
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import polars as pl

# 添加脚本目录到路径
sys.path.append(str(Path(__file__).parent))

from _helpers import configure_logging, mock_snakemake
from constants import PROV_NAMES

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransportLoadIntegrator:
    """交通负荷集成器"""
    
    def __init__(self, config: Dict):
        """
        初始化集成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.planning_year = int(config.get('planning_horizons', 2060))
        self.provinces = PROV_NAMES
        
    def load_existing_load_data(self, load_file: str) -> pd.DataFrame:
        """
        加载现有负荷数据
        
        Args:
            load_file: 负荷文件路径
            
        Returns:
            pd.DataFrame: 现有负荷数据
        """
        logger.info(f"加载现有负荷数据: {load_file}")
        
        try:
            with pd.HDFStore(load_file, mode="r") as store:
                load_data = store["load"]
                logger.info(f"现有负荷数据形状: {load_data.shape}")
                logger.info(f"现有负荷数据列: {list(load_data.columns)}")
                logger.info(f"现有负荷数据时间范围: {load_data.index.min()} 到 {load_data.index.max()}")
                return load_data
        except Exception as e:
            logger.error(f"加载现有负荷数据失败: {e}")
            raise
    
    def load_transport_load_data(self, transport_file: str) -> pd.DataFrame:
        """
        加载交通负荷数据
        
        Args:
            transport_file: 交通负荷文件路径
            
        Returns:
            pd.DataFrame: 交通负荷数据
        """
        logger.info(f"加载交通负荷数据: {transport_file}")
        
        # 如果文件路径为空或不存在，返回空的DataFrame
        if not transport_file or transport_file.strip() == "":
            logger.info("交通负荷文件路径为空，跳过交通负荷加载")
            return pd.DataFrame()
        
        try:
            transport_data = pd.read_csv(transport_file, index_col=0, parse_dates=True)
            logger.info(f"交通负荷数据形状: {transport_data.shape}")
            logger.info(f"交通负荷数据列: {list(transport_data.columns)}")
            logger.info(f"交通负荷数据时间范围: {transport_data.index.min()} 到 {transport_data.index.max()}")
            return transport_data
        except Exception as e:
            logger.error(f"加载交通负荷数据失败: {e}")
            return pd.DataFrame()
    
    def match_provinces(self, existing_load: pd.DataFrame, transport_load: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        匹配省份，找出共同省份、缺失省份和多余省份
        
        Args:
            existing_load: 现有负荷数据
            transport_load: 交通负荷数据
            
        Returns:
            Tuple[List[str], List[str], List[str]]: (共同省份, 缺失省份, 多余省份)
        """
        existing_provinces = set(existing_load.columns)
        transport_provinces = set(transport_load.columns)
        
        # 找出共同省份
        common_provinces = existing_provinces.intersection(transport_provinces)
        
        # 找出缺失省份（在交通负荷中但不在现有负荷中）
        missing_provinces = transport_provinces - existing_provinces
        
        # 找出多余省份（在现有负荷中但不在交通负荷中）
        extra_provinces = existing_provinces - transport_provinces
        
        logger.info(f"共同省份 ({len(common_provinces)}): {sorted(common_provinces)}")
        logger.info(f"缺失省份 ({len(missing_provinces)}): {sorted(missing_provinces)}")
        logger.info(f"多余省份 ({len(extra_provinces)}): {sorted(extra_provinces)}")
        
        return list(common_provinces), list(missing_provinces), list(extra_provinces)
    
    def match_time_periods(self, existing_load: pd.DataFrame, transport_load: pd.DataFrame) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        匹配时间周期
        
        Args:
            existing_load: 现有负荷数据
            transport_load: 交通负荷数据
            
        Returns:
            Tuple[pd.DatetimeIndex, pd.DatetimeIndex]: (现有负荷时间索引, 交通负荷时间索引)
        """
        existing_time = existing_load.index
        transport_time = transport_load.index
        
        logger.info(f"现有负荷时间范围: {existing_time.min()} 到 {existing_time.max()}")
        logger.info(f"交通负荷时间范围: {transport_time.min()} 到 {transport_time.max()}")
        
        # 检查时间范围是否匹配
        if existing_time.min() != transport_time.min() or existing_time.max() != transport_time.max():
            logger.warning("时间范围不匹配，将进行时间对齐")
            
            # 找出共同的时间范围
            common_start = max(existing_time.min(), transport_time.min())
            common_end = min(existing_time.max(), transport_time.max())
            
            logger.info(f"共同时间范围: {common_start} 到 {common_end}")
            
            # 重新索引到共同时间范围
            existing_load_aligned = existing_load.loc[common_start:common_end]
            transport_load_aligned = transport_load.loc[common_start:common_end]
            
            return existing_load_aligned.index, transport_load_aligned.index
        else:
            return existing_time, transport_time
    
    def integrate_loads(self, existing_load: pd.DataFrame, transport_load: pd.DataFrame, 
                       common_provinces: List[str]) -> pd.DataFrame:
        """
        集成负荷数据
        
        Args:
            existing_load: 现有负荷数据
            transport_load: 交通负荷数据
            common_provinces: 共同省份列表
            
        Returns:
            pd.DataFrame: 集成后的负荷数据
        """
        logger.info("开始集成负荷数据")
        
        # 创建集成后的负荷数据
        integrated_load = existing_load.copy()
        
        # 对每个共同省份进行负荷集成
        for province in common_provinces:
            if province in transport_load.columns:
                # 获取交通负荷数据 - 优先使用省份特定的负荷
                transport_province_load = transport_load[province]
                
                # 将交通负荷添加到现有负荷中
                integrated_load[province] = integrated_load[province] + transport_province_load
                
                logger.info(f"已集成 {province} 的交通负荷")
                
                # 记录负荷变化统计
                original_mean = existing_load[province].mean()
                transport_mean = transport_province_load.mean()
                integrated_mean = integrated_load[province].mean()
                
                logger.info(f"{province} 负荷统计:")
                logger.info(f"  原始平均负荷: {original_mean:.2f} MW")
                logger.info(f"  交通平均负荷: {transport_mean:.2f} MW")
                logger.info(f"  集成后平均负荷: {integrated_mean:.2f} MW")
                logger.info(f"  负荷增加比例: {((integrated_mean - original_mean) / original_mean * 100):.2f}%")
        
        return integrated_load
    
    def save_integrated_load(self, integrated_load: pd.DataFrame, output_file: str):
        """
        保存集成后的负荷数据
        
        Args:
            integrated_load: 集成后的负荷数据
            output_file: 输出文件路径
        """
        logger.info(f"保存集成后的负荷数据: {output_file}")
        
        try:
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为HDF5格式
            with pd.HDFStore(output_file, mode="w", complevel=4) as store:
                store["load"] = integrated_load
            
            logger.info(f"成功保存集成负荷数据到: {output_file}")
            
            # 输出统计信息
            logger.info("集成负荷数据统计:")
            logger.info(f"  时间范围: {integrated_load.index.min()} 到 {integrated_load.index.max()}")
            logger.info(f"  省份数量: {len(integrated_load.columns)}")
            logger.info(f"  时间步数: {len(integrated_load)}")
            logger.info(f"  总负荷范围: {integrated_load.sum(axis=1).min():.2f} - {integrated_load.sum(axis=1).max():.2f} MW")
            
        except Exception as e:
            logger.error(f"保存集成负荷数据失败: {e}")
            raise
    
    def verify_total_energy(self, integrated_load: pd.DataFrame, remind_loads_file: str, target_year: int) -> bool:
        """
        验证集成负荷的总电量是否与原始REMIND总电力负荷一致
        
        Args:
            integrated_load: 集成后的负荷数据
            remind_loads_file: REMIND负荷数据文件路径（已处理）
            target_year: 目标年份
            
        Returns:
            bool: 是否一致
        """
        logger.info("=== 验证总电量一致性 ===")
        
        # 1. 计算集成负荷的总电量（MWh）
        integrated_total_energy = integrated_load.sum().sum()  # 所有省份、所有时间的总和
        logger.info(f"集成负荷总电量: {integrated_total_energy:.2f} MWh")
        
        # 2. 从处理好的REMIND负荷数据中获取总电力负荷
        try:
            # 读取REMIND负荷数据（已处理）
            loads_data = pd.read_csv(remind_loads_file)
            
            # 过滤出目标年份的总电力负荷
            target_year_loads = loads_data[loads_data['year'] == target_year]
            ac_loads = target_year_loads[target_year_loads['load'] == 'ac']
            
            if not ac_loads.empty:
                original_energy_mwh = ac_loads['value'].sum()
                logger.info(f"原始REMIND总电力负荷: {original_energy_mwh:.2f} MWh")
                
                # 3. 比较差异
                difference = abs(integrated_total_energy - original_energy_mwh)
                relative_difference = (difference / original_energy_mwh) * 100
                
                logger.info(f"绝对差异: {difference:.2f} MWh")
                logger.info(f"相对差异: {relative_difference:.4f}%")
                
                # 4. 判断是否一致（允许1%的误差）
                if relative_difference <= 1.0:
                    logger.info("✅ 总电量一致性检查通过")
                    return True
                else:
                    logger.warning(f"❌ 总电量一致性检查失败，差异过大: {relative_difference:.4f}%")
                    return False
            else:
                logger.warning(f"❌ 在REMIND负荷数据中未找到 {target_year} 年的总电力负荷")
                return False
                
        except Exception as e:
            logger.error(f"❌ 验证总电量时出错: {e}")
            return False

    def run_integration(self, existing_load_file: str, transport_load_file: str, 
                       output_file: str, ev_load_file: str, original_remind_file: str = None) -> pd.DataFrame:
        """
        运行负荷集成流程
        
        Args:
            existing_load_file: 现有负荷文件路径
            transport_load_file: 交通负荷文件路径
            output_file: 输出文件路径
            ev_load_file: EV负荷文件路径（可选）
            original_remind_file: 原始REMIND数据文件路径（用于验证）
            
        Returns:
            pd.DataFrame: 集成后的负荷数据
        """
        logger.info("=== 开始交通负荷集成流程 ===")
        
        # 加载现有负荷数据
        existing_load = self.load_existing_load_data(existing_load_file)
        
        # 加载EV负荷数据
        ev_load = None
        if ev_load_file and Path(ev_load_file).exists():
            logger.info(f"加载EV负荷数据: {ev_load_file}")
            try:
                with pd.HDFStore(ev_load_file, mode="r") as store:
                    ev_load = store["load"]
                logger.info(f"EV负荷数据形状: {ev_load.shape}")
            except Exception as e:
                logger.warning(f"加载EV负荷数据失败: {e}")
                ev_load = None
        
        # 只有当交通负荷文件存在且不为空时才加载
        transport_load = pd.DataFrame()
        if transport_load_file and transport_load_file.strip():
            transport_load = self.load_transport_load_data(transport_load_file)
        
        # 检查是否有任何负荷数据需要集成
        has_transport = not transport_load.empty
        has_ev = ev_load is not None
        
        if not has_transport and not has_ev:
            logger.info("交通负荷和EV负荷数据都为空，直接使用现有负荷数据")
            integrated_load = existing_load.copy()
        else:
            # 2. 集成交通负荷
            if has_transport:
                # 匹配省份
                common_provinces, missing_provinces, extra_provinces = self.match_provinces(
                    existing_load, transport_load
                )
                
                if not common_provinces:
                    logger.warning("没有找到共同省份，跳过交通负荷集成")
                    integrated_load = existing_load.copy()
                else:
                    # 匹配时间周期
                    existing_time, transport_time = self.match_time_periods(existing_load, transport_load)
                    
                    # 重新索引数据到共同时间范围
                    existing_load_aligned = existing_load.loc[existing_time]
                    transport_load_aligned = transport_load.loc[transport_time]
                    
                    # 集成交通负荷
                    integrated_load = self.integrate_loads(
                        existing_load_aligned, transport_load_aligned, common_provinces
                    )
            else:
                integrated_load = existing_load.copy()
            
            # 3. 集成EV负荷
            if has_ev:
                logger.info("开始集成EV负荷")
                
                # 匹配省份
                ev_common_provinces, ev_missing_provinces, ev_extra_provinces = self.match_provinces(
                    integrated_load, ev_load
                )
                
                if ev_common_provinces:
                    # 匹配时间周期
                    integrated_time, ev_time = self.match_time_periods(integrated_load, ev_load)
                    
                    # 重新索引数据到共同时间范围
                    integrated_load_aligned = integrated_load.loc[integrated_time]
                    ev_load_aligned = ev_load.loc[ev_time]
                    
                    # 集成EV负荷
                    integrated_load = self.integrate_loads(
                        integrated_load_aligned, ev_load_aligned, ev_common_provinces
                    )
                else:
                    logger.warning("没有找到EV负荷的共同省份，跳过EV负荷集成")
        
        # 5. 验证总电量一致性
        if original_remind_file and Path(original_remind_file).exists():
            target_year = self.planning_year
            is_consistent = self.verify_total_energy(integrated_load, original_remind_file, target_year)
            
            if not is_consistent:
                logger.error("❌ 总电量一致性检查失败，请检查数据")
                # 可以选择是否继续或抛出异常
                # raise ValueError("总电量一致性检查失败")
            else:
                logger.info("✅ 总电量一致性检查通过，数据正确")
        else:
            logger.warning("⚠️ 未提供原始REMIND数据文件，跳过总电量验证")
        
        # 6. 保存结果
        self.save_integrated_load(integrated_load, output_file)
        
        logger.info("=== 交通负荷集成流程完成 ===")
        
        return integrated_load

def main():
    """主函数"""
    # 直接使用 snakemake 变量（在 Snakemake 环境中会自动存在）
    configure_logging(snakemake)
    logger.info("在Snakemake环境中运行")
    logger.info(f"输入文件: {snakemake.input}")
    logger.info(f"输出文件: {snakemake.output}")
    logger.info(f"配置: {snakemake.config}")
    
    # 获取配置
    config = snakemake.config
    planning_year = int(snakemake.wildcards.planning_horizons)
    
    # 创建集成器
    integrator = TransportLoadIntegrator({
        'planning_horizons': planning_year,
        'provinces': PROV_NAMES
    })
    
    # 检查是否有EV负荷文件输入
    ev_load_file = None
    if hasattr(snakemake.input, 'ev_hourly_load'):
        ev_load_file = snakemake.input.ev_hourly_load
        logger.info(f"EV负荷文件: {ev_load_file}")
    
    # 根据配置选择输入文件
    integrate_transport_load = snakemake.config["run"].get("integrate_transport_load", False)
    
    if integrate_transport_load:
        # 如果集成交通负荷，使用非EV负荷
        existing_load_file = snakemake.input.non_ev_load
        logger.info(f"使用非EV负荷: {existing_load_file}")
    else:
        # 如果不集成交通负荷，使用总负荷
        existing_load_file = snakemake.input.elec_load
        logger.info(f"使用总负荷: {existing_load_file}")
    
    # 获取REMIND负荷数据文件路径（用于验证）
    remind_loads_file = snakemake.input.remind_loads
    logger.info(f"REMIND负荷数据文件: {remind_loads_file}")
    
    # 运行集成流程
    integrated_load = integrator.run_integration(
        existing_load_file, "", snakemake.output.integrated_load, ev_load_file, remind_loads_file
    )
    
    logger.info(f"负荷集成完成，输出文件: {snakemake.output.integrated_load}")

if __name__ == "__main__":
    main() 