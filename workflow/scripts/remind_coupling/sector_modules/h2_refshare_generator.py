"""Hydrogen sector reference data generator

Generates reference data for the hydrogen sector using population-based decomposition.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SimplePopulationModel:
    """Simple population-based hydrogen demand decomposition model"""

    def __init__(self):
        """Initialize the simple population model"""
        pass

    def calculate_h2_shares_from_population(
        self, population_data: pd.DataFrame, years: list[int]
    ) -> dict:
        """Calculate hydrogen demand shares directly from population data.
        
        Args:
            population_data: Future population projections 
                             (rows = provinces, columns = years)
            years: List of target years
            
        Returns:
            dict: Dictionary with year as key and {province: H2 share} as value
        """
        predictions = {}

        for year in years:
            # 🔹 Step 1: 取某年的省份人口数据
            # 这里默认 population_data.columns 是年份
            if year in population_data.columns:
                year_population = population_data[year]
            else:
                # 如果目标年份不存在，就用最后一个可用年份替代
                year_population = population_data.iloc[:, -1]

            # 🔹 Step 2: 计算氢气需求比例
            # ⚠️ 未来如果你想用更复杂的算法（如人均GDP、产业结构），就在这里改！
            # 现在是简单的 "人口占比 = H2需求占比"
            h2_shares = year_population / year_population.sum()

            # 🔹 Step 3: 存储为 dict，方便后续转 DataFrame
            predictions[year] = h2_shares.to_dict()

        logger.info(
            f"Generated H2 demand predictions for {len(years)} years based on population"
        )
        return predictions


def generate_reference(
    years: list[int], input_files: dict[str, str], output_dir: str, config: dict = None
):
    """Generate hydrogen sector reference data.
    
    Args:
        years: List of target years for projections
        input_files: Dictionary mapping data types to file paths
        output_dir: Directory to save generated reference files
        config: Configuration dictionary for H2 sector
    """
    logger.info("Starting hydrogen sector reference data generation")

    # 🔹 Step A: 读取输入人口数据
    try:
        # 假设 Excel 文件格式： index=province, columns=year
        ssp2_pop = pd.read_excel(input_files["ssp2_pop"], index_col=0)
        logger.info(f"Loaded future population data: {ssp2_pop.shape}")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return

    # 🔹 Step B: 初始化模型
    model = SimplePopulationModel()

    # 🔹 Step C: 生成预测
    try:
        predictions = model.calculate_h2_shares_from_population(ssp2_pop, years)

        # 🔹 Step D: 转成 DataFrame，行=省份，列=年份
        shares_df = pd.DataFrame(predictions)

        # 🔹 Step E: 保存结果
        output_file = f"{output_dir}/h2_demand_shares.csv"
        shares_df.to_csv(output_file)

        logger.info(f"H2 reference data saved to {output_file}")
        logger.info(
            f"Generated H2 demand shares for {len(years)} years and {len(shares_df)} provinces"
        )

        # 🔹 Step F: 打印 debug 统计，快速检查
        logger.info("H2 demand shares summary:")
        for year in years:
            year_data = shares_df[year]
            logger.info(
                f"  {year}: Total={year_data.sum():.4f}, "
                f"Max={year_data.max():.4f} ({year_data.idxmax()}), "
                f"Min={year_data.min():.4f} ({year_data.idxmin()})"
            )

        logger.info("Hydrogen sector reference data generation completed")

    except Exception as e:
        logger.error(f"Failed to generate H2 predictions: {e}")
        return
