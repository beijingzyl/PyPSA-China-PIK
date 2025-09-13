"""Hydrogen sector reference data generator

Generates reference data for the hydrogen sector using per-capita GDP-based decomposition.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SimplePopulationModel:
    """Simple population-based hydrogen demand decomposition model"""

    def __init__(self):
        """Initialize the simple population model"""
        pass

    def calculate_h2_shares_from_population(self, population_data: pd.DataFrame, years: list[int]) -> dict:
        """Calculate hydrogen demand shares directly from population data.
        
        Args:
            population_data: Future population projections by province
            years: List of target years
            
        Returns:
            dict: Dictionary with year as key and H2 shares as value
        """
        predictions = {}
        
        for year in years:
            # Get population data for the specific year
            if year in population_data.index:
                year_population = population_data.loc[year]
            else:
                # Use the latest available year
                year_population = population_data.iloc[-1]
            
            # Calculate H2 shares based on population (proportional to population)
            h2_shares = year_population / year_population.sum()
            
            predictions[year] = h2_shares.to_dict()
            
        logger.info(f"Generated H2 demand predictions for {len(years)} years based on population")
        return predictions


def generate_reference(years: list[int], input_files: dict[str, str], 
                      output_dir: str, config: dict = None):
    """Generate hydrogen sector reference data.
    
    Args:
        years: List of target years for projections
        input_files: Dictionary mapping data types to file paths
        output_dir: Directory to save generated reference files
        config: Configuration dictionary for H2 sector
    """
    logger.info("Starting hydrogen sector reference data generation")
    
    # Load input data
    try:
        # Load only future population data
        ssp2_pop = pd.read_csv(input_files["ssp2_pop"], index_col=0)
        
        logger.info(f"Loaded future population data: {ssp2_pop.shape}")
        
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return
    
    # Initialize simple model
    model = SimplePopulationModel()
    
    # Generate predictions for future years based on population
    try:
        predictions = model.calculate_h2_shares_from_population(ssp2_pop, years)
        
        # Convert to DataFrame
        shares_df = pd.DataFrame(predictions)
        
        # Save to CSV
        output_file = f"{output_dir}/h2_demand_shares.csv"
        shares_df.to_csv(output_file)
        
        logger.info(f"H2 reference data saved to {output_file}")
        logger.info(f"Generated H2 demand shares for {len(years)} years and {len(shares_df)} provinces")
        
        # Print summary statistics
        logger.info("H2 demand shares summary:")
        for year in years:
            year_data = shares_df[year]
            logger.info(f"  {year}: Total={year_data.sum():.4f}, "
                       f"Max={year_data.max():.4f} ({year_data.idxmax()}), "
                       f"Min={year_data.min():.4f} ({year_data.idxmin()})")
        
    except Exception as e:
        logger.error(f"Failed to generate H2 predictions: {e}")
        return
    
    logger.info("Hydrogen sector reference data generation completed")
