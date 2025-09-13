#!/usr/bin/env python3
"""Simple test script for H2 reference data generator"""

import pandas as pd
import numpy as np
from workflow.scripts.remind_coupling.sector_modules.h2_refshare_generator import SimplePopulationModel

def create_test_data():
    """Create test population data"""
    # Create test population data for a few provinces
    provinces = ['Beijing', 'Shanghai', 'Guangdong', 'Sichuan', 'Xinjiang']
    
    # Future population data (SSP2 projections, millions)
    future_pop_data = {
        'Beijing': [24.0, 24.5, 25.0],
        'Shanghai': [26.5, 27.0, 27.5],
        'Guangdong': [120.0, 122.0, 124.0],
        'Sichuan': [87.0, 87.5, 88.0],
        'Xinjiang': [29.0, 29.5, 30.0]
    }
    
    future_pop_df = pd.DataFrame(future_pop_data, index=[2030, 2040, 2050])
    
    return future_pop_df

def test_h2_simple():
    """Test the simplified H2 reference data generator"""
    print("Testing simplified H2 reference data generator (Population-based)...")
    
    # Create test data
    future_pop = create_test_data()
    print(f"Created test data: {future_pop.shape}")
    print(f"Population data:\n{future_pop}")
    
    # Initialize model
    model = SimplePopulationModel()
    
    # Generate predictions
    years = [2030, 2040, 2050]
    predictions = model.calculate_h2_shares_from_population(future_pop, years)
    
    print(f"✅ Generated predictions for {len(years)} years")
    
    # Display results
    print("\nH2 demand shares by province and year:")
    for year in years:
        print(f"\n{year}:")
        year_data = predictions[year]
        total = sum(year_data.values())
        for province, share in sorted(year_data.items(), key=lambda x: x[1], reverse=True):
            percentage = (share / total) * 100
            print(f"  {province}: {share:.4f} ({percentage:.1f}%)")
    
    # Save to CSV for inspection
    shares_df = pd.DataFrame(predictions)
    shares_df.to_csv("test_h2_simple_shares.csv")
    print(f"\n✅ Results saved to test_h2_simple_shares.csv")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_h2_simple()
