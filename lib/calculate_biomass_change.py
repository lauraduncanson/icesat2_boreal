#!/usr/bin/env python3
"""
Calculate age×trend-specific biomass change with Monte Carlo uncertainty
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import argparse
from itertools import product

def get_age_trend_combinations(df, year_prefix, 
                               age_classes = ['non-forest', 'young', 'maturing','mature', 'old-growth'],
                                trend_classes = ['strong_decline', 'moderate_decline', 'stable', 'moderate_increase', 'strong_increase']
                                  ):

    """
    Identify all age×trend combinations present in the data
    """
    
    # Find actual combinations in the data
    actual_combinations = []
    
    for age_class in age_classes:
        for trend_class in trend_classes:
            # Check if this combination exists in the data
            mean_col = f"{year_prefix}_{age_class}_trend_{trend_class}_median_biomass_pg"
            std_col = f"{year_prefix}_{age_class}_trend_{trend_class}_std_biomass_pg"
            
            if mean_col in df.columns and std_col in df.columns:
                actual_combinations.append((age_class, trend_class))
    
    print(f"📊 Found {len(actual_combinations)} age×trend combinations in data")
    return actual_combinations

def calculate_age_trend_biomass_change_mc(df, year1_prefix='agb_2020', year2_prefix='agb_2025',
                                        n_simulations=10000, confidence_level=0.95):
    """
    Calculate biomass change for each age×trend combination with Monte Carlo uncertainty
    """
    
    print(f"🔬 Calculating age×trend-specific biomass change: {year1_prefix} → {year2_prefix}")
    print(f"🎲 Using {n_simulations} Monte Carlo simulations per combination")
    
    # Get all age×trend combinations
    combinations = get_age_trend_combinations(df, year1_prefix)
    
    if not combinations:
        print("❌ No age×trend combinations found in data")
        return df
    
    results_dict = {}
    
    for idx, row in df.iterrows():
        
        if idx % 100 == 0:
            print(f"📍 Processing landscape {idx+1}/{len(df)}")
        
        landscape_results = {}
        
        for age_class, trend_class in combinations:
            
            # Column names for this age×trend combination
            y1_mean_col = f"{year1_prefix}_{age_class}_trend_{trend_class}_median_biomass_pg"
            y1_std_col = f"{year1_prefix}_{age_class}_trend_{trend_class}_std_biomass_pg"
            y2_mean_col = f"{year2_prefix}_{age_class}_trend_{trend_class}_median_biomass_pg"
            y2_std_col = f"{year2_prefix}_{age_class}_trend_{trend_class}_std_biomass_pg"
            
            try:
                # Extract values for this landscape and combination
                y1_mean = row.get(y1_mean_col, np.nan)
                y1_std = row.get(y1_std_col, np.nan)
                y2_mean = row.get(y2_mean_col, np.nan)
                y2_std = row.get(y2_std_col, np.nan)
                
                # Handle cases where std is NaN but mean is valid (zero pixels case)
                # If mean is 0 and std is NaN, treat as zero biomass with no uncertainty
                if pd.isna(y1_std) and pd.notna(y1_mean) and y1_mean == 0:
                    y1_std = 0.0
                if pd.isna(y2_std) and pd.notna(y2_mean) and y2_mean == 0:
                    y2_std = 0.0
                
                # Skip if any values are missing or invalid (but now after NaN handling)
                if pd.isna([y1_mean, y1_std, y2_mean, y2_std]).any() or any(v < 0 for v in [y1_std, y2_std] if pd.notna(v)):
                    combo_stats = {
                        'y1_mc_mean': np.nan, 'y1_mc_std': np.nan,
                        'y2_mc_mean': np.nan, 'y2_mc_std': np.nan,
                        'change_mc_mean': np.nan, 'change_mc_std': np.nan,
                        'change_mc_median': np.nan, 'change_lower_ci': np.nan,
                        'change_upper_ci': np.nan, 'change_percent': np.nan,
                        'prob_increase': np.nan, 'prob_decrease': np.nan,
                        'prob_significant_change': np.nan
                    }
                else:
                    # Monte Carlo simulation for this age×trend combination
                    # Handle zero std case (no uncertainty)
                    if y1_std == 0:
                        y1_simulations = np.full(n_simulations, y1_mean)
                    else:
                        y1_simulations = np.maximum(0, np.random.normal(y1_mean, y1_std, n_simulations))
                    
                    if y2_std == 0:
                        y2_simulations = np.full(n_simulations, y2_mean)
                    else:
                        y2_simulations = np.maximum(0, np.random.normal(y2_mean, y2_std, n_simulations))
                    
                    change_simulations = y2_simulations - y1_simulations
                    
                    # Calculate statistics
                    alpha = 1 - confidence_level
                    
                    combo_stats = {
                        # Year 1 Monte Carlo stats
                        'y1_mc_mean': np.mean(y1_simulations),
                        'y1_mc_std': np.std(y1_simulations),
                        
                        # Year 2 Monte Carlo stats
                        'y2_mc_mean': np.mean(y2_simulations),
                        'y2_mc_std': np.std(y2_simulations),
                        
                        # Change Monte Carlo stats
                        'change_mc_mean': np.mean(change_simulations),
                        'change_mc_std': np.std(change_simulations),
                        'change_mc_median': np.median(change_simulations),
                        'change_lower_ci': np.percentile(change_simulations, 100 * alpha/2),
                        'change_upper_ci': np.percentile(change_simulations, 100 * (1 - alpha/2)),
                        
                        # Relative change
                        'change_percent': (np.mean(change_simulations) / np.mean(y1_simulations)) * 100 if np.mean(y1_simulations) > 0 else np.nan,
                        
                        # Probabilities
                        'prob_increase': np.sum(change_simulations > 0) / n_simulations,
                        'prob_decrease': np.sum(change_simulations < 0) / n_simulations,
                        
                        # Significant change (95% CI doesn't include zero)
                        'prob_significant_change': 1.0 if (np.percentile(change_simulations, 2.5) > 0 or 
                                                          np.percentile(change_simulations, 97.5) < 0) else 0.0
                    }
                
                # Store results with appropriate column names
                combo_prefix = f"{age_class}_trend_{trend_class}"
                for stat_name, stat_value in combo_stats.items():
                    col_name = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_{stat_name}_pg"
                    landscape_results[col_name] = stat_value
                    
            except Exception as e:
                print(f"⚠️  Error processing {age_class}×{trend_class} for landscape {idx}: {e}")
                continue
        
        # Store all results for this landscape
        for col_name, value in landscape_results.items():
            if col_name not in results_dict:
                results_dict[col_name] = [np.nan] * len(df)
            results_dict[col_name][idx] = value
    
    # Add results to dataframe
    for col_name, values in results_dict.items():
        df[col_name] = values
    
    print(f"✅ Added {len(results_dict)} age×trend-specific change columns")
    return df

def create_age_trend_summary_table(df, year1_prefix='agb_2020', year2_prefix='agb_2025',
                                  age_order = ['non-forest', 'young', 'maturing','mature', 'old-growth'],
                                   trend_order = ['strong_decline', 'moderate_decline', 'stable', 'moderate_increase', 'strong_increase']
                                  ):

    """
    Create a summary table showing aggregate statistics across all landscapes
    """
    
    print("📊 Creating age×trend summary table...")
    
    # Find all age×trend combination columns
    change_cols = [col for col in df.columns if f"{year1_prefix}_to_{year2_prefix}" in col and "change_mc_mean" in col]
    
    summary_rows = []
    
    for col in change_cols:
        # Extract age and trend class from column name
        parts = col.replace(f"{year1_prefix}_to_{year2_prefix}_", "").replace("_change_mc_mean_pg", "")
        
        if "_trend_" in parts:
            age_class = parts.split("_trend_")[0]
            trend_class = parts.split("_trend_")[1]
            
            # Calculate summary statistics across all landscapes - KEEP NaN values
            all_data = df[col]  # Don't use dropna() here
            valid_data = all_data.dropna()  # Only for counting valid landscapes
            
            if len(valid_data) > 0:
                # Get related columns
                std_col = col.replace("change_mc_mean", "change_mc_std")
                percent_col = col.replace("change_mc_mean_pg", "change_percent_pg")
                prob_increase_col = col.replace("change_mc_mean", "prob_increase")
                
                summary_row = {
                    'age_class': age_class,
                    'trend_class': trend_class,
                    'n_landscapes_total': len(all_data),  # Total landscapes
                    'n_landscapes_with_data': len(valid_data),  # Landscapes with this combination
                    'n_landscapes_no_pixels': len(all_data) - len(valid_data),  # Landscapes with zero pixels
                    'mean_change_pg': valid_data.mean(),
                    'median_change_pg': valid_data.median(),
                    'std_change_pg': valid_data.std(),
                    'total_change_pg': valid_data.sum(),
                    'landscapes_with_increase': (valid_data > 0).sum(),
                    'landscapes_with_decrease': (valid_data < 0).sum(),
                    'percent_increasing': (valid_data > 0).mean() * 100 if len(valid_data) > 0 else np.nan
                }
                
                # Add uncertainty statistics if available
                if std_col in df.columns:
                    uncertainty_data = df[std_col].dropna()
                    if len(uncertainty_data) > 0:
                        summary_row['mean_uncertainty_pg'] = uncertainty_data.mean()
                        summary_row['median_uncertainty_pg'] = uncertainty_data.median()
                
                summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    if len(summary_df) > 0:
        # Sort by age class and trend class
        
        summary_df['age_class'] = pd.Categorical(summary_df['age_class'], categories=age_order, ordered=True)
        summary_df['trend_class'] = pd.Categorical(summary_df['trend_class'], categories=trend_order, ordered=True)
        summary_df = summary_df.sort_values(['age_class', 'trend_class'])
    
    return summary_df

def create_long_format_summary_table(df, year1_prefix='agb_2020', year2_prefix='agb_2025', 
                                   polygon_id_col='polygon_id',
                                    age_order = ['non-forest', 'young', 'maturing','mature', 'old-growth'],
                                   trend_order = ['strong_decline', 'moderate_decline', 'stable', 'moderate_increase', 'strong_increase']
                                  ):
    """
    Create a long-format summary table with age_class, trend_class, and value_type columns
    """
    
    print("📊 Creating long-format biomass change table...")
    
    # Find all age×trend combination columns
    change_cols = [col for col in df.columns if f"{year1_prefix}_to_{year2_prefix}" in col and "change_mc_mean" in col]
    
    # Extract unique age×trend combinations from the column names
    combinations = []
    for col in change_cols:
        parts = col.replace(f"{year1_prefix}_to_{year2_prefix}_", "").replace("_change_mc_mean_pg", "")
        if "_trend_" in parts:
            age_class = parts.split("_trend_")[0]
            trend_class = parts.split("_trend_")[1]
            combinations.append((age_class, trend_class))
    
    print(f"Found {len(combinations)} age×trend combinations")
    
    long_rows = []
    
    # Process each polygon
    for idx, row in df.iterrows():
        
        # Get polygon ID
        if polygon_id_col in df.columns:
            polygon_id = row[polygon_id_col]
        else:
            polygon_id = idx
        
        # Process each age×trend combination
        for age_class, trend_class in combinations:
            
            # Column prefixes for this combination
            combo_prefix = f"{age_class}_trend_{trend_class}"
            
            # Column names for the three value types
            year1_col = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_y1_mc_mean_pg"
            year2_col = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_y2_mc_mean_pg"
            change_col = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_change_mc_mean_pg"
            
            # Standard deviation columns
            year1_std_col = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_y1_mc_std_pg"
            year2_std_col = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_y2_mc_std_pg"
            change_std_col = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_change_mc_std_pg"
            
            # Other useful columns
            change_percent_col = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_change_percent_pg"
            prob_increase_col = f"{year1_prefix}_to_{year2_prefix}_{combo_prefix}_prob_increase_pg"
            
            # Create three rows for this polygon×age×trend combination
            
            # Row 1: Year 1 (e.g., agb_2020)
            long_rows.append({
                'polygon_id': polygon_id,
                'age_class': age_class,
                'trend_class': trend_class,
                'value_type': year1_prefix,
                'biomass_mean_pg': row.get(year1_col, np.nan),
                'biomass_std_pg': row.get(year1_std_col, np.nan),
                'change_percent': np.nan,  # Not applicable for individual years
                'prob_increase': np.nan,   # Not applicable for individual years
            })
            
            # Row 2: Year 2 (e.g., agb_2025)
            long_rows.append({
                'polygon_id': polygon_id,
                'age_class': age_class,
                'trend_class': trend_class,
                'value_type': year2_prefix,
                'biomass_mean_pg': row.get(year2_col, np.nan),
                'biomass_std_pg': row.get(year2_std_col, np.nan),
                'change_percent': np.nan,  # Not applicable for individual years
                'prob_increase': np.nan,   # Not applicable for individual years
            })
            
            # Row 3: Change (e.g., agb_2020_to_agb_2025)
            long_rows.append({
                'polygon_id': polygon_id,
                'age_class': age_class,
                'trend_class': trend_class,
                'value_type': f"{year1_prefix}_to_{year2_prefix}",
                'biomass_mean_pg': row.get(change_col, np.nan),
                'biomass_std_pg': row.get(change_std_col, np.nan),
                'change_percent': row.get(change_percent_col, np.nan),
                'prob_increase': row.get(prob_increase_col, np.nan),
            })
    
    # Create DataFrame
    long_df = pd.DataFrame(long_rows)
    
    # Set proper categorical ordering
    long_df['age_class'] = pd.Categorical(long_df['age_class'], categories=age_order, ordered=True)
    long_df['trend_class'] = pd.Categorical(long_df['trend_class'], categories=trend_order, ordered=True)
    long_df['value_type'] = pd.Categorical(long_df['value_type'], 
                                          categories=[year1_prefix, year2_prefix, f"{year1_prefix}_to_{year2_prefix}"], 
                                          ordered=True)
    
    # Sort the data
    long_df = long_df.sort_values(['polygon_id', 'age_class', 'trend_class', 'value_type'])
    
    print(f"✅ Created long-format table with {len(long_df)} rows")
    print(f"📊 Covers {long_df['polygon_id'].nunique()} polygons")
    
    return long_df

def create_simplified_long_format_table(df, year1_prefix='agb_2020', year2_prefix='agb_2025', 
                                      polygon_id_col='polygon_id'):
    """
    Create a simplified long-format summary table ignoring age and trend classes
    Returns one row per polygon per value_type (year1, year2, change)
    """
    
    print("📊 Creating simplified long-format biomass table...")
    
    # Look for total biomass columns
    year1_mean_col = f"{year1_prefix}_total_median_biomass_pg"
    year1_std_col = f"{year1_prefix}_total_std_biomass_pg"
    year2_mean_col = f"{year2_prefix}_total_median_biomass_pg"
    year2_std_col = f"{year2_prefix}_total_std_biomass_pg"
    
    # Check if columns exist
    required_cols = [year1_mean_col, year1_std_col, year2_mean_col, year2_std_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ Missing required columns for simplified table: {missing_cols}")
        print("🔍 Calculating total biomass values from Monte Carlo simulations...")
        
        # Calculate total values using Monte Carlo for each polygon
        simplified_rows = []
        
        for idx, row in df.iterrows():
            # Get polygon ID
            if polygon_id_col in df.columns:
                polygon_id = row[polygon_id_col]
            else:
                polygon_id = idx
            
            # Initialize with nan values
            poly_data = {
                'polygon_id': polygon_id,
                f'{year1_prefix}_mean_pg': np.nan,
                f'{year1_prefix}_std_pg': np.nan,
                f'{year2_prefix}_mean_pg': np.nan,
                f'{year2_prefix}_std_pg': np.nan,
                f'{year1_prefix}_to_{year2_prefix}_mean_pg': np.nan,
                f'{year1_prefix}_to_{year2_prefix}_std_pg': np.nan,
                f'{year1_prefix}_to_{year2_prefix}_percent': np.nan,
                f'{year1_prefix}_to_{year2_prefix}_prob_increase': np.nan
            }
            
            # Collect all Monte Carlo simulation values for year 1
            y1_values = []
            y2_values = []
            change_values = []
            
            # Find all Monte Carlo simulation columns for this polygon
            y1_cols = [c for c in df.columns if f"{year1_prefix}_to_{year2_prefix}" in c and "y1_mc_mean" in c]
            y2_cols = [c for c in df.columns if f"{year1_prefix}_to_{year2_prefix}" in c and "y2_mc_mean" in c]
            change_cols = [c for c in df.columns if f"{year1_prefix}_to_{year2_prefix}" in c and "change_mc_mean" in c]
            
            # Aggregate mean values
            if y1_cols:
                y1_means = [row[col] for col in y1_cols if pd.notna(row[col])]
                if y1_means:
                    poly_data[f'{year1_prefix}_mean_pg'] = sum(y1_means)
            
            if y2_cols:
                y2_means = [row[col] for col in y2_cols if pd.notna(row[col])]
                if y2_means:
                    poly_data[f'{year2_prefix}_mean_pg'] = sum(y2_means)
            
            if change_cols:
                change_means = [row[col] for col in change_cols if pd.notna(row[col])]
                if change_means:
                    poly_data[f'{year1_prefix}_to_{year2_prefix}_mean_pg'] = sum(change_means)
            
            # Aggregate std values - use quadrature sum (sqrt of sum of squares)
            y1_std_cols = [c.replace("y1_mc_mean", "y1_mc_std") for c in y1_cols]
            y2_std_cols = [c.replace("y2_mc_mean", "y2_mc_std") for c in y2_cols]
            change_std_cols = [c.replace("change_mc_mean", "change_mc_std") for c in change_cols]
            
            if y1_std_cols:
                y1_stds = [row[col]**2 for col in y1_std_cols if pd.notna(row[col])]
                if y1_stds:
                    poly_data[f'{year1_prefix}_std_pg'] = np.sqrt(sum(y1_stds))
            
            if y2_std_cols:
                y2_stds = [row[col]**2 for col in y2_std_cols if pd.notna(row[col])]
                if y2_stds:
                    poly_data[f'{year2_prefix}_std_pg'] = np.sqrt(sum(y2_stds))
            
            if change_std_cols:
                change_stds = [row[col]**2 for col in change_std_cols if pd.notna(row[col])]
                if change_stds:
                    poly_data[f'{year1_prefix}_to_{year2_prefix}_std_pg'] = np.sqrt(sum(change_stds))
            
            # Calculate change percent and probability
            if pd.notna(poly_data[f'{year1_prefix}_mean_pg']) and pd.notna(poly_data[f'{year1_prefix}_to_{year2_prefix}_mean_pg']):
                if poly_data[f'{year1_prefix}_mean_pg'] > 0:
                    poly_data[f'{year1_prefix}_to_{year2_prefix}_percent'] = (
                        poly_data[f'{year1_prefix}_to_{year2_prefix}_mean_pg'] / 
                        poly_data[f'{year1_prefix}_mean_pg'] * 100
                    )
                
                # Calculate probability of increase (very approximate)
                if pd.notna(poly_data[f'{year1_prefix}_to_{year2_prefix}_std_pg']):
                    change_mean = poly_data[f'{year1_prefix}_to_{year2_prefix}_mean_pg']
                    change_std = poly_data[f'{year1_prefix}_to_{year2_prefix}_std_pg']
                    
                    if change_std > 0:
                        z_score = change_mean / change_std
                        poly_data[f'{year1_prefix}_to_{year2_prefix}_prob_increase'] = 1 - scipy.stats.norm.cdf(-z_score)
            
            simplified_rows.append(poly_data)
        
        # Create DataFrame
        simplified_df = pd.DataFrame(simplified_rows)
        
    else:
        # Use existing total columns
        simplified_rows = []
        
        for idx, row in df.iterrows():
            # Get polygon ID
            if polygon_id_col in df.columns:
                polygon_id = row[polygon_id_col]
            else:
                polygon_id = idx
            
            # Get existing total values
            year1_mean = row.get(year1_mean_col, np.nan)
            year1_std = row.get(year1_std_col, np.nan)
            year2_mean = row.get(year2_mean_col, np.nan)
            year2_std = row.get(year2_std_col, np.nan)
            
            # Calculate change values
            change_mean = year2_mean - year1_mean if pd.notna(year1_mean) and pd.notna(year2_mean) else np.nan
            
            # Calculate change std using error propagation
            change_std = np.nan
            if pd.notna(year1_std) and pd.notna(year2_std):
                change_std = np.sqrt(year1_std**2 + year2_std**2)
            
            # Calculate change percent
            change_percent = np.nan
            if pd.notna(change_mean) and pd.notna(year1_mean) and year1_mean > 0:
                change_percent = (change_mean / year1_mean) * 100
            
            # Calculate probability of increase
            prob_increase = np.nan
            if pd.notna(change_mean) and pd.notna(change_std) and change_std > 0:
                z_score = change_mean / change_std
                prob_increase = 1 - scipy.stats.norm.cdf(-z_score)
            
            simplified_rows.append({
                'polygon_id': polygon_id,
                f'{year1_prefix}_mean_pg': year1_mean,
                f'{year1_prefix}_std_pg': year1_std,
                f'{year2_prefix}_mean_pg': year2_mean,
                f'{year2_prefix}_std_pg': year2_std,
                f'{year1_prefix}_to_{year2_prefix}_mean_pg': change_mean,
                f'{year1_prefix}_to_{year2_prefix}_std_pg': change_std,
                f'{year1_prefix}_to_{year2_prefix}_percent': change_percent,
                f'{year1_prefix}_to_{year2_prefix}_prob_increase': prob_increase
            })
        
        # Create DataFrame
        simplified_df = pd.DataFrame(simplified_rows)
    
    # Convert to long format
    long_rows = []
    
    for idx, row in simplified_df.iterrows():
        polygon_id = row['polygon_id']
        
        # Add row for year 1
        long_rows.append({
            'polygon_id': polygon_id,
            'value_type': year1_prefix,
            'biomass_mean_pg': row[f'{year1_prefix}_mean_pg'],
            'biomass_std_pg': row[f'{year1_prefix}_std_pg'],
            'change_percent': np.nan,
            'prob_increase': np.nan,
        })
        
        # Add row for year 2
        long_rows.append({
            'polygon_id': polygon_id,
            'value_type': year2_prefix,
            'biomass_mean_pg': row[f'{year2_prefix}_mean_pg'],
            'biomass_std_pg': row[f'{year2_prefix}_std_pg'],
            'change_percent': np.nan,
            'prob_increase': np.nan,
        })
        
        # Add row for change
        long_rows.append({
            'polygon_id': polygon_id,
            'value_type': f'{year1_prefix}_to_{year2_prefix}',
            'biomass_mean_pg': row[f'{year1_prefix}_to_{year2_prefix}_mean_pg'],
            'biomass_std_pg': row[f'{year1_prefix}_to_{year2_prefix}_std_pg'],
            'change_percent': row[f'{year1_prefix}_to_{year2_prefix}_percent'],
            'prob_increase': row[f'{year1_prefix}_to_{year2_prefix}_prob_increase'],
        })
    
    # Create long format DataFrame
    long_df = pd.DataFrame(long_rows)
    
    # Set categorical ordering
    long_df['value_type'] = pd.Categorical(long_df['value_type'], 
                                          categories=[year1_prefix, year2_prefix, f"{year1_prefix}_to_{year2_prefix}"], 
                                          ordered=True)
    
    # Sort the data
    long_df = long_df.sort_values(['polygon_id', 'value_type'])
    
    print(f"✅ Created simplified long-format table with {len(long_df)} rows")
    print(f"📊 Covers {long_df['polygon_id'].nunique()} polygons")
    
    return long_df

def main():
    parser = argparse.ArgumentParser(
        description="Calculate age×trend-specific biomass change with Monte Carlo uncertainty"
    )
    
    parser.add_argument('--input', required=True, help='Input file with age×trend biomass data')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--summary-output', help='Summary table output path')
    parser.add_argument('--year1-prefix', default='biomass_2020', help='Year 1 biomass prefix')
    parser.add_argument('--year2-prefix', default='biomass_2025', help='Year 2 biomass prefix')
    parser.add_argument('--simulations', type=int, default=10000, help='Number of Monte Carlo simulations')
    parser.add_argument('--confidence-level', type=float, default=0.95, help='Confidence level')
    
    args = parser.parse_args()
    
    # Read input data
    print(f"📥 Reading input: {args.input}")
    
    if args.input.endswith('.gpkg'):
        import geopandas as gpd
        df = gpd.read_file(args.input)
        is_spatial = True
    else:
        df = pd.read_csv(args.input)
        is_spatial = False
    
    print(f"📊 Loaded {len(df)} landscapes")
    
    # Calculate age×trend-specific biomass change with uncertainty
    df_with_change = calculate_age_trend_biomass_change_mc(
        df, args.year1_prefix, args.year2_prefix, args.simulations, args.confidence_level
    )
    
    # Save main results
    print(f"💾 Saving results: {args.output}")
    
    if is_spatial and args.output.endswith('.gpkg'):
        df_with_change.to_file(args.output, driver='GPKG')
    else:
        if 'geometry' in df_with_change.columns:
            df_with_change = df_with_change.drop('geometry', axis=1)
        df_with_change.to_csv(args.output, index=False)
    
    # Create and save summary table
    if args.summary_output:
        summary_df = create_age_trend_summary_table(df_with_change, args.year1_prefix, args.year2_prefix)
        summary_df.to_csv(args.summary_output, index=False)
        print(f"📄 Summary table saved: {args.summary_output}")

        # NEW: Create and save long-format table
        long_format_output = args.output.replace('.csv', '_long_format.csv').replace('.gpkg', '_long_format.csv')
        long_df = create_long_format_summary_table(df_with_change, args.year1_prefix, args.year2_prefix)
        long_df.to_csv(long_format_output, index=False)
        print(f"📄 Long-format table saved: {long_format_output}")

        # NEW: Create and save simplified long-format table
        simplified_long_output = args.output.replace('.csv', '_simplified_long_format.csv').replace('.gpkg', '_simplified_long_format.csv')
        simplified_long_df = create_simplified_long_format_table(df_with_change, args.year1_prefix, args.year2_prefix)
        simplified_long_df.to_csv(simplified_long_output, index=False)
        print(f"📄 Simplified long-format table saved: {simplified_long_output}")
        
        # Print key findings
        if len(summary_df) > 0:
            print(f"\n📈 Key Findings ({args.year1_prefix} → {args.year2_prefix}):")
            print("="*60)
            
            for _, row in summary_df.iterrows():
                change_pg = row['mean_change_pg']
                pct_sign = "+" if change_pg > 0 else ""
                print(f"{row['age_class']} × {row['trend_class']}:")
                print(f"  Mean change: {pct_sign}{change_pg:.6f} Pg ({row['percent_increasing']:.1f}% of landscapes increasing)")
                
                if pd.notna(row.get('mean_uncertainty_pg')):
                    print(f"  Mean uncertainty: ±{row['mean_uncertainty_pg']:.6f} Pg")
    
    print("\n✅ Age×trend biomass change analysis complete!")

if __name__ == '__main__':
    main()