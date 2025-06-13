import numpy as np
import rasterio
from rasterio.plot import show, show_hist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats


# ESA Worldcover
names_worldcover = [ 'Trees', 'Shrubland', 'Grassland','Cropland',\
                    'Built-up',
                    'Barren/sparse','Snow and ice','Open water','Herbaceous\nwetland',
                    #'Mangroves',
                    'Moss and lichen']
cols_worldcover = [ "#006400","#ffbb22","#ffff4c","#f096ff",\
                   "#fa0000",\
                   "#b4b4b4","#f0f0f0","#0064c8","#0096a0",
                   #"#00cf75",
                   "#fae6a0"]

# The land cover values in which you want to summarize AGB
values_worldcover = [10,20,30,40,\
                     50,\
                     60,70,80,90,\
                     #95,\
                     100] 

cmap_worldcover = ListedColormap(cols_worldcover)
norm_worldcover = BoundaryNorm(values_worldcover, len(cols_worldcover))
dict_worldcover_colors = dict(zip(names_worldcover, cols_worldcover))

dict_worldcover_values = dict(zip(names_worldcover, values_worldcover))
dict_worldcover_classes = dict(zip(values_worldcover, names_worldcover))

# # TP TCC change rate
# names_tcctrend = ['decrease', 'stable', 'increase']
# colors_tcctrend = ['#a6611a','#f5f5f5','#018571']
# values_tcctrend = [-0.5,0,0.5]

# cmap_tcctrend = ListedColormap(colors_tcctrend)
# norm_tcctrend = BoundaryNorm(values_tcctrend, len(colors_tcctrend))

# dict_tcctrend_colors = dict(zip(names_tcctrend, colors_tcctrend))
# dict_tcctrend_colors

# dict_tcctrend_values = dict(zip(names_tcctrend, values_tcctrend))
# dict_tcctrend_values

# Massey et al deciduous fraction
names_decidpred = ['conifer', 'mixed', 'deciduous']
colors_decidpred = ['#5e3c99','#fdb863','#ffffbf']
values_decidpred = [0,1,2] #['<25','25-75','>75']

cmap_decidpred = ListedColormap(colors_decidpred)
norm_decidpred = BoundaryNorm(values_decidpred, len(colors_decidpred))

dict_decidpred_colors = dict(zip(names_decidpred, colors_decidpred))

dict_decidpred_values = dict(zip(names_decidpred, values_decidpred))

# Forest/Stand Age
names_standage  = ['non-forest', 're-growth forest','young forest', 'mature forest','old-growth forest']
colors_standage = ['gray','#e41a1c','#d9ef8b', '#5aae61', '#1b7837']
values_standage = [0,1,2,3,4]

cmap_standage = ListedColormap(colors_standage)
norm_standage = BoundaryNorm(values_standage, len(colors_standage))

dict_standage_colors = dict(zip(names_standage, colors_standage))

dict_standage_values = dict(zip(names_standage, values_standage))

def monte_carlo_uncertainty(arr, n_simulations=50, seed=None, confidence=0.95):
    """
    Perform Monte Carlo uncertainty analysis on a 2-band raster.
    
    Parameters:
    -----------
    raster_path : str
        Path to raster file with band 1 as mean and band 2 as standard deviation
    n_simulations : int
        Number of Monte Carlo simulations to run
    seed : int or None
        Random seed for reproducibility
    
    Returns:
    --------
    mean_of_means : numpy.ndarray
        Mean of the simulated means for each pixel
    std_of_means : numpy.ndarray
        Standard deviation of the simulated means for each pixel
    simulations : numpy.ndarray
        All simulations (n_simulations, height, width)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # # Read raster data
    # with rasterio.open(raster_path) as src:
    #     mean_band = src.read(1)
    #     std_band = src.read(2)
    #     profile = src.profile
    mean_band = arr[0,:,:]
    std_band = arr[1,:,:]
        
    # Initialize array to store simulations
    simulations = np.zeros((n_simulations, mean_band.shape[0], mean_band.shape[1]))
    
    # Run Monte Carlo simulations
    for i in range(n_simulations):
        # For each pixel, sample from normal distribution with pixel's mean and std
        simulations[i] = np.random.normal(mean_band, std_band)
    
    # Calculate statistics across simulations
    mean_of_means = np.mean(simulations, axis=0)
    std_of_means = np.std(simulations, axis=0)
    #pcts_of_means = np.nanpercentile(simulations, [5,95], axis=0

    # Calculate standard error of the mean
    sem = std_of_means / np.sqrt(n_simulations)
    
    # Calculate 95% confidence intervals for the mean
    # t-value for 95% CI with n-1 degrees of freedom
    t_value = stats.t.ppf(0.975, n_simulations - 1)
    
    ci_lower = mean_of_means - t_value * sem
    ci_upper = mean_of_means + t_value * sem
    
    # Calculate 95% prediction intervals
    # This includes both the uncertainty in the mean and the variability of individual observations
    pred_std = np.sqrt(std_band**2 + sem**2)  # Combines original variability and uncertainty in mean
    pi_lower = mean_of_means - t_value * pred_std
    pi_upper = mean_of_means + t_value * pred_std

    results = {
        'mean_of_means': mean_of_means,
        'std_of_means': std_of_means,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'pi_lower': pi_lower,
        'pi_upper': pi_upper,
        'agb_total_pg': round(np.nansum(mean_of_means)/1e9 * area_pix_ha, 4),  # AGB total in Pg
        #'raster_profile': profile
    }

    # Additional calculations for the total sum, converted from density (Mg/ha) to stock (Pg)
    #
    Pg_conversion = (1 / 1e9 * area_pix_ha)
    
    # Calculate sum for each simulation
    #simulation_sums = round(np.nansum(simulations, axis=(1, 2)) / 1e9 * area_pix_ha, 4)
    simulation_sums = np.nansum(simulations * Pg_conversion, axis=(1, 2))
    
    # Mean of the simulation sums
    mean_sum = np.mean(simulation_sums)
    
    # Standard deviation of the simulation sums
    std_sum = np.std(simulation_sums, ddof=1)
    
    # Standard error of the mean sum
    sem_sum = std_sum / np.sqrt(n_simulations)
    
    # t-value for 95% CI with n-1 degrees of freedom
    t_value = stats.t.ppf(0.975, n_simulations - 1)
    
    # 95% confidence interval for the sum
    sum_ci_lower = mean_sum - t_value * sem_sum
    sum_ci_upper = mean_sum + t_value * sem_sum
    
    # Calculate total variance for prediction interval
    # For the prediction interval, we need to account for the total variance in the data
    total_variance = np.sum((std_band * Pg_conversion)**2)  # Sum of variance across all pixels
    pred_std_sum = np.sqrt(total_variance + sem_sum**2)
    
    # 95% prediction interval for the sum
    sum_pi_lower = mean_sum - t_value * pred_std_sum
    sum_pi_upper = mean_sum + t_value * pred_std_sum
    
    # Add sum results to the results dictionary
    results.update({
        'mean_sum_pg': mean_sum,
        'std_sum_pg': std_sum,
        'sum_ci_lower_pg': sum_ci_lower,
        'sum_ci_upper_pg': sum_ci_upper,
        'sum_pi_lower_pg': sum_pi_lower,
        'sum_pi_upper_pg': sum_pi_upper,
        'simulation_sums_pg': simulation_sums
    })
    plot_results(mean_of_means, std_of_means, ci_upper, ci_lower)
    
    return results

def plot_results(mean_of_means, std_of_means, ci_upper, ci_lower):
    """Helper function to visualize results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 4))
    
    im1 = axes[0, 0].imshow(mean_of_means, vmin=0, vmax=100)
    axes[0, 0].set_title('Mean of Simulated Means')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(std_of_means, vmin=0, vmax=15, cmap='plasma')
    axes[0, 1].set_title('Standard Deviation of Simulated Means')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].imshow(ci_upper-ci_lower, vmin=0, vmax=20, cmap='inferno')
    axes[1, 0].set_title('95th CI Range')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow((ci_upper-ci_lower)/mean_of_means, vmin=0, vmax=1, cmap='turbo')
    axes[1, 1].set_title('Relative 95th CI Range')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png', dpi=300)
    plt.show()

# # Example usage
# if __name__ == "__main__":
#     # Replace with your raster path
#     raster_path = "path_to_your_raster.tif"
    
#     # Run Monte Carlo analysis
#     mean_of_means, std_of_means, simulations = monte_carlo_uncertainty(raster_path)
    
#     # Read original data for comparison
#     with rasterio.open(raster_path) as src:
#         original_mean = src.read(1)
#         original_std = src.read(2)
    
#     # Plot results
#     plot_results(original_mean, mean_of_means, original_std, std_of_means)
    
#     # Calculate summary statistics
#     print(f"Original mean average: {np.mean(original_mean)}")
#     print(f"Simulated mean average: {np.mean(mean_of_means)}")
#     print(f"Original std average: {np.mean(original_std)}")
#     print(f"Std of means average: {np.mean(std_of_means)}")
    
#     # The standard error should be approximately original_std/sqrt(n_simulations)
#     theoretical_se = original_std / np.sqrt(50)
#     print(f"Theoretical SE average: {np.mean(theoretical_se)}")
#     print(f"Ratio of std_of_means to theoretical SE: {np.mean(std_of_means/theoretical_se)}")

#Return a common mask for a set of input ma
def common_mask(ma_list, apply=False):
    if type(ma_list) is not list:
        print("Input must be list of masked arrays")
        return None
    #Note: a.mask will return single False if all elements are False
    #np.ma.getmaskarray(a) will return full array of False
    #ma_list = [np.ma.array(a, mask=np.ma.getmaskarray(a), shrink=False) for a in ma_list]
    a = np.ma.array(ma_list, shrink=False)
    #Check array dimensions
    #Check dtype = bool
    #Masked values are listed as true, so want to return any()
    #a+b+c - OR (any)
    mask = np.ma.getmaskarray(a).any(axis=0)
    #a*b*c - AND (all)
    #return a.all(axis=0)
    if apply:
        return [np.ma.array(b, mask=mask) for b in ma_list] 
    else:
        return mask

def get_cog_s3_path(TILE_NUM, TINDEX_FN):
    tindex = pd.read_csv(TINDEX_FN)
    if tindex.tile_num.isin([TILE_NUM]).any():
        return tindex[tindex.tile_num.isin([TILE_NUM])].s3_path.to_list()[0]
    else:
        print(f"Tile {TILE_NUM} not in {TINDEX_FN}")
        return None

def get_discrete_cmap(cmap='plasma', N_COLORS=5):
    # Get the Viridis colormap
    cmap = cm.get_cmap(cmap)
    
    # Create a discrete colormap with N colors
    colors = cmap(np.linspace(0, 1, N_COLORS))
    discrete_cmap = plt.cm.colors.ListedColormap(colors)
    return discrete_cmap
    
def add_colorbar(mappable, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    plt.sca(last_axes)
    return cbar

def do_colorbar(ax, fig, image_hidden, label, SIZE='5%', EXTEND='max', TICKS=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=SIZE, pad=0.15) #pad=-0.1)
    if TICKS is None:
        cb = fig.colorbar(image_hidden, cax=cax, orientation='vertical', extend=EXTEND)
    else:
        cb = fig.colorbar(image_hidden, cax=cax, orientation='vertical', extend=EXTEND, ticks=TICKS)
    cb.set_label(label)
    return cb

def read_clip_raster(r_fn, shapes, bandnum=None, ndv=None):
    with rasterio.open(r_fn) as dataset:
        print(dataset.profile)
        arr3d, out_transform = rasterio.mask.mask(dataset, shapes, crop=True)
        if bandnum is not None:
            arr3d = arr3d[bandnum-1,:,:]
        if ndv is not None:   
            arr3d = np.ma.masked_where( (arr3d == ndv ) , arr3d)
    return arr3d

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def classify_forest(age_array):
    """
    Classify a 2D array of forest age values into forest classes.
    
    Parameters:
    -----------
    age_array : numpy.ndarray
        2D array of forest age values
        
    Returns:
    --------
    numpy.ndarray
        2D array with classified values:
        0 = Non-forest (age <= 0, including NaN)
        1 = Re-growth forest (0 < age <= 36)
        2 = Young forest (36 < age <= 100)
        3 = Mature forest (100 < age <= 150)
        4 = Old-growth forest (age > 150)
    """
    # Create output array with same shape as input
    classified = np.zeros_like(age_array, dtype=np.int8)
    
    # Handle NaN values (classify as 0 - Non-forest)
    classified[np.isnan(age_array)] = 0
    
    # Apply thresholds
    classified[(age_array > 0) & (age_array < 36)] = 1    # Re-growth forest
    classified[(age_array >= 36) & (age_array < 100)] = 2  # Young forest
    classified[(age_array >= 100) & (age_array < 150)] = 3 # Mature forest
    classified[age_array >= 150] = 4                        # Old-growth forest
    
    return classified

# Helper function to create a single histogram plot
def plot_single_histogram(ax, raster_data, title, class_values, class_colors):
    # Use np.histogram for binning and counts
    values, counts = np.unique(raster_data, return_counts=True)
    
    # Sort the values to ensure proper alignment of colors
    sorted_indices = np.argsort(values)
    print(sorted_indices)
    values = values[sorted_indices]
    counts = counts[sorted_indices]
    print(f'{title}: {values}, {counts}')

    # Create bar plot
    ax.bar(values, counts, color=[class_colors[np.where(class_values == val)[0][0]] if val in class_values else 'blue' for val in values], edgecolor='black')
    
    # Label bars with counts on top
    for i, count in enumerate(counts):
        ax.text(values[i], count + 0.5, str(count), ha='center', va='bottom')

    ax.set_title(title)
    ax.set_xlabel("Class Value")
    ax.set_ylabel("Frequency")
    ax.set_xticks(class_values) # Ensure x-ticks match class values
    ax.tick_params(axis='x', rotation=45)  # Rotate x-tick labels for readability
        
def plot_raster_histograms(raster_data1, raster_data2, class_values, class_colors, title1, title2):
    """
    Plots histograms of two raster datasets, comparing their class distributions.

    Args:
        raster_data1:  The first raster data (e.g., a masked array).
        raster_data2:  The second raster data.
        class_values:  A list of unique class values in the rasters.
        class_colors: A list of colors corresponding to the class values. 
                       Must be the same length as class_values.
        title1:  Title for the first raster's histogram.
        title2:  Title for the second raster's histogram.
        save_path: Optional path to save the plot as an image.
    """

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Use subplots for side-by-side comparison

    # Plot the first raster's histogram
    plot_single_histogram(axs[0], raster_data1, title1, class_values, class_colors)

    # Plot the second raster's histogram
    plot_single_histogram(axs[1], raster_data2, title2, class_values, class_colors)

    fig.tight_layout()  # Adjust layout to prevent labels from overlapping

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

def visualize_forest_classes(classified_array, original_age=None, title="Forest Age Classes", cmap = cmap_standage, 
                             class_colors = colors_standage,  class_names = names_standage, class_values=values_standage):
    """
    Visualize the forest classification.
    
    Parameters:
    -----------
    classified_array : numpy.ndarray
        2D array with classified forest values (0-4)
    original_age : numpy.ndarray, optional
        Original age values for reference
    title : str
        Plot title
    """
    # Define class names and colors
    
    
    # Create custom colormap
    
    norm_standage_cats = mcolors.BoundaryNorm(np.arange(-0.5, len(class_names)+0.5, 1), cmap_standage.N)
    
    # Set up figure
    if original_age is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Plot classified forest
    im = ax1.imshow(classified_array, cmap=cmap, norm=norm_standage_cats, 
                    interpolation='nearest'
                   )
    ax1.set_title(title)
    
    # Create legend-like color bar
    cbar = plt.colorbar(im, ax=ax1, ticks=np.arange(len(names_standage)), shrink=0.8)
    cbar.ax.set_yticklabels(class_names)
    
    
    # Plot original age data if provided
    if original_age is not None:
        im2 = ax2.imshow(original_age, cmap='nipy_spectral', vmin=0, vmax=250, interpolation='nearest')
        ax2.set_title('Original Forest Age (years)')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('forest_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification summary
    print("\nForest Classification Summary:")
    values, counts = np.unique(classified_array, return_counts=True)
    summary = pd.DataFrame({
        'Class ID': values,
        'Class Name': [class_names[v] for v in values],
        'Pixel Count': counts,
        'Percentage': counts / np.sum(counts) * 100
    })
    print(summary)

def classify_and_analyze_forest_age(age_array, pixel_size_ha=0.09, class_names = names_standage, class_values = values_standage, class_colors = colors_standage):
    """
    Classify forest age array and analyze area statistics.
    
    Parameters:
    -----------
    age_array : numpy.ndarray
        2D array of forest age values
    pixel_size_ha : float
        Size of each pixel in hectares (default 0.09 ha for 30m resolution)
        
    Returns:
    --------
    tuple
        (classified_array, statistics_df)
    """
    # Classify the forest
    classified = classify_forest(age_array)
    
    # Create visualization
    visualize_forest_classes(classified, original_age=age_array)
    
    # Calculate statistics
    values, counts = np.unique(classified, return_counts=True)
    area_ha = counts * pixel_size_ha
    
    # Define class names and thresholds for the table
    #class_names = ['Non-forest', 'Re-growth forest', 'Young forest', 'Mature forest', 'Old-growth forest']
    threshold_desc = [
        '≤ 0 years (including NaN)',
        '0-36 years',
        '36-100 years',
        '100-150 years',
        '> 150 years'
    ]
    
    # Create DataFrame for statistics
    stats_df = pd.DataFrame({
        'Class ID': values,
        'Class Name': [class_names[int(v)] for v in values],
        #'Age Range': [threshold_desc[v] for v in values],
        'Pixel Count': counts,
        'Area (hectares)': area_ha,
        'Percentage': counts / np.sum(counts) * 100
    })
    
    # Print results
    print("\nForest Age Classification Statistics:")
    #print(stats_df)
    
    # Create pie chart for visualization
    plt.figure(figsize=(4, 4))
    plt.pie(stats_df['Area (hectares)'], 
            labels=stats_df['Class Name'], 
            autopct='%1.1f%%',
            colors=[class_colors[v].lower() for v in values],
            startangle=90)
    plt.axis('equal')
    plt.title('Forest Classification by Area')
    plt.tight_layout()
    plt.savefig('forest_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return classified, stats_df

# def monte_carlo_raster_uncertainty(arr, num_simulations=50):
#     """
#     Performs Monte Carlo simulation on a 2-band raster where:
#     - Band 1 contains pixel-level means
#     - Band 2 contains pixel-level standard deviations
    
#     Parameters:
#         raster_path (str): Path to the raster file
#         num_simulations (int): Number of Monte Carlo simulations to run
        
#     Returns:
#         tuple: (mean_sum, std_sum) - The mean and standard deviation of the sum
#                across all simulations
#     """
#     # # Read the raster data
#     # with rasterio.open(raster_path) as src:
#     #     # Read all bands
#     #     raster_data = src.read()
        
#     #     # Extract the bands
#     #     means = raster_data[0]  # Band 1 (means)
#     #     stds = raster_data[1]   # Band 2 (standard deviations)

#     means = arr[0,:,:]
#     stds = arr[1,:,:]
    
#     # Initialize array to store the sum for each simulation
#     simulation_sums = np.zeros(num_simulations)
    
#     # Generate random samples based on normal distribution
#     for i in range(num_simulations):
#         # Generate random values for each pixel based on its mean and std
#         random_sample = np.random.normal(loc=means * area_pix_ha , scale=stds * area_pix_ha)
        
#         # Calculate the sum of this realization and store it
#         simulation_sums[i] = np.nansum(random_sample)

#     #simulation_sums = simulation_sums[~np.isnan(simulation_sums)]
    
#     # Calculate the mean and standard deviation of the sums
#     mean_of_sums = np.mean(simulation_sums)
#     std_of_sums = np.std(simulation_sums)
    
#     # Optional: visualization of the distribution of sums
#     plt.figure(figsize=(10, 3))
#     plt.hist(simulation_sums, bins=20, alpha=0.7)
#     plt.axvline(mean_of_sums, color='red', linestyle='dashed', linewidth=2)
#     plt.axvline(mean_of_sums + std_of_sums, color='green', linestyle='dashed', linewidth=1)
#     plt.axvline(mean_of_sums - std_of_sums, color='green', linestyle='dashed', linewidth=1)
#     plt.title('Distribution of Monte Carlo Sum Estimates')
#     plt.xlabel('Sum Value')
#     plt.ylabel('Frequency')
#     plt.grid(alpha=0.3)
    
#     print(f"Mean of sums: {mean_of_sums}")
#     print(f"Standard deviation of sums: {std_of_sums}")
#     print(f"Coefficient of variation: {std_of_sums/mean_of_sums:.4f}")
    
#     return mean_of_sums, std_of_sums, random_sample

# # Example usage:
# # mean_sum, std_sum = monte_carlo_raster_uncertainty('path/to/your/raster.tif')
