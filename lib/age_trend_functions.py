import numpy as np

def get_default_age_classification():
    """Return default age cohort classification scheme."""
    return {
        'non_forest': [0, 0],
        'young_forest': [1, 20],
        'maturing_forest': [21, 80],
        'mature_forest': [81, 200],
        'old_growth_forest': [201, 9999]  # Using 9999 instead of inf for JSON compatibility
    }

def get_kendall_trend_class_labels():
    """
    Return the Kendall Tau trend class labels from compute_trends.py.
    Maps the class values (0-10) to descriptive labels.
    """
    return {
        0: 'No data',
        1: 'Strong sig. positive',
        2: 'Moderate sig. positive',
        3: 'Weak sig. positive',
        4: 'Very weak sig. positive',
        5: 'Non-sig. positive',
        6: 'Non-sig. negative',
        7: 'Very weak sig. negative',
        8: 'Weak sig. negative',
        9: 'Moderate sig. negative',
        10: 'Strong sig. negative'
    }

def get_trend_category_mapping():
    """
    Return mapping from Kendall Tau classes to simpler trend categories.
    Groups the detailed Kendall classes into 5 simplified categories.
    """
    return {
        0: -1,   # No data -> ignore
        1: 4,    # Strong sig. positive -> strong increase
        2: 4,    # Moderate sig. positive -> strong increase
        3: 3,    # Weak sig. positive -> moderate increase
        4: 3,    # Very weak sig. positive -> moderate increase
        5: 2,    # Non-sig. positive -> stable
        6: 2,    # Non-sig. negative -> stable
        7: 1,    # Very weak sig. negative -> moderate decline
        8: 1,    # Weak sig. negative -> moderate decline
        9: 0,    # Moderate sig. negative -> strong decline
        10: 0    # Strong sig. negative -> strong decline
    }

def classify_age_cohorts(age_array, classification_scheme):
    """
    Classify age values into cohorts using provided scheme.
    
    Parameters:
    age_array: numpy array of age values
    classification_scheme: dict with age ranges
    """
    # Initialize cohort array
    cohorts = np.full_like(age_array, -1, dtype=np.int8)  # -1 for unclassified
    
    for i, (cohort_name, age_range) in enumerate(classification_scheme.items()):
        min_age, max_age = age_range
        
        if max_age >= 9999:  # Handle large numbers as infinity
            mask = age_array > min_age
        else:
            mask = (age_array >= min_age) & (age_array <= max_age)
        
        valid_mask = ~np.isnan(age_array)
        cohorts[mask & valid_mask] = i
    
    return cohorts, list(classification_scheme.keys())

def remap_trend_classes(kendall_classes):
    """
    Remap detailed Kendall Tau classes to simplified trend categories.
    
    Parameters:
    kendall_classes: numpy array of Kendall Tau class values (0-10)
    
    Returns:
    Numpy array with remapped trend categories (0-4)
    """
    # Get the mapping dictionary
    mapping = get_trend_category_mapping()
    
    # Create output array
    trend_categories = np.full_like(kendall_classes, -1, dtype=np.int8)
    
    # Apply mapping
    for kendall_class, trend_category in mapping.items():
        trend_categories[kendall_classes == kendall_class] = trend_category
    
    return trend_categories