import numpy as np
import pandas as pd

def apply_2D_polynomial_flattening(df, degree=2):
    """
    Apply 2D polynomial flattening to height data within masked regions.    
    """
    
    # Filter data for mask == 1 to perform polynomial fitting
    mask_1_df = df[df['mask'] == 1]
    
    # Extract coordinates and height values where mask == 1
    X_mask_1 = mask_1_df['X'].values.flatten()
    Y_mask_1 = mask_1_df['Y'].values.flatten()
    Z_mask_1 = mask_1_df['Z Height'].values.flatten()
    
    # Build design matrix for 2D polynomial fitting
    A_mask_1 = np.vstack([X_mask_1**i * Y_mask_1**j 
                          for i in range(degree + 1) 
                          for j in range(degree + 1)]).T
    
    # Solve for polynomial coefficients using least squares
    coeffs_mask_1, _, _, _ = np.linalg.lstsq(A_mask_1, Z_mask_1, rcond=None)
    
    # Apply fitted polynomial to entire dataset
    X_full = df['X'].values.flatten()
    Y_full = df['Y'].values.flatten()
    Z_full = df['Z Height'].values.flatten()
    
    # Build design matrix for entire dataset
    A_full = np.vstack([X_full**i * Y_full**j 
                        for i in range(degree + 1) 
                        for j in range(degree + 1)]).T
    
    # Compute fitted surface using coefficients from masked region
    Z_fitted = np.dot(A_full, coeffs_mask_1)
    
    # Calculate flattened height by subtracting fitted surface
    Z_flattened = Z_full - Z_fitted
    
    # Add flattened heights to dataframe
    df['Z_Height_Flattening'] = Z_flattened
    
    return df