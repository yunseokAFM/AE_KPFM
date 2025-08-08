#!/usr/bin/env python3
"""
GMM Analysis for 2D Material Characterization
Comprehensive analysis tool for Gaussian Mixture Model clustering of height data
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import normaltest
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

try:
    import diptest
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False

def analyze_csv_and_generate_outputs(csv_file_path, height_column='KPFM_Fixed', output_dir=None, 
                                    min_weight_threshold=0.05, filter_minor_components=True):
    """
    Main function to analyze CSV data and generate GMM analysis results    
    """
    
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        output_dir = f"{base_name}_outputs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file_path)
    
    if height_column not in df.columns:
        return None
    
    x_col = None
    y_col = None
    
    for col in df.columns:
        if col.lower() in ['x']:
            x_col = col
        elif col.lower() in ['y']:
            y_col = col
    
    if x_col is None or y_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
    
    height_data = df[height_column].dropna().values.reshape(-1, 1)
    
    validation_results = perform_gmm_analysis_with_filtering(
        height_data, 
        max_components=6,
        min_weight_threshold=min_weight_threshold, 
        filter_minor_components=filter_minor_components
    )
    
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    spatial_csv_path = os.path.join(output_dir, f"{base_name}_spatial_data.csv")
    generate_spatial_csv(df, x_col, y_col, height_column, validation_results, spatial_csv_path)
    
    results_csv_path = os.path.join(output_dir, f"{base_name}_gmm_fitting_results.csv")
    generate_gmm_results_csv(validation_results, base_name, results_csv_path)
    
    return validation_results

def perform_gmm_analysis_with_filtering(height_data, max_components=6, min_weight_threshold=0.05, filter_minor_components=True):
    """Perform comprehensive GMM analysis with statistical validation and optional filtering"""
    
    validation_results = {}
    
    n_components_range = range(1, max_components + 1)
    bic_scores = []
    aic_scores = []
    log_likelihoods = []
    silhouette_scores = []
    
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=42, max_iter=200, n_init=1)
        gmm.fit(height_data)
        
        bic_scores.append(gmm.bic(height_data))
        aic_scores.append(gmm.aic(height_data))
        log_likelihoods.append(gmm.score(height_data))
        
        if n > 1:
            labels = gmm.predict(height_data)
            try:
                sil_score = silhouette_score(height_data, labels)
                silhouette_scores.append(sil_score)
            except:
                silhouette_scores.append(0)
        else:
            silhouette_scores.append(0)
    
    optimal_n_bic = n_components_range[np.argmin(bic_scores)]
    optimal_n_aic = n_components_range[np.argmin(aic_scores)]
    
    validation_results['model_selection'] = {
        'bic_scores': bic_scores,
        'aic_scores': aic_scores,
        'log_likelihoods': log_likelihoods,
        'silhouette_scores': silhouette_scores,
        'optimal_n_bic': optimal_n_bic,
        'optimal_n_aic': optimal_n_aic
    }
    
    if HAS_DIPTEST:
        try:
            dip_stat, dip_pval = diptest.diptest(height_data.flatten())
            validation_results['dip_test'] = {
                'statistic': dip_stat,
                'p_value': dip_pval,
                'is_multimodal': dip_pval < 0.05,
                'interpretation': 'Multimodal' if dip_pval < 0.05 else 'Unimodal'
            }
        except Exception as e:
            validation_results['dip_test'] = {'error': str(e)}
    else:
        validation_results['dip_test'] = {'error': 'diptest package not available'}
    
    initial_gmm = GaussianMixture(n_components=optimal_n_bic, random_state=42, max_iter=200, n_init=1)
    initial_gmm.fit(height_data)
    initial_labels = initial_gmm.predict(height_data)
    initial_probabilities = initial_gmm.predict_proba(height_data)
    
    if filter_minor_components and min_weight_threshold > 0:
        component_info = {}
        components_to_keep = []
        
        for i in range(optimal_n_bic):
            mask = initial_labels == i
            n_points = np.sum(mask)
            weight = initial_gmm.weights_[i]
            mean = initial_gmm.means_[i, 0]
            std = np.sqrt(initial_gmm.covariances_[i, 0, 0])
            
            keep = weight >= min_weight_threshold
            
            component_info[i] = {
                'n_points': n_points,
                'weight': weight,
                'mean': mean,
                'std': std,
                'keep': keep
            }
            
            if keep:
                components_to_keep.append(i)
        
        validation_results['component_info'] = component_info
        validation_results['original_components'] = optimal_n_bic
        validation_results['filtered_components'] = components_to_keep
        
        if len(components_to_keep) < optimal_n_bic:
            major_mask = np.isin(initial_labels, components_to_keep)
            filtered_data = height_data[major_mask]
            
            final_gmm = GaussianMixture(n_components=len(components_to_keep), 
                                       random_state=42, max_iter=200, n_init=1)
            final_gmm.fit(filtered_data)
            
            final_labels = final_gmm.predict(height_data)
            final_probabilities = final_gmm.predict_proba(height_data)
            
            validation_results['filtering_applied'] = True
        else:
            final_gmm = initial_gmm
            final_labels = initial_labels  
            final_probabilities = initial_probabilities
            validation_results['filtering_applied'] = False
    else:
        final_gmm = initial_gmm
        final_labels = initial_labels
        final_probabilities = initial_probabilities
        validation_results['filtering_applied'] = False
        validation_results['original_components'] = optimal_n_bic
        validation_results['filtered_components'] = list(range(optimal_n_bic))
    
    component_tests = {}
    for i in range(final_gmm.n_components):
        component_data = height_data[final_labels == i]
        
        component_info = {
            'n_points': len(component_data),
            'percentage': len(component_data) / len(height_data) * 100,
            'mean': np.mean(component_data),
            'std': np.std(component_data),
            'gmm_mean': final_gmm.means_[i, 0],
            'gmm_std': np.sqrt(final_gmm.covariances_[i, 0, 0]),
            'gmm_weight': final_gmm.weights_[i]
        }
        
        if len(component_data) > 8:
            try:
                stat, pval = normaltest(component_data.flatten())
                component_info.update({
                    'normality_stat': stat,
                    'normality_pval': pval,
                    'is_normal': pval > 0.05
                })
            except:
                component_info.update({
                    'normality_stat': np.nan,
                    'normality_pval': np.nan,
                    'is_normal': False
                })
        
        component_tests[f'component_{i}'] = component_info
    
    max_probs = np.max(final_probabilities, axis=1)
    validation_results['model_quality'] = {
        'final_bic': final_gmm.bic(height_data),
        'final_aic': final_gmm.aic(height_data),
        'log_likelihood': final_gmm.score(height_data),
        'mean_assignment_probability': np.mean(max_probs),
        'min_assignment_probability': np.min(max_probs)
    }
    
    validation_results['component_tests'] = component_tests
    validation_results['final_gmm'] = final_gmm
    validation_results['final_labels'] = final_labels
    validation_results['final_probabilities'] = final_probabilities
    
    return validation_results

def generate_spatial_csv(df, x_col, y_col, height_column, validation_results, output_path):
    """Generate CSV File: X, Y, Z, Label"""
    
    spatial_data = pd.DataFrame()
    
    if x_col and x_col in df.columns:
        spatial_data['X'] = df[x_col].astype(float)
    else:
        spatial_data['X'] = range(len(df))
        
    if y_col and y_col in df.columns:
        spatial_data['Y'] = df[y_col].astype(float)
    else:
        spatial_data['Y'] = 0.0
    
    spatial_data['Z'] = df[height_column].astype(float)
    
    labels = validation_results['final_labels']
    probabilities = validation_results['final_probabilities']
    
    min_length = min(len(spatial_data), len(labels))
    spatial_data = spatial_data.iloc[:min_length].copy()
    
    spatial_data['GMM_Label'] = labels[:min_length].astype(int)
    spatial_data['Assignment_Probability'] = np.max(probabilities[:min_length], axis=1)
    
    n_components = probabilities.shape[1]
    for i in range(n_components):
        spatial_data[f'Component_{i}_Probability'] = probabilities[:min_length, i]
    
    spatial_data.to_csv(output_path, index=False)
    
    return spatial_data

def generate_gmm_results_csv(validation_results, sample_name, output_path):
    """Generate CSV File: GMM Fitting Results"""
    
    results_data = []
    
    results_data.append(['sample_name', sample_name])
    results_data.append(['analysis_type', 'GMM_validation_with_statistical_testing'])
    results_data.append(['total_data_points', len(validation_results['final_labels'])])
    
    filtering_applied = validation_results.get('filtering_applied', False)
    original_n = validation_results.get('original_components', validation_results['final_gmm'].n_components)
    final_n = validation_results['final_gmm'].n_components
    
    results_data.append(['filtering_applied', filtering_applied])
    results_data.append(['original_components_before_filtering', original_n])
    results_data.append(['final_components_after_filtering', final_n])
    results_data.append(['components_removed', original_n - final_n])
    
    model_sel = validation_results['model_selection']
    results_data.append(['optimal_components_bic', model_sel['optimal_n_bic']])
    results_data.append(['optimal_components_aic', model_sel['optimal_n_aic']])
    results_data.append(['best_silhouette_score', max(model_sel['silhouette_scores'])])
    
    dip_test = validation_results.get('dip_test', {})
    results_data.append(['dip_test_statistic', dip_test.get('statistic', 'N/A')])
    results_data.append(['dip_test_pvalue', dip_test.get('p_value', 'N/A')])
    results_data.append(['is_multimodal', dip_test.get('is_multimodal', 'N/A')])
    results_data.append(['multimodality_interpretation', dip_test.get('interpretation', 'N/A')])
    
    quality = validation_results['model_quality']
    results_data.append(['final_bic_score', quality['final_bic']])
    results_data.append(['final_aic_score', quality['final_aic']])
    results_data.append(['log_likelihood', quality['log_likelihood']])
    results_data.append(['mean_assignment_probability', quality['mean_assignment_probability']])
    results_data.append(['min_assignment_probability', quality['min_assignment_probability']])
    
    n_components = final_n
    results_data.append(['number_of_components', n_components])
    
    component_tests = validation_results.get('component_tests', {})
    for i in range(n_components):
        comp_key = f'component_{i}'
        comp_info = component_tests.get(comp_key, {})
        
        results_data.append([f'component_{i}_mean', comp_info.get('gmm_mean', 'N/A')])
        results_data.append([f'component_{i}_std', comp_info.get('gmm_std', 'N/A')])
        results_data.append([f'component_{i}_weight', comp_info.get('gmm_weight', 'N/A')])
        results_data.append([f'component_{i}_data_points', comp_info.get('n_points', 'N/A')])
        results_data.append([f'component_{i}_percentage', comp_info.get('percentage', 'N/A')])
        results_data.append([f'component_{i}_normality_pvalue', comp_info.get('normality_pval', 'N/A')])
        results_data.append([f'component_{i}_is_normal', comp_info.get('is_normal', 'N/A')])
    
    if filtering_applied and 'component_info' in validation_results:
        filtered_info = validation_results['component_info']
        removed_components = [i for i, info in filtered_info.items() if not info['keep']]
        results_data.append(['removed_components_count', len(removed_components)])
        results_data.append(['removed_components_list', str(removed_components)])
        
        for i in removed_components:
            info = filtered_info[i]
            results_data.append([f'removed_component_{i}_weight', info['weight']])
            results_data.append([f'removed_component_{i}_points', info['n_points']])
            results_data.append([f'removed_component_{i}_mean', info['mean']])
    
    for i, (bic, aic, loglik) in enumerate(zip(
        model_sel['bic_scores'],
        model_sel['aic_scores'], 
        model_sel['log_likelihoods']
    )):
        results_data.append([f'model_{i+1}_components_bic', bic])
        results_data.append([f'model_{i+1}_components_aic', aic])
        results_data.append([f'model_{i+1}_components_loglik', loglik])
    
    results_df = pd.DataFrame(results_data, columns=['Parameter', 'Value'])
    results_df.to_csv(output_path, index=False)
    
    return results_df

if __name__ == "__main__":
    csv_file = 'gmm_test_5.csv'
    height_col = 'Height'
    min_weight_threshold = 0.15
    filter_components = True
    
    if os.path.exists(csv_file):
        results = analyze_csv_and_generate_outputs(
            csv_file, 
            height_col, 
            min_weight_threshold=min_weight_threshold,
            filter_minor_components=filter_components
        )