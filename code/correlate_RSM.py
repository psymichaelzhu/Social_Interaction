#%% packages
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multitest import multipletests


#%% function

def correlate_two_rsms(rsm1, rsm2, n_permutation=1000, if_display=True):
    """Compute correlation between two RSMs with permutation testing
    Args:
        rsm1 (np.ndarray): First RSM
        rsm2 (np.ndarray): Second RSM
        n_permutation (int): Number of permutations for null distribution
        if_display (bool): Whether to display null distribution plot
    Returns:
        tuple: (correlation coefficient, permutation p-value)
    """
    # Get lower triangular indices
    tril_idx = np.tril_indices_from(rsm1, k=-1)
    
    vec1 = rsm1[tril_idx]
    vec2 = rsm2[tril_idx]
    
    # Compute observed correlation
    observed_r, _ = spearmanr(vec1, vec2)
    
    # Permutation test
    null_dist = np.zeros(n_permutation)
    for i in range(n_permutation):
        # Shuffle first RSM by row
        perm_idx = np.random.permutation(len(vec1))
        null_dist[i], _ = spearmanr(vec1[perm_idx], vec2)

    # Compute two-tailed p-value
    p_value = (1 + np.sum(np.abs(null_dist) >= np.abs(observed_r))) / (1 + n_permutation)
    
    if if_display:
        plt.figure(figsize=(10,6))
        plt.hist(null_dist, bins=50, edgecolor='black')
        plt.axvline(x=observed_r, color='r', linestyle='--')
        plt.title('Null Distribution of Correlations', fontsize=22)
        plt.xlabel('Correlation Coefficient', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.annotate(f'r = {observed_r:.3f}\np = {p_value:.3f}', 
                    xy=(0.7, 0.8), xycoords='axes fraction', 
                    fontsize=18)
        plt.show()
        
    return observed_r, p_value

def correlate_two_dicts(rsm_dict1, rsm_dict2, n_permutation=1000, multiple_comparison='fdr_bh'):
    """Compute correlations between all pairs of RSMs from two dictionaries
    Args:
        rsm_dict1 (dict): First dictionary with RSM names as keys and RSMs as values
        rsm_dict2 (dict): Second dictionary with RSM names as keys and RSMs as values
        n_permutation (int): Number of permutations for null distribution
        multiple_comparison (str): Multiple comparison correction method ('fdr_bh' or 'bonferroni')
    Returns:
        pd.DataFrame: DataFrame with correlation results including adjusted p-values and significance
    """
    # Initialize lists to store results
    results = []
    
    # Compute correlations for all pairs
    for key1, rsm1 in rsm_dict1.items():
        for key2, rsm2 in rsm_dict2.items():
            print(key1,"-",key2)
            r, p = correlate_two_rsms(rsm1, rsm2, n_permutation=n_permutation, if_display=False)
            results.append({
                'dict1': key1,
                'dict2': key2,
                'r': r,
                'p': p
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Apply multiple comparison correction
    _, p_adj, _, _ = multipletests(df['p'], method=multiple_comparison)
    df['q'] = p_adj
    
    # Add significance symbols
    def get_sig_symbol(p):
        if p > 0.05:
            return 'n.s.'
        elif p > 0.01:
            return '*'
        elif p > 0.001:
            return '**'
        else:
            return '***'
    
    df['sig_sign'] = df['q'].apply(get_sig_symbol)
    
    return df

def plot_correlation_matrix(correlation_df, figsize=(10,10)):
    """Plot correlation matrix with significance markers
    Args:
        correlation_df (pd.DataFrame): DataFrame from correlate_two_dicts
        figsize (tuple): Figure size
    """
    # Pivot the data into a matrix form
    matrix = correlation_df.pivot(index='dict1', columns='dict2', values='r')
    sig_matrix = correlation_df.pivot(index='dict1', columns='dict2', values='sig_sign')
    
    plt.figure(figsize=figsize)
    scale_length=0.5
    sns.heatmap(matrix, cmap='RdBu_r', center=0, vmin=-scale_length, vmax=scale_length,
                annot=False, fmt='.2f', cbar_kws={'label': 'Correlation'})
    
    # Add significance markers
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            if sig_matrix.iloc[i,j] != 'n.s.':
                plt.text(j+0.5, i+0.3, sig_matrix.iloc[i,j],
                        ha='center', va='center', color='black',
                        fontsize=12)
    
    plt.title('RSM Correlations', fontsize=22)
    plt.xlabel('Dict 2', fontsize=18)
    plt.ylabel('Dict 1', fontsize=18)
    plt.xticks(fontsize=18, rotation=90)
    plt.yticks(fontsize=18, rotation=0)
    
    plt.tight_layout()
    plt.show()

#%% load neural and clip data
neural_rsm=np.load('../data/RSA/neural_rsm.npy',allow_pickle=True).item()


#%% reorganize neural RSM to combine roi and side
roi_neural_rsm = {}

# Process each subject/group level
for level in neural_rsm.keys():
    roi_neural_rsm[level] = {}
    
    # For each ROI
    for roi in neural_rsm[level].keys():
        # For each side
        for side in neural_rsm[level][roi].keys():
            # Create new key combining side and ROI
            new_key = f'{roi}_{side}'
            roi_neural_rsm[level][new_key] = neural_rsm[level][roi][side]

#%% correlate CLIP RSM with reorganized neural RSM for each subject/group
for sub in roi_neural_rsm.keys():
    print(f'\nCorrelating {sub} with CLIP RSM:')
    corr_df = correlate_two_dicts(roi_neural_rsm[sub], roi_neural_rsm[sub], 
                                 n_permutation=0, 
                                 multiple_comparison='fdr_bh')
    plot_correlation_matrix(corr_df)

#%% load model_for_rsm
model_rsm=np.load('../data/RSA/model_rsm.npy',allow_pickle=True).item()
model_rsm.keys()


#%% load CLIP RSM
clip_rsm=np.load('../data/RSA/clip_rsm.npy',allow_pickle=True).item()
clip_rsm.keys()

#%% correlate CLIP RSM with reorganized neural RSM for each subject/group
# Correlate each level with CLIP RSM and store results
# List of RSMs to combine
rsm_list = [clip_rsm, model_rsm]

# Combine all RSMs into one dictionary
combined_rsm = {}
for rsm in rsm_list:
    combined_rsm.update(rsm)

# Correlate combined RSM with neural RSM for each subject/group
corr_dfs = {}
for level in roi_neural_rsm.keys():
    print(level)
    print(f'\nCorrelating {level} with combined RSM:')
    corr_df = correlate_two_dicts(roi_neural_rsm[level], combined_rsm,
                                 n_permutation=0,
                                 multiple_comparison='fdr_bh') 
    plot_correlation_matrix(corr_df)
    corr_dfs[level] = corr_df

# Average correlation values across subjects (excluding group)
subject_dfs = [corr_dfs[level] for level in corr_dfs.keys() if level != 'group']
average_df = subject_dfs[0].copy()

# Calculate mean correlation values
for i in range(len(average_df)):
    r_values = [df.iloc[i]['r'] for df in subject_dfs]
    average_df.iloc[i, average_df.columns.get_loc('r')] = np.mean(r_values)
    
    # Use majority vote for significance
    sig_signs = [df.iloc[i]['sig_sign'] for df in subject_dfs]
    most_common_sign = max(set(sig_signs), key=sig_signs.count)
    average_df.iloc[i, average_df.columns.get_loc('sig_sign')] = most_common_sign

print('\nAverage correlation across subjects:')
plot_correlation_matrix(average_df)

# %% CLIP
corr_df = correlate_two_dicts(combined_rsm, combined_rsm, 
                                 n_permutation=0, 
                                 multiple_comparison='fdr_bh')
plot_correlation_matrix(corr_df)
# %% 

