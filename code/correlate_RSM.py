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

def correlate_dicts_asymmetric(reference_dict, candidate_dict_dict, n_permutation=1000, multiple_comparison='fdr_bh'):
    """Compute correlations between all pairs of RSMs from two dictionaries
    Args:
        reference_dict (dict): First dictionary with RSM names as keys and RSMs as values
        candidate_dict_dict (dict): Dictionary of dictionaries, where each inner dictionary contains RSM names as keys and RSMs as values
        n_permutation (int): Number of permutations for null distribution
        multiple_comparison (str): Multiple comparison correction method ('fdr_bh' or 'bonferroni')
    Returns:
        pd.DataFrame: DataFrame with correlation results including adjusted p-values and significance
    """
    # Initialize lists to store results
    results = []
    
    # Iterate through each module in candidate_dict_dict
    for module, candidate_dict in candidate_dict_dict.items():
        # Compute correlations for all pairs within this module
        for key1, rsm1 in reference_dict.items():
            for key2, rsm2 in candidate_dict.items():
                #print(f"{key1} - {module}_{key2}")
                r, p = correlate_two_rsms(rsm1, rsm2, n_permutation=n_permutation, if_display=False)
                results.append({
                    'reference': key1,
                    'candidate': key2,
                    'module': module,
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

def plot_heatmap_asymmetric(correlation_df, candidate_name = "Candidate RSM", reference_name = "Neural RSM (by ROI)", figsize=(10,10)):
    """Plot correlation matrix with significance markers
    Args:
        correlation_df (pd.DataFrame): DataFrame from correlate_two_dicts
        figsize (tuple): Figure size
    """
    def pivot_correlation_df(correlation_df, value_col):
        """Create pivot tables for correlation values and significance markers, with columns grouped by module
        """

        sorted_df = correlation_df.sort_values('module')
        
        ordered_refs = np.sort(sorted_df['reference'].unique())
        ordered_cands = sorted_df['candidate'].unique()
        
        matrix = sorted_df.pivot(
            index='reference',
            columns='candidate',
            values=value_col
        )
        
        matrix = matrix.reindex(columns=ordered_cands)
        matrix = matrix.reindex(index=ordered_refs)
        
        return matrix
    
    matrix = pivot_correlation_df(correlation_df, 'r')
    sig_matrix = pivot_correlation_df(correlation_df, 'sig_sign')
    
    plt.figure(figsize=figsize)
    scale_length=0.5
    
    # Get module boundaries
    modules = correlation_df['module'].unique()
    module_boundaries = []
    current_pos = 0
    for module in modules:
        module_cols = correlation_df[correlation_df['module'] == module]['candidate'].nunique()
        current_pos += module_cols
        if current_pos < len(matrix.columns):  
            module_boundaries.append(current_pos)
    
    # Plot heatmap
    g = sns.heatmap(matrix, cmap='RdBu_r', center=0, vmin=-scale_length, vmax=scale_length,
                annot=False, fmt='.2f', cbar_kws={'label': 'Correlation'})
    cbar = g.collections[0].colorbar
    cbar.set_label('Spearman Correlation', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    
    # Add white lines between modules
    for boundary in module_boundaries:
        plt.axvline(x=boundary, color='white', linewidth=2)
    # Add significance markers
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            if sig_matrix.iloc[i,j] != 'n.s.':
                plt.text(j+0.5, i+0.3, sig_matrix.iloc[i,j],
                        ha='center', va='center', color='black',
                        fontsize=12)
    
    #plt.title('RSM Correlations', fontsize=24)
    plt.xlabel(candidate_name, fontsize=22)
    plt.ylabel(reference_name, fontsize=22)
    plt.xticks(fontsize=18, rotation=90)
    plt.yticks(fontsize=18, rotation=0)

    
    plt.tight_layout()
    plt.show()

def correlate_self(dict_dict, n_permutation=1000, multiple_comparison='fdr_bh', if_display=True, figsize=(10,10), scale_length=0.5):
    """
    Compute correlations between all pairs of RSMs within a dictionary of dictionaries
    
    Args:
        dict_dict (dict): Dictionary of dictionaries containing RSMs
        n_permutation (int): Number of permutations for null distribution
        multiple_comparison (str): Multiple comparison correction method
        if_display (bool): Whether to display correlation heatmap
        figsize (tuple): Figure size for heatmap
        
    Returns:
        tuple: (correlation DataFrame, correlation matrix ordered by modules)
    """
    # Compute all correlations
    results = []
    for module1, dict1 in dict_dict.items():
        for module2, dict2 in dict_dict.items():
            for key1, rsm1 in dict1.items():
                for key2, rsm2 in dict2.items():
                    r, p = correlate_two_rsms(rsm1, rsm2, 
                                            n_permutation=n_permutation, 
                                            if_display=False)
                    results.append({
                        'module': module1,
                        'reference': f"{module1}~{key1}",
                        'candidate': f"{module2}~{key2}",
                        'r': r,
                        'p': p
                    })
    
   
    df = pd.DataFrame(results)
    #  multiple comparison correction
    _, p_adj, _, _ = multipletests(df['p'], method=multiple_comparison)
    df['q'] = p_adj
    
    # Add significance symbols
    df['sig'] = pd.cut(df['q'], 
                       bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
                       labels=['***', '**', '*', 'n.s.'])
    def create_ordered_matrix(df, value_col):
        """
        Create an ordered correlation matrix from a DataFrame with module-based indices
        """
        # Get module-ordered list of all RSMs
        ordered_rsms = []
        for module in dict_dict.keys():
            module_rsms = [f"{module}~{key}" for key in dict_dict[module].keys()]
            ordered_rsms.extend(sorted(module_rsms))
            
        # Create and reorder matrix
        matrix = df.pivot(index='reference', columns='candidate', values=value_col)
        matrix = matrix.reindex(index=ordered_rsms, columns=ordered_rsms)
        
        # Remove module names from index/columns
        new_index = [x.split('~', 1)[1] for x in matrix.index]
        new_columns = [x.split('~', 1)[1] for x in matrix.columns]
        matrix.index = new_index
        matrix.columns = new_columns
        
        return matrix
        
    # Create ordered matrices
    matrix = create_ordered_matrix(df, 'r')
    sig_matrix = create_ordered_matrix(df, 'sig')
    
    if if_display:
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        g = sns.heatmap(matrix, cmap='RdBu_r', center=0, vmin=-scale_length, vmax=scale_length,
                   annot=False, fmt='.2f', cbar_kws={'label': 'Correlation'})
        
        # Customize colorbar
        cbar = g.collections[0].colorbar
        cbar.set_label('Spearman Correlation', fontsize=20)
        cbar.ax.tick_params(labelsize=20)
  
        
        # Add module boundaries
        current_pos = 0
        for module in dict_dict.keys():
            current_pos += len(dict_dict[module])
            if current_pos < len(matrix):
                plt.axhline(y=current_pos, color='white', linewidth=2)
                plt.axvline(x=current_pos, color='white', linewidth=2)
        
        # Add significance markers
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if sig_matrix.iloc[i,j] != 'n.s.':
                    plt.text(j+0.5, i+0.5, sig_matrix.iloc[i,j],
                            ha='center', va='bottom', color='black',
                            fontsize=8)
        
        #plt.title('RSM Correlations', fontsize=24)
        plt.xlabel('RSMs', fontsize=22)
        plt.ylabel('RSMs', fontsize=22)
        plt.xticks(rotation=90, ha='right', fontsize=20)
        plt.yticks(rotation=0, fontsize=20)
        plt.tight_layout()
        plt.show()
    
    return df, matrix


#%% load neural rsm
neural_rsm=np.load('../data/RSA/neural_rsm.npy',allow_pickle=True).item()
# Try to load roi_neural_rsm from file, if it exists
try:
    roi_neural_rsm = np.load('../data/RSA/roi_neural_rsm.npy', allow_pickle=True).item()
except FileNotFoundError:
    # If file doesn't exist, reorganize neural RSM to combine roi and side
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
    
    # Save the reorganized RSM
    np.save('../data/RSA/roi_neural_rsm.npy', roi_neural_rsm)

_,_ = correlate_self({"neural":roi_neural_rsm['sub01']}, n_permutation=0, multiple_comparison='fdr_bh', if_display=True, figsize=(14,13))

#%% load model and clip rsm
# model rsm
model_rsm=np.load('../data/RSA/model_rsm.npy',allow_pickle=True).item()
print(model_rsm.keys())
# CLIP RSM
clip_rsm=np.load('../data/RSA/clip_rsm.npy',allow_pickle=True).item()
print(clip_rsm.keys())
# Combine all RSMs into one dictionary
combined_rsm = {"CLIP_annotation":clip_rsm, "model_embedding":model_rsm}
_,_ = correlate_self(combined_rsm, n_permutation=0, multiple_comparison='fdr_bh', if_display=True, figsize=(14,13))
#%% correlate candidate RSM with reorganized neural RSM for each subject/group
# Correlate combined RSM with neural RSM for each subject/group
corr_dfs = {}
for level in roi_neural_rsm.keys():
    print(level)
    print(f'\nCorrelating {level} with combined RSM:')
    corr_df = correlate_dicts_asymmetric(roi_neural_rsm[level], combined_rsm,
                                 n_permutation=0,
                                 multiple_comparison='fdr_bh') 
    plot_heatmap_asymmetric(corr_df)
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
plot_heatmap_asymmetric(average_df)
#for each df (subject or average): (20 CLIP annotations + 4 model embeddings) * (11 RoIs * 2 sides) = 528

# %%  regression analysis
def prepare_regression_data(roi, candidate_list, candidate_rsm_dict, neural_rsm):
    """Prepare data for regression analysis by extracting and standardizing RSM data
    Args:
        roi (str): ROI name in 'roi_side' format
        candidate_list (dict): Dictionary specifying which RSMs to use from each module
        candidate_rsm_dict (dict): Dictionary of dictionaries containing candidate RSMs
        neural_rsm (dict): Dictionary containing neural RSMs
    Returns:
        df (pd.DataFrame): DataFrame with RSM values formatted for regression
    """
    def standardize(x,method="rank"):
        """Standardize array by subtracting mean and dividing by std
        """
        if method=="rank":
            return pd.Series(x).rank().values
        elif method=="zscore":
            return (x - np.mean(x)) / np.std(x)
        else:
            return x
    
    # Get lower triangular indices
    n = len(neural_rsm['sub01'][roi])
    tril_idx = np.tril_indices(n, k=-1)
    
    # Extract neural data for all subjects
    neural_data = {}
    for sub in neural_rsm.keys():
        if sub != 'group':
            neural_vec = neural_rsm[sub][roi][tril_idx]
            # Standardize within each subject
            neural_data[sub] = standardize(neural_vec)
    
    # Extract candidate features
    feature_data = {}
    for module, rsm_list in candidate_list.items():
        for rsm_name in rsm_list:
            feature_vec = candidate_rsm_dict[module][rsm_name][tril_idx]
            # Standardize each feature
            feature_data[f"{rsm_name}"] = standardize(feature_vec)
            
    # Create pairs of video indices
    video1_idx = []
    video2_idx = []
    for i, j in zip(*tril_idx):
        video1_idx.append(i)
        video2_idx.append(j)
        
    # Build dataframe
    df_list = []
    for sub in neural_data.keys():
        sub_df = pd.DataFrame({
            'sub': sub,
            'video1': video1_idx,
            'video2': video2_idx,
            'roi': roi,
            'neural': neural_data[sub]
        })
        for feat_name, feat_data in feature_data.items():
            sub_df[feat_name] = feat_data
        df_list.append(sub_df)
    df = pd.concat(df_list, ignore_index=True)
    print(df.head())
    print(df.shape)#n_sub (4) x n_video_pair (244 x 243/2)
    return df

def fit_regression_model(df, multiple_comparison='bonferroni', if_display=True):
    """Fit regression model to predict neural RSM values from candidate features
    Args:
        df (pd.DataFrame): DataFrame with neural and feature data
        if_display (bool): Whether to display regression coefficients plot
    Returns:
        dict: Dictionary containing regression coefficients
    """
    from sklearn.linear_model import LinearRegression
    
    # Prepare X and y
    feature_cols = [col for col in df.columns if col not in ['sub', 'video1', 'video2', 'roi', 'neural']]
    X = df[feature_cols]
    y = df['neural']
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    coef_dict = {'intercept': model.intercept_}
    for feat, coef in zip(feature_cols, model.coef_):
        coef_dict[feat] = coef
        
    if if_display:
        # Sort coefficients to match mixed model order
        feature_cols = [col for col in df.columns if col not in ['sub', 'video1', 'video2', 'roi', 'neural']]
        coef_items = [(feat, coef_dict[feat]) for feat in feature_cols]
        
        plt.figure(figsize=(12,6))
        plt.bar([x[0] for x in coef_items], [x[1] for x in coef_items])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Regression Coefficients for {df.roi.iloc[0]}', fontsize=22)
        plt.xlabel('CandidateFeatures', fontsize=18)
        plt.ylabel('Beta', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.show()
    return coef_dict

def fit_mixed_model(df, multiple_comparison='bonferroni',n_total_comparison=None,if_display=True):
    """Fit mixed effects model to predict neural RSM values from candidate features with random subject effects
    """
    import statsmodels.formula.api as smf
    from tqdm import tqdm
    
    # Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Fit mixed effects model with random intercept for subjects
    feature_cols = [col for col in df.columns if col not in ['sub', 'video1', 'video2', 'roi', 'neural']]
    feature_formula = ' + '.join(feature_cols)
    formula = f"neural ~ {feature_formula}"
    
    mixed_model = smf.mixedlm(formula, df, groups=df["sub"], re_formula="1")
    mixed_result = mixed_model.fit()

    # Manual bootstrap
    try:
        n_bootstrap = 1000
        print(f"Performing bootstrap with {n_bootstrap} iterations...")
        bootstrap_params = []
        
        # Get unique subjects
        subjects = df['sub'].unique()
        
        # Perform bootstrap
        for _ in tqdm(range(n_bootstrap)):
            # Resample subjects with replacement
            boot_subjects = np.random.choice(subjects, size=len(subjects), replace=True)
            
            # Create bootstrap sample
            boot_df = pd.concat([df[df['sub'] == sub] for sub in boot_subjects])
            
            # Fit model on bootstrap sample
            boot_model = smf.mixedlm(formula, boot_df, groups=boot_df["sub"], re_formula="1")
            boot_result = boot_model.fit(disp=False)
            
            # Store parameters
            bootstrap_params.append(boot_result.params)
        
        # Convert to array
        bootstrap_params = np.array(bootstrap_params)
        
        # Calculate confidence intervals
        bootstrap_ci = np.percentile(bootstrap_params, [2.5, 97.5], axis=0)
        bootstrap_ci = pd.DataFrame(bootstrap_ci.T, 
                                  columns=['ci_lower_boot', 'ci_upper_boot'],
                                  index=mixed_result.params.index)
        
        use_bootstrap = True
        print("Bootstrap successful!")
        
    except Exception as e:
        print(f"Bootstrap failed: {str(e)}")
        print("Using standard confidence intervals instead")
        use_bootstrap = False
        bootstrap_ci = pd.DataFrame({
            'ci_lower_boot': mixed_result.conf_int()[0],
            'ci_upper_boot': mixed_result.conf_int()[1]
        }, index=mixed_result.params.index)

    # Multiple comparison correction
    qvalues = mixed_result.pvalues*n_total_comparison
    
    # Get fixed effects coefficients with confidence intervals
    fixed_effects = pd.DataFrame({
        'coef': mixed_result.fe_params,
        'std': mixed_result.bse_fe,
        'pvalue': mixed_result.pvalues,
        'qvalue': qvalues,
        'ci_lower': mixed_result.conf_int()[0],
        'ci_upper': mixed_result.conf_int()[1],
        'tvalue': mixed_result.tvalues,
        'ci_lower_boot': bootstrap_ci['ci_lower_boot'],
        'ci_upper_boot': bootstrap_ci['ci_upper_boot']
    })
    
    # Store results
    results_dict = {
        'fixed_effects': fixed_effects,
        'random_effects': mixed_result.random_effects,
        'model_summary': mixed_result.summary(),
        'bootstrap_success': use_bootstrap
    }
    
    if if_display:
        print("\nMixed Effects Model Results:")
        print(mixed_result.summary())
        # Plot fixed effects coefficients with confidence intervals and significance
        coef_items = [(k,v) for k,v in fixed_effects['coef'].items() if k != 'Intercept']
        feature_names = [x[0] for x in coef_items]
        coefs = [x[1] for x in coef_items]
        
        plt.figure(figsize=(9,6))
        
        # Plot 95% bootstrap confidence intervals
        yerr_lower = [fixed_effects.loc[name,'coef'] - fixed_effects.loc[name,'ci_lower_boot'] for name in feature_names]
        yerr_upper = [fixed_effects.loc[name,'ci_upper_boot'] - fixed_effects.loc[name,'coef'] for name in feature_names]
        plt.errorbar(feature_names, coefs,
                    yerr=[yerr_lower, yerr_upper],
                    fmt='none', color='black', capsize=5)
        bars = plt.bar(feature_names, coefs)
        
        # Add significance markers based on corrected p-values
        for idx, name in enumerate(feature_names):
            if fixed_effects.loc[name,'qvalue'] < 0.001:
                plt.text(idx, coefs[idx], '***', ha='center', va='bottom', fontsize=18)
            elif fixed_effects.loc[name,'qvalue'] < 0.01:
                plt.text(idx, coefs[idx], '**', ha='center', va='bottom', fontsize=18)
            elif fixed_effects.loc[name,'qvalue'] < 0.05:
                plt.text(idx, coefs[idx], '*', ha='center', va='bottom', fontsize=18)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        plt.xticks(rotation=45, ha='right', fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(-0.25,0.25)
        plt.title(f'Fixed Effects Coefficients for {df.roi.iloc[0]}', fontsize=22)
        plt.xlabel('Features', fontsize=22)
        plt.ylabel('Beta', fontsize=22)
        plt.tight_layout()
        plt.show()
        
    return results_dict

def analyze_remarkable_correlations(group_correlation_df, banned_candidate_dict, candidate_rsm_dict, neural_rsm, r_threshold=0.1):
    """Analyze regression coefficients for ROIs with significant correlations
    Args:
        group_correlation_df (pd.DataFrame): DataFrame with columns [reference, candidate, module, r, p, q], as the criterion to select candidates of remarkable correlations with neural RSMs
        banned_candidate_dict (dict): Dictionary of banned candidates for each module
        candidate_rsm_dict (dict): Dictionary of candidate RSMs
        neural_rsm (dict): Dictionary of neural RSMs
        r_threshold (float): Correlation threshold for including ROIs
    Returns:
        dict: Dictionary of regression coefficients for each ROI
    """
    df = group_correlation_df.copy()
    sig_rows = df[np.abs(df.r) >= r_threshold]
    # Get union of all candidates across ROIs and exclude banned candidates
    candidate_list = {}
    total_comparisons = 0
    for module in candidate_rsm_dict.keys():
        module_rows = sig_rows[sig_rows['module'] == module]
        if not module_rows.empty:
            candidates = list(set(module_rows['candidate'].tolist()))
            # Remove banned candidates for this module
            if module in banned_candidate_dict.keys():
                candidates = [c for c in candidates if c not in banned_candidate_dict[module]]
            candidate_list[module] = candidates
            total_comparisons += len(candidates)
    
    roi_groups = sig_rows['reference'].unique()
    total_comparisons *= len(roi_groups)
    
    # run linear mixed-effects model
    results_lm = {}
    #results_lr = {}
    for roi in roi_groups:
        print("Linear mixed-effects model for ",roi)
        # Prepare regression data using the same candidate list for all ROIs
        reg_df = prepare_regression_data(roi, candidate_list, candidate_rsm_dict, neural_rsm)
        
        #results_lr[roi] = fit_regression_model(reg_df)
        results_lm[roi] = fit_mixed_model(reg_df, multiple_comparison='bonferroni', n_total_comparison=total_comparisons)
        
    return results_lr, results_lm

# %%
# Filter for FFA_l reference only
results_lr, results_lm = analyze_remarkable_correlations(average_df, {"CLIP_annotation":["fishing","cooking","sport","scene","outdoors"],"model_embedding":["resnet3d50","multi_resnet3d50","resnet50"]}, combined_rsm, roi_neural_rsm, r_threshold=0.1)





# %%
