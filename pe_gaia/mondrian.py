import sklearn.neighbors
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

################################## MONDRIAN CONFORMAL REGRESSION ##################################

def get_binning_prediction(n_bins, pred_cal):
    '''
    Calculate binning indices and edges based on the model’s raw prediction values. 
    It uses "equal-sized" binning so that each bin contains an approximately equal number of calibration samples.  
    -------------------------------------------------------------------------------------------------------------
    Arguments:
    n_bins -- The number of bins to create.
    pred_cal -- The predictions for the calibration set.

    Returns:
    bin_idx: An array of integers mapping each sample to its corresponding bin.
    bin_edges: The specific values that define the boundaries of each bin.
    '''
    rng = np.random.RandomState(1234)
    jitter = rng.uniform(-1e-10, 1e-10, size=pred_cal.shape)
    pred_jittered = pred_cal + jitter

    # compute edges for equal-sized bins on the predicion values
    bin_edges = np.quantile(pred_jittered, q=np.linspace(0, 1, n_bins+1), axis=0)
    bin_idx = [np.digitize(pred_jittered[:, j], bin_edges[1:-1][:, j], right=False) 
               for j in range(pred_jittered.shape[1])]
    bin_idx = np.stack(bin_idx, axis=1)

    return bin_idx, bin_edges

def get_features(data, model):
    '''
    Extract features from CNN penultimate layer.
    ----------------------------------------------

    Arguments:
    cal_ds -- The input data to process.
    model -- The trained model from which to extract features.

    Returns:
    features -- The activation values from the layer before the output.
    '''

    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output  
    )

    return feature_extractor.predict(data)

def compute_sigmas(X_feat, residuals, k=5):
    '''
    Estimate the difficulty for each sample using a k-Nearest Neighbors approach in the feature space. 
    It calculates a distance-weighted average of the residuals of the nearest neighbors.
    ---------------------------------------------------------------------------------------------------
    Arguments:
    X_feat -- The extracted features from the penultimate layer.
    residuals -- The calibration errors.
    k -- Number of neighbors to consider (default is 5).
    
    Return:
    sigmas -- A vector of difficulty scores used to scale the prediction intervals.
    '''

    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    knn.fit(X_feat)

    distances, indices = knn.kneighbors(X_feat)
    distances += 1e-10
    distances = distances[:,1:] # drop self
    distances = distances[..., np.newaxis] # expand to broadcast later 
    indices = indices[:, 1:] # drop self 
    
    neigh_residuals = residuals[indices]
    
    # mean of the residual errors of the k nearest neighbors for each element  
    sigmas = np.sum(neigh_residuals/distances, axis=1) / np.sum(1/distances, axis=1)  

    return sigmas

def compute_sigma_inference(X, model, X_cal_feat, residuals_cal):
    
    X_feat = get_features(X, model)
    
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(X_cal_feat)
    
    distances, indices = knn.kneighbors(X_feat)
    distances += 1e-10
    distances = distances[..., np.newaxis]

    neigh_residuals = residuals_cal[indices]

    weights = 1.0 / distances
    sigmas = np.sum(neigh_residuals * weights, axis=1) / np.sum(weights, axis=1)
    
    return sigmas

def get_binning_quality(n_bins, residuals, cal_ds, model):
    '''
    Calculate binning indices based on a difficulty estimate. 
    ---------------------------------------------------------------
    Arguments:
    n_bins -- The number of bins to create.
    residuals -- The residuals of the calibration set.
    cal_ds -- The calibration input data.
    model -- The trained neural network.
    
    Returns:
    bin_idx: An array of integers mapping each sample to its corresponding bin.
    bin_edges: The specific values that define the boundaries of each bin.
    '''
    X_cal_feat = get_features(cal_ds, model)
    sigmas = compute_sigmas(X_cal_feat, residuals)

    # compute edges for equal-sized bins on the predicion values
    bin_edges = np.quantile(sigmas, q=np.linspace(0, 1, n_bins+1), axis=0)
    bin_idx = [np.digitize(sigmas[:, j], bin_edges[1:-1][:, j], right=False) 
               for j in range(sigmas.shape[1])]
    bin_idx = np.stack(bin_idx, axis=1)
    
    return bin_idx, bin_edges, X_cal_feat 


def get_binned_array(array, bin_idx, n_bins=12):
    '''
    Reorganize a flat array into a nested structure where data is grouped by its assigned bin index.
    -----------------------------------------------------------------------------------------------
    Arguments:

    array -- The data to be partitioned (e.g., residuals).
    bin_idx -- The indices mapping data to bins.
    n_bins -- The total number of bins.

    Returns:
    binned_array -- A list of arrays, where each sub-array contains only the samples belonging to a specific bin.
    '''
    binned_array = []
    
    for j in range(array.shape[1]): # itero sui labels 
        arr_j = []
        for i in range(n_bins): # itero sui bin 
            mask = (bin_idx[:, j] == i)
            arr_j_bin_i = array[mask, j] # residuals del bin i per il label j 
            arr_j.append(arr_j_bin_i)
            
        binned_array.append(arr_j)
            
    return np.array(binned_array, dtype=object)

def prediction_interval(conf_level, residuals, mode):
    '''
    Compute the statistical credibility intervals for each bin based on a desired confidence level. 
    -----------------------------------------------------------------------------------------------
    Arguments:
    conf_level -- The desired confidence percentage (e.g., 95).
    residuals -- The binned residual errors.
    mode -- The interval type ('symm', 'asymm', or 'asymm_minmax').
    
    Returns:
    credibility_intervals -- An array of tuples containing the (low, high) bounds for each bin and each label.
    '''

    epsilon = 1-conf_level/100
    
    credibility_intervals = []
    
    for j in range(residuals.shape[0]):
        cred_j = []
        for i in range(residuals.shape[1]):
            alphas = np.abs(residuals[j,i])
            # select the nonconformity score corresponding to the 95% of the samples 
            thresh = np.percentile(alphas, conf_level)

            if mode == 'symm':
                ordered_alphas = np.sort(alphas)[::-1]
                s = int(np.floor(epsilon * (len(ordered_alphas) + 1)))
                low = -float(ordered_alphas[s])
                cred_j.append((low,-low))

            if mode == 'asymm':
                res = residuals[j,i]
                ordered_res = np.sort(res)[::-1]
                h = int(np.floor(epsilon/2 * (len(ordered_res) + 1))) 
                l = int(np.floor((1-epsilon/2) * (len(ordered_res) + 1)))
                low = float(ordered_res[l])
                high = float(ordered_res[h])
                cred_j.append((low, high))

            if mode == 'asymm_minmax':
                central_residuals = residuals[j,i][residuals[j,i] <= thresh]
                cred_j.append((np.min(central_residuals),np.max(central_residuals)))
        
        credibility_intervals.append(cred_j)
        
        
    return np.array(credibility_intervals)

def plot_mondrian_coverage(categories, pred_test, y_test, credibility_intervals, title, colors, target_coverage=0.90):
    """
    This function plots the empirical coverage for Mondrian Conformal Regression bins.
    """
    
    parameter_names = [r'$\mathcal{M}$', r'$M_{tot}$', r'$\chi_{eff}$']
    n_params = pred_test.shape[1]
    n_bins = int(np.max(categories) + 1)
    
    pred_test_low = np.zeros_like(pred_test)
    pred_test_high = np.zeros_like(pred_test)
    
    for j in range(n_params):
        pred_test_low[:, j] = pred_test[:, j] + credibility_intervals[j, categories[:, j], 0]
        pred_test_high[:, j] = pred_test[:, j] + credibility_intervals[j, categories[:, j], 1]

    coverage = (y_test >= pred_test_low) & (y_test <= pred_test_high)
    
    print(f"\nConfiguration: {title}")
    print("-" * 45)
    
    real_coverage = np.zeros((n_bins, n_params))
    
    for j in range(n_params):
        avg_error = (1 - np.mean(coverage[:, j]))
        print(fr"{parameter_names[j]:<25} | Average Error Rate: {avg_error:.3f}")
        
        for b in range(n_bins):
            mask_bin = (categories[:, j] == b)
            if np.any(mask_bin):
                real_coverage[b, j] = np.mean(coverage[mask_bin, j])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(n_bins)
    width = 0.25

    for j in range(n_params):
        offset = (j - 1) * width
        ax.bar(x + offset, real_coverage[:, j], width, 
               label=parameter_names[j], color=colors[j], 
               edgecolor='#4D4D4D', linewidth=1, zorder=3)

    ax.axhline(y=target_coverage, color='red', linestyle='--', 
               label=f'Target ({int(target_coverage*100)}%)', linewidth=2, zorder=5)

    ax.set_xlabel('Category Index', fontsize=25, labelpad=15)
    ax.set_ylabel('Empirical Coverage', fontsize=25, labelpad=15)
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.set_xticks(x)
    ax.set_ylim(0, 1.18)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
              ncol=4, frameon=False, prop={'size': 20}, columnspacing=1.2)

    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()

############################################ MONDRIAN CONFORMAL PREDICTION ############################################ 

# # step 3
# bin_idx, bin_edges = get_binning_prediction(n_bins, pred_cal)

# # step 4
# binned_pred = get_binned_array(predictions, bin_idx)
# binned_residuals = get_binned_array(residuals, bin_idx)

# X_cal_feat = get_features(cal_ds, model)
# sigmas = compute_sigmas(X_feat, residuals, k=5)
# binned_sigmas = get_binned_array(sigmas, bin_idx)

# # step 5
# y = scaler.inverse_transform(model.predict(data))[0]
# feat_data = get_features(data, model)
# sigma = compute_sigmas(feat_data, residuals, k=5)

# # step 6
# category = int(np.digitize(y, bin_edges[1:-1], right=False))
# residuals_bin = binned_residuals[category]
# sigmas_bin = binned_sigmas[category]

# calib_scores = get_calibration_scores(y, sigma, residuals_bin, sigmas_bin)

# def get_calibration_scores(y, sigma, residuals_bin, sigmas_bin):
#     calib_scores = []
#     for i in range(len(residuals_bin)):
#         score = y + (sigma / sigmas_bin[i]) * (residuals_bin[i])
#         calib_scores.append(score)

#     return calib_scores

# class MondrianCPD:

#     def __init__(self, calibration_scores, tau):
        
#         self.q = len(calibration_scores)

#         ### STEP 7
#         # sort the calibration scores in increasing order 
#         self.C = np.sort(np.array(calibration_scores).flatten())
        
#         ### STEP 8
#         # set the first and last values 
#         self.C_extended = np.concatenate([[-np.inf], self.C, [np.inf]])

#         ### STEP 9 
#         # let tau be unormly sampled in [0,1]
#         self.tau = tau
    
#     def Q(self, y):
#         """
#         Calculate cumulative probability Q(y)
#         """
#         n = np.searchsorted(self.C, y, side='left')  # n is the index where to insert y
        
#         # f y = C_j(n) for some n 
#         if n < self.q and np.isclose(y, self.C[n]):
#             n_prime = np.where(np.isclose(self.C, y))[0][0] 
#             n_double_prime = np.where(np.isclose(self.C, y))[0][-1]
            
#             numerator = n_prime - 1 + (n_double_prime - n_prime + 2) * self.tau
#             return numerator / (self.q + 1)
        
#         # IF C_j(n) < y < C_j(n+1)
#         else:
#             return (n + self.tau) / (self.q + 1)
    
#     def get_prediction_interval(self, alpha= 0.05):
#         """
#         Obtain the confidence interval for a certain confidence level
#         """
#         lower_percentile = alpha / 2
#         upper_percentile = 1 - alpha / 2
        
#         lower_bound = self.find_quantile(lower_percentile, 'lower')
#         upper_bound = self.find_quantile(upper_percentile, 'upper')
        
#         return lower_bound, upper_bound
    
#     def find_quantile(self, p, bound_type):
#         """
#         Find the quantile corresponding to the probability p 
#         """
#         if bound_type == 'lower':
#             # max{m | Q(C_m) < p}
#             for i in range(self.q - 1, -1, -1):
#                 if self.Q(self.C[i]) < p:
#                     return self.C[i]
#             return self.C_extended[0]  # -inf
#         else:  # upper
#             # min{m | Q(C_m) > p}
#             for i in range(self.q):
#                 if self.Q(self.C[i]) > p:
#                     return self.C[i]
#             return self.C_extended[-1]  # inf
        
# def plot_distribution(calibration_scores, median, true_M, cpd):

#     n_points=1000
#     calibration_scores = np.array(calibration_scores)
    
#     y_values = np.linspace(calibration_scores.min(), calibration_scores.max(), n_points)
#     q_values = [cpd.Q(y) for y in y_values]
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(y_values, q_values, color='blue', linewidth=0.5)
    
#     # Intervallo di predizione al 95%
#     lower, upper = cpd.get_prediction_interval(0.05)
#     plt.axvline(lower, color='olivedrab', linestyle='--', linewidth=0.5, label='95% CI')
#     plt.axvline(upper, color='olivedrab', linestyle='--', linewidth=0.5)
    
#     # Mediana
#     plt.axvline(median, color='olivedrab', linestyle='-', linewidth=1, label='Median')
#     plt.axvline(true_M, color='red', linestyle='-', linewidth=1, label='LVK estimate')
    
#     plt.title('CDF - Mondrian Conformal Predictive Distributions')
#     plt.xlabel('Chirp Mass value', fontsize=12)
#     plt.ylabel('Cumulative probability', fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()