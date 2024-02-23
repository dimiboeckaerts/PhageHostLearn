"""
LOGOCV EVALUATION - RESULTS

@author: dimiboeckaerts
@date: 2024-02-20
"""

# 0 - LIBRARIES & PATHS
# ------------------------------------------
general_path = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/Valencia_data'
results_path = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/43_RESULTS/models'
data_suffix = 'Valencia'

import numpy as np
import matplotlib.pyplot as plt
import pickle
import phagehostlearn_features as phlf
from sklearn.metrics import roc_auc_score, auc, roc_curve


# 1 - RESULTS
# ------------------------------------------
# load the data
matrix = np.loadtxt(general_path+'/all_loci_score_matrix.txt', delimiter='\t')
rbp_embeddings_path = general_path+'/esm2_embeddings_rbp'+data_suffix+'.csv'
loci_embeddings_path = general_path+'/esm2_embeddings_loci'+data_suffix+'.csv'
features_esm2, labels, groups_loci, groups_phage = phlf.construct_feature_matrices(general_path, data_suffix, loci_embeddings_path, rbp_embeddings_path)

# make the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
thresholds = [1, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
tstr = ['100', '99', '95', '90', '85', '80', '75', '70', '65', '60']
colors = plt.cm.Greens(np.linspace(0.99,0.3,len(thresholds)))
for i, thres in enumerate(thresholds):
    # make new_groups if needed
    if tstr[i] == '100':
        new_groups = groups_loci
    else:
        group_i = 0
        new_groups = [np.nan] * len(groups_loci)
        for i in range(matrix.shape[0]):
            cluster = np.where(matrix[i,:] >= thres)[0]
            oldgroups_i = [k for k, x in enumerate(groups_loci) if x in cluster]
            if np.isnan(new_groups[groups_loci.index(i)]):
                for ogi in oldgroups_i:
                    new_groups[ogi] = group_i
                group_i += 1
    
    # read results
    with open(results_path+'/v3.4/combined_logocv_results_v34_'+tstr[i]+'.pickle', 'rb') as f:
        logo_results = pickle.load(f)
    scores_lan = logo_results['scores_language']
    label_list = logo_results['labels']

    # compute performance
    rqueries_lan = []
    for i in range(len(set(new_groups))):
        score_lan = scores_lan[i]
        y_test = label_list[i]
        try:
            roc_auc = roc_auc_score(y_test, score_lan)
            ranked_lan = [x for _, x in sorted(zip(score_lan, y_test), reverse=True)]
            rqueries_lan.append(ranked_lan)
        except:
            pass

    # plot AUC
    labels = np.concatenate(label_list).ravel()
    scoreslr = np.concatenate(scores_lan).ravel()
    fpr, tpr, thrs = roc_curve(labels, scoreslr)
    rauclr = round(auc(fpr, tpr), 3)
    ax1.plot(fpr, tpr, c=colors[i], linewidth=2.5, label='LOGOCV @ '+tstr[i]+'% (AUC= '+str(rauclr)+')')

    

ax1.set_xlabel('False positive rate', size=24)
ax1.set_ylabel('True positive rate', size=24)
ax1.legend(loc=4, prop={'size': 20})
ax1.grid(True, linestyle=':')
ax1.yaxis.set_tick_params(labelsize = 14)
ax1.xaxis.set_tick_params(labelsize = 14)
fig.savefig(results_path+'/v3.4/logocv_rocauc_complete.png', dpi=400)
#fig.savefig(results_path+'/v3.4/logocv_rocauc_svg.svg', format='svg', dpi=400)



# ------------------------------------------
matrix = np.loadtxt(general_path+'/all_loci_score_matrix.txt', delimiter='\t')
np.fill_diagonal(matrix, 0) # replace the identity comparisons
upp = np.triu(matrix) # get upper triangle
upp = upp.flatten()
upp = upp[np.nonzero(upp)] # remove the zeros

# Plot the histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(upp, bins=50)
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')