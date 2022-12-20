"""
Created on 20/12/22

@author: dimiboeckaerts

LEARNING UTILS FOR THE RBP-R PREDICTION FRAMEWORK
"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import subprocess
import numpy as np


# 1 - FUNCTIONS
# --------------------------------------------------
def compute_hdc_representation(julia_file, loci_file, rbp_file, output_file):
    """
    Computes joint hyperdimensional representations for loci proteins and RBPs in Julia.
    
    Input:
    - julia_file: path to compute_hdc_rep.jl
    - loci_file: path to LociBase dictionary (as .json)
    - rbp_file: path to RBPbase dataframe (as .csv)
    - output_file: path for saving the resulting feature matrix (.txt)
    Output: feature matrix for machine learning
    
    Remark: first run the alias command once in therminal to enable julia from command line!
    """
    #alias_command = 'sudo ln -fs julia="/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia" /usr/local/bin/julia'
    cd_command = 'cd ' + julia_file
    pw_command = 'julia compute_hdc_rep.jl ' + loci_file + ' ' + rbp_file + ' ' + output_file
    command = cd_command + '; ' + pw_command

    ssprocess = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ssout, sserr = ssprocess.communicate()
    
    return ssout, sserr

def mean_reciprocal_rank(queries):
    """
    This function computes the mean reciprocal rank for a given array or
    matrix of queries. It deals with relevant vs. non-relevant queries that are
    binary. If queries is a matrix, then it will compute the reciprocal ranks over
    all rows individually (for each 'query') and then average those.
    E.g.:
    queries = [[0, 0, 0], [0, 1, 0], [1, 0, 0]]
    mean_reciprocal_rank(queries) -> 0.5
    """
    queries = (np.asarray(r).nonzero()[0] for r in queries)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in queries])

def recallatk(queries, k):
    """
    recall at top K for binary problems. Relevant items should be labeled as 1.
    
    Input: 
    - queries: list of lists of queries that are sorted (i.e. true labels sorted by prediction score)
    - k: the top you want to look at
    """
    recalls_k = [query[:k].count(1) / query.count(1) for query in queries]
    return np.mean(recalls_k)

def marecallatk(queries, kmax):
    """
    mean average recall @ K; computes the average recall@K for values up to kmax and then
    takes the mean over all queries. Relevant items should be labeled as 1.
    """
    average_recalls = []
    for query in queries:
        recalls = [query[:k+1].count(1) / query.count(1) for k in range(kmax)]
        average_recalls.append(np.mean(recalls))
    return np.mean(average_recalls)

def hitratio(queries, k):
    """
    hit ratio for in the first k elements (sorted queries)
    """
    return sum([1 for query in queries if sum(query[:k]) > 0]) / len(queries)

def uninorm(x, y):
    result = (x*y) / (x*y + (1-x)*(1-y))
    return result