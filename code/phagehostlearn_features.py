"""
Created on 04/01/23

@author: dimiboeckaerts

PhageHostLearn FEATURE CONSTRUCTION
"""

# 0 - LIBRARIES
# --------------------------------------------------
import ast
import math
import json
import subprocess
import numpy as np
import pandas as pd
import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm


# 1 - FUNCTIONS
# --------------------------------------------------
def compute_esm2_embeddings_rbp(general_path, data_suffix=''):
    """
    This function computes ESM-2 embeddings for the RBPs, from the RBPbase.csv file.

    INPUTS:
    - general path to the project data folder
    - data suffix to optionally add to the saved file name (default='')
    OUTPUT: esm2_embeddings_rbp.csv
    """
    # load the ESM2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # loop over data and embed (batch size = 1)
    sequence_representations = []
    RBPbase = pd.read_csv(general_path+'/RBPbase'+data_suffix+'.csv')
    bar = tqdm(total=len(RBPbase['protein_sequence']), position=0, leave=True)
    for i, sequence in enumerate(RBPbase['protein_sequence']):
        data = [(RBPbase['protein_ID'][i], sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        for j, (_, seq) in enumerate(data):
            sequence_representations.append(token_representations[j, 1 : len(seq) + 1].mean(0))
        bar.update(1)
    bar.close()

    # save results
    ids = RBPbase['protein_ID']
    embeddings_df = pd.concat([pd.DataFrame({'protein_ID':ids}), pd.DataFrame(sequence_representations).astype('float')], axis=1)
    embeddings_df.to_csv('esm2_embeddings_rbp'+data_suffix+'.csv', index=False)
    return


def compute_esm2_embeddings_loci(general_path, data_suffix=''):
    """
    This function computes ESM-2 embeddings for the loci proteins, from the Locibase.json file.

    INPUTS:
    - general path to the project data folder
    - data suffix to optionally add to the saved file name (default='')
    OUTPUT: esm2_embeddings_loci.csv
    """
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Load json file
    dict_file = open(general_path+'/Locibase'+data_suffix+'.json')
    loci_dict = json.load(dict_file)

    # loop over data and embed (batch size = 1)
    loci_representations = []
    for key in tqdm(loci_dict.keys()):
        embeddings = []
        for sequence in loci_dict[key]:
            data = [(key, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            for i, (_, seq) in enumerate(data):
                embeddings.append(token_representations[i, 1 : len(seq) + 1].mean(0))
        locus_embedding = np.mean(np.vstack(embeddings), axis=0)
        loci_representations.append(locus_embedding)

    # save results
    embeddings_df = pd.concat([pd.DataFrame({'accession':list(loci_dict.keys())}), pd.DataFrame(loci_representations)], axis=1)
    embeddings_df.to_csv('esm2_embeddings_loci'+data_suffix+'.csv', index=False)


def compute_hdc_embedding(general_path, data_suffix=''):
    """
    Computes joint hyperdimensional representations for loci proteins and RBPs in Julia.
    
    INPUTS:
    - general path to the project data folder
    - data suffix to optionally add to the saved file name (default='')
    OUTPUT: hdc_features.txt
    REMARK: first run the alias command once in therminal to enable julia from command line!
    """
    #alias_command = 'sudo ln -fs julia="/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia" /usr/local/bin/julia'
    loci_file = general_path+'/Locibase'+data_suffix+'.json'
    rbp_file = general_path+'/RBPbase'+data_suffix+'.csv'
    output_file = general_path+'/hdc_features'+data_suffix+'.txt'
    command = 'julia compute_hdc_rep.jl ' + loci_file + ' ' + rbp_file + ' ' + output_file
    ssprocess = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ssout, sserr = ssprocess.communicate()
    return ssout, sserr


def construct_feature_matrices(general_path, data_suffix=''):
    """
    This function constructs two corresponding feature matrices ready for machine learning, 
    starting from the ESM-2 embeddings & the HDC embeddings of RBPs and loci proteins.

    INPUTS:
    - general path to the project data folder
    - data suffix to optionally add to the saved file name (default='')
    OUTPUT: features_esm2, features_hdc, labels, groups_loci, groups_phage
    """
    RBP_embeddings = pd.read_csv(general_path+'/esm2_embeddings_rbp'+data_suffix+'.csv')
    loci_embeddings = pd.read_csv(general_path+'/esm2_embeddings_loci'+data_suffix+'.csv')
    interactions = pd.read_csv(general_path+'/phage_host_interactions'+data_suffix+'.csv', index_col=0)
    hdc_embeddings = pd.read_csv(general_path+'/hdc_features'+data_suffix+'.txt', sep="\t", header=None)
    pairs = [ast.literal_eval(i) for i in hdc_embeddings.iloc[:,0]]

    # construct multi-RBP representations
    multi_embeddings = []
    names = []
    for phage_id in list(set(RBP_embeddings['phage_ID'])):
        rbp_embeddings = RBP_embeddings.iloc[:,2:][RBP_embeddings['phage_ID'] == phage_id]
        multi_embedding = np.mean(np.asarray(rbp_embeddings), axis=0)
        names.append(phage_id)
        multi_embeddings.append(multi_embedding)
    multiRBP_embeddings = pd.concat([pd.DataFrame({'phage_ID': names}), pd.DataFrame(multi_embeddings)], axis=1)

    # construct dataframe for training
    features_lan = []
    features_hdc = []
    labels = []
    groups_loci = []
    groups_phage = []

    for i, accession in enumerate(loci_embeddings['accession']):
        for j, phage_id in enumerate(multiRBP_embeddings['phage_ID']):
            interaction = interactions.loc[accession][phage_id]
            if math.isnan(interaction) == False: # if the interaction is known
                # language embeddings
                features_lan.append(pd.concat([loci_embeddings.iloc[i, 1:], multiRBP_embeddings.iloc[j, 1:]]))
                
                # hdc embeddings reorder
                pair = (accession, phage_id)
                this_index = pairs.index(pair)
                features_hdc.append(hdc_embeddings.iloc[this_index, 1:])
                
                # append labels and groups
                labels.append(int(interaction))
                groups_loci.append(i)
                groups_phage.append(j)
                
    features_lan = np.asarray(features_lan)
    features_hdc = np.asarray(features_hdc)
    print("Dimensions match?", features_lan.shape[1] == (loci_embeddings.shape[1]+multiRBP_embeddings.shape[1]-2))
    print("Dimensions match?", features_lan.shape[0] == features_hdc.shape[0])

    #np.save(general_path+'/esm2_features'+data_suffix+'.txt', features_lan)
    return features_lan, features_hdc, labels, groups_loci, groups_phage
