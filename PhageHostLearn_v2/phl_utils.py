"""
PhageHostLearn - utils

@author: dimiboeckaerts
@date: 2024-01-31
"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import esm
import json
import torch
import pickle
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold

# 1 - FUNCTIONS
# --------------------------------------------------
def phanotate_py(files, path, phanotate_path, suffix=''):
    """
    Computes PHANOTATE for every phage genome in the phages_path folder.
    Inputs:
    - files: list of strings of names of FASTA files in phage_genomes
    - path: path to the data folder (which contains phage_genomes subfolder)
    - phanotate_path: path to the phanotate.py file
    - suffix: suffix to add to the output files
    Outputs: phage_genes.csv
    Remark: make sure that files are not named with '|', this causes issues with Phanotate.
    """
    name_list = []; gene_list = []; gene_ids = []

    print('Running Phanotate')
    for file in tqdm(files):
        # run PHANOTATE
        count = 1
        file_path = path+'/phage_genomes/'+file
        command = phanotate_path + ' ' + file_path
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        std_splits = stdout.split(sep=b'\n')
        std_splits = std_splits[2:] #std_splits.pop(0)
        # Save and reload TSV
        temp_tab = open(path+'/phage_results'+suffix+'.tsv', 'wb')
        for split in std_splits:
            split = split.replace(b',', b'') # replace commas for pandas compatibility
            temp_tab.write(split + b'\n')
        temp_tab.close()
        results_orfs = pd.read_csv(path+'/phage_results'+suffix+'.tsv', sep='\t', lineterminator='\n', index_col=False)
        
        # fill up lists accordingly
        name = file.split('.fna')[0].split('.fasta')[0]
        sequence = str(SeqIO.read(file_path, 'fasta').seq)
        for j, strand in enumerate(results_orfs['FRAME']):
            start = results_orfs['#START'][j]
            stop = results_orfs['STOP'][j]
            if strand == '+':
                gene = sequence[start-1:stop]
            else:
                sequence_part = sequence[stop-1:start]
                gene = str(Seq(sequence_part).reverse_complement())
            name_list.append(name)
            gene_list.append(gene)
            gene_ids.append(name+'_gp'+str(count))
            count = count + 1
    os.remove(path+'/phage_results'+suffix+'.tsv')

    # Export final genes database
    genebase = pd.DataFrame(list(zip(name_list, gene_ids, gene_list)), columns=['phage_ID', 'gene_ID', 'gene_sequence'])
    genebase.to_csv(path+'/phage_genes'+suffix+'.csv', index=False)
    return

def phagerbpdetect(path, model_path, suffix=''):
    """
    This function runs the RBPdetect v3 model on the phage proteins
    Inputs:
    - path: the path to the folder containing the phage proteins
    Output: rbps.csv 
    """
    # initiation the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if device == 'cuda':
        model.eval().cuda()
    else:
        model.eval()

    # make predictions
    genes = pd.read_csv(path+'/phage_genes'+suffix+'.csv')
    sequences = [str(Seq(gene).translate()[:-1]) for gene in genes['gene_sequence']]
    phagenames = [gid for gid in genes['phage_ID']]
    genenames = [gid for gid in genes['gene_ID']]
    predictions = []
    scores = []
    for sequence in tqdm(sequences):
        encoding = tokenizer(sequence, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            output = model(**encoding)
            predictions.append(int(output.logits.argmax(-1)))
            scores.append(float(output.logits.softmax(-1)[:, 1]))
    
    # save the results
    results = pd.DataFrame({'accession': phagenames, 'protein_ID': genenames, 'protein_sequence': sequences,
                             'pred': predictions, 'score': scores})
    detected_rbps = results[results['score'] > 0.5]
    detected_rbps.to_csv(path+'/rbps'+suffix+'.csv', index=False)
    return

def process_phages(path, phanotate_path, model_path, suffix=''):
    """
    Process the phage genomes with Phanotate and PhageRBPdetect v3.
    Outputs: phage_genes.csv, rbps.csv
    """
    # collect the files
    files = [file for file in os.listdir(path+'/phage_genomes') if (file.endswith('.fasta')) or (file.endswith('.fna'))]
    # run Phanotate
    phanotate_py(files, path, phanotate_path, suffix)
    # run PhageRBPdetect v3
    phagerbpdetect(path, model_path, suffix)
    return

def kaptive_py(files, path, kaptive_path, db_path, suffix=''):
    """
    Computes Kaptive for each FASTA in fastas list (of strings).
    """
    accessions = [file.split('.fna')[0].split('.fasta')[0] for file in files]
    accessions_list = []; sero_list = []; proteins_list = []; loci_names_list = []
    loci_list = []; confidence_list = []
    print('Running Kaptive')
    pbar = tqdm(total=len(files))
    for i, file in enumerate(files):
        # run kaptive
        file_path = path+'/bacterial_genomes/'+file
        #mkcommand = 'mkdir '+path+'/kaptive_results'+data_suffix
        command = 'python '+kaptive_path+' -a '+file_path+' -k '+db_path+' -o '+path+'/results'+suffix+' --no_table'
        #command = mkcommand+'; '+kapcommand
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, err = process.communicate()
                
        # process json -> proteins in dictionary
        results = json.load(open(path+'/results'+suffix+'.json'))
        serotype = results[0]['Best match']['Type']
        match_confidence = results[0]['Best match']['Match confidence']
        count = 0
        if (match_confidence != 'Low') or (match_confidence != 'None'):
            for gene in results[0]['Locus genes']:
                try:
                    protein = gene['tblastn result']['Protein sequence']
                    protein = protein.replace('-', '')
                    protein = protein.replace('*', '')
                except KeyError:
                    protein = gene['Reference']['Protein sequence']
                    protein = protein.replace('-', '')
                    protein = protein.replace('*', '')
                accessions_list.append(accessions[i])
                loci_names_list.append(accessions[i]+'_'+str(count))
                proteins_list.append(protein)
                count += 1
        loci_sequence = ''
        for record in SeqIO.parse(path+'/results'+suffix+'_'+accessions[i]+'.fasta', 'fasta'):
            loci_sequence = loci_sequence + str(record.seq)
        if len(loci_sequence) > 100000:
            print('Warning: extremely long locus sequence, please check manually:', accessions[i])
        loci_list.append(loci_sequence)
        sero_list.append(serotype)
        confidence_list.append(match_confidence)

        # delete temp kaptive files
        os.remove(file_path+'.ndb')
        os.remove(file_path+'.not')
        os.remove(file_path+'.ntf')
        os.remove(file_path+'.nto')
        os.remove(path+'/results'+suffix+'.json')
        os.remove(path+'/results'+suffix+'_'+accessions[i]+'.fasta')

        # update progress
        pbar.update(1)
    pbar.close()

    # Export final databases
    kaptive_df = pd.DataFrame({'accession': accessions_list, 'protein_ID': loci_names_list, 'protein_sequence': proteins_list})
    kaptive_df.to_csv(path+'/locis'+suffix+'.csv', index=False)
    loci_df = pd.DataFrame({'accession': accessions, 'KL-type': sero_list, 'confidence': confidence_list, 'locus_sequence': loci_list})
    loci_df.to_csv(path+'/loci_summary'+suffix+'.csv', index=False)
    return

def process_bacteria(path, kaptive_path, db_path, suffix=''):
    """
    Process the bacterial genomes with Kaptive.
    Outputs: locis.csv, loci_summary.csv
    """
    # collect the files
    files = [file for file in os.listdir(path+'/bacterial_genomes') if (file.endswith('.fasta')) or (file.endswith('.fna'))]
    # run Kaptive
    kaptive_py(files, path, kaptive_path, db_path, suffix)
    return

def compute_representations(file, path, suffix=''):
    """
    This function computes ESM-2 embeddings for the proteins in a given file, 
    and averages the embeddings for each accession.

    INPUTS:
    - file: path to the .csv file with three columns: accession, protein_ID and protein_sequence
    - path: path to the general data folder
    - data suffix to optionally add to the saved file name (default='')
    OUTPUT: representations.csv
    """
    # load the ESM2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        model.eval().cuda()
    else:
        model.eval()

    # loop over data and embed (batch size = 1)
    proteins = pd.read_csv(path+'/'+file)
    bar = tqdm(total=len(proteins['protein_sequence']), position=0, leave=True)
    sequence_representations = []
    for i, sequence in enumerate(proteins['protein_sequence']):
        data = [(proteins['protein_ID'][i], sequence[:2000])]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        if device == 'cuda':
            batch_tokens = batch_tokens.to(device=device, non_blocking=True)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        for j, (_, seq) in enumerate(data):
            sequence_representations.append(token_representations[j, 1 : len(seq) + 1].mean(0))
        bar.update(1)
    bar.close()

    # save results
    aids = proteins['accession']
    pids = proteins['protein_ID']
    embeddings_df = pd.concat([pd.DataFrame({'accession':aids}), pd.DataFrame({'protein_ID':pids}), pd.DataFrame(sequence_representations).astype('float')], axis=1)
    embeddings_df.to_csv(path+'/representations'+suffix+'.csv', index=False)
    return

def combine_representations(file, path, suffix='', mode='mean'):
    """
    This function combines the computed ESM-2 embeddings for the proteins in a given file,
    according to their accessions, and averages the embeddings for each accession.

    TO DO: implement min-max mode

    INPUTS:
    - file: path to the represenations.csv file with columns: accession, protein_ID and the embeddings
    - path: path to the general data folder
    - data suffix to optionally add to the saved file name (default='')
    OUTPUT: multi_representations.csv
    """
    reps = pd.read_csv(path+'/'+file)
    average_representations = []
    # loop over the unique accessions
    for acc in tqdm(list(set(reps['accession']))):
        # get the subset of reps for each accession
        acc_reps = reps.iloc[:,2:][reps['accession'] == acc]
        if mode == 'mean':
            #  average the reps
            average_representations.append(np.mean(np.asarray(acc_reps), axis=0))
    multirep = pd.concat([pd.DataFrame({'accession': list(set(reps['accession']))}), pd.DataFrame(average_representations)], axis=1)
    multirep.to_csv(path+'/multi_representations'+suffix+'.csv', index=False)

def construct_inputframe(rbp_multirep, loci_multirep, path, interactions=None):
    """
    This function constructs the input dataframe for the machine learning model.

    INPUTS:
    - rbp_multirep: path to the multi_representations.csv file for the RBPs
    - loci_multirep: path to the multi_representations.csv file for the loci
    - path: path to the general data folder
    - data suffix to optionally add to the saved file name (default='')
    - interactions: optional interactions dataframe if in training mode (columns: host, phage, interaction)

    OUTPUTS: features, labels
    """
    features = []
    phagerep = pd.read_csv(path+'/'+rbp_multirep)
    hostrep = pd.read_csv(path+'/'+loci_multirep)
    if interactions is not None:
        bar = tqdm(total=len(interactions['host']), position=0)
        for i, host in enumerate(interactions['host']):
            phage = list(interactions['phage'])[i]
            this_phagerep = np.asarray(phagerep.iloc[:, 1:][phagerep['accession'] == phage])
            this_hostrep = np.asarray(hostrep.iloc[:, 1:][hostrep['accession'] == host])
            features.append(np.concatenate([this_hostrep, this_phagerep], axis=1)) # first host rep, then phage rep!
            bar.update(1)
        bar.close()
    else:
        host_acc = []
        phage_acc = []
        bar = tqdm(total=len(hostrep['accession']), position=0)
        for i, host in enumerate(hostrep['accession']):
            for j, phage in enumerate(phagerep['accession']):
                features.append(pd.concat([hostrep.iloc[i, 1:], phagerep.iloc[j, 1:]], axis=1))
                host_acc.append(host)
                phage_acc.append(phage)
            bar.update(1)
        bar.close()
        interactions = pd.DataFrame({'host': host_acc, 'phage': phage_acc})
    features = np.vstack(features)
    return features, interactions

def train_model(features, interactions, path, suffix='', checkpoint=None):
    """
    This function trains a (XGBoost) PhageHostLearn model on the input features and labels.
    It can either train from scratch or continue from a checkpoint. If training from scratch, hyperparameter tuning is performed.
    Info: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/#h-what-are-xgboost-parameters

    INPUTS:
    - features: the input features for the model (output of construct_inputframe)
    - interactions: the dataframe of interactions, including labels for the model (output of construct_inputframe)
    - path: path to the general data folder
    - data suffix to optionally add to the saved file name (default='')
    - checkpoint: an optional path to trained model to continue training from (e.g. active learning setting)
    """
    # general config
    cpus = max(os.cpu_count()-2, 1)
    labels = list(interactions['label'])
    #imbalance = sum([1 for i in labels if i==1]) / sum([1 for i in labels if i==0])

    # do training
    if checkpoint is not None: # continue from checkpoint
        print('Continuing training from checkpoint...')
        xgb = XGBClassifier()
        xgb.fit(X=features, y=labels, xgb_model=path+'/'+checkpoint)
        xgb.save_model(path+'/phagehostlearn'+suffix+'.json')
    else: # train from scratch, do hyperparameter tuning
        print('Training from scratch...')
        params_xgb = {'max_depth': [3, 5, 7], 'n_estimators': [200, 300, 400], 'min_child_weight': [1, 2, 3, 4],
              'learning_rate': [0.05, 0.1, 0.2]}
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        xgb = XGBClassifier(max_delta_step=4, n_jobs=cpus, eval_metric='logloss', use_label_encoder=False)
        tuned_xgb = GridSearchCV(xgb, cv=cv, param_grid=params_xgb, scoring='roc_auc', verbose=3)
        tuned_xgb.fit(X=features, y=labels)
        best_xgb = tuned_xgb.best_estimator_
        best_xgb.save_model(path+'/phagehostlearn'+suffix+'.json')
        
    print('Training completed. Model saved. Best hyperparameters: ', tuned_xgb.best_params_)
    return

def eval_model(features, interactions, groups, path, suffix='', noutcv=40):
    """
    This function trains a (XGBoost) PhageHostLearn model on the input features and labels
    and evaluates that model on the same dataset using a nested cross-validation. This function
    is to be used if one wants to train and eval on the same dataset in a correct way. The idea
    is to loop over the datapoints in such a way that each point at least gets predicted for once.
    This function does not save a final model, but saves prediction results. To save a model, use train_model.
    Downside: it is computationally expensive.

    INPUTS:
    - features: the input features for the model (output of construct_inputframe)
    - interactions: the dataframe of interactions, including labels for the model (output of construct_inputframe)
    - groups: the groups for the LeaveOneGroupOut cross-validation
    - path: path to save the results to
    - data suffix to optionally add to the saved file name (default='')
    """
    # general config
    cpus = max(os.cpu_count()-1, 1)
    labels = np.array(interactions['label'])
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True)
    outer_cv = GroupKFold(n_splits=noutcv) #LeaveOneGroupOut()
    params_xgb = {'max_depth': [3, 5, 7], 'n_estimators': [200, 300, 400], 'min_child_weight': [2, 4],
              'learning_rate': [0.05, 0.1, 0.2]}
    scores = []; label_list = []; test_indices = []
    interactions.reset_index(drop=True, inplace=True)

    # outer loop
    pbar = tqdm(total=len(set(groups)))
    for train_i, test_i in outer_cv.split(features, labels, groups):
        # get the training and test data
        X_train, X_test = features[train_i], features[test_i]
        y_train, y_test = labels[train_i], labels[test_i]
        #imbalance = sum([1 for i in y_train if i==1]) / sum([1 for i in y_train if i==0])

        # inner loop for hyperparameter tuning
        xgb = XGBClassifier(max_delta_step=4, n_jobs=cpus, eval_metric='logloss', use_label_encoder=False)
        tuned_xgb = GridSearchCV(xgb, cv=inner_cv, param_grid=params_xgb, scoring='roc_auc', verbose=3)
        tuned_xgb.fit(X=X_train, y=y_train)
        best_xgb = tuned_xgb.best_estimator_

        # evaluate the model
        scores_xgb = best_xgb.predict_proba(X_test)[:,1]
        scores.append(scores_xgb)
        label_list.append(y_test)
        test_indices.append(test_i)
        pbar.update(1)
    pbar.close()

    # save results
    travel = list(np.concatenate(test_indices).ravel())
    sravel = list(np.concatenate(scores).ravel())
    sorted_scores = [s for _, s in sorted(zip(travel, sravel))]
    results = pd.concat([interactions, pd.DataFrame({'prediction_score': sorted_scores})], axis=1)
    results.to_csv(path+'/cvresults'+suffix+'.csv', index=False)
    return

def hitratio(queries, k):
    """
    Hit ratio for in the first k elements.
    Queries is a sorted list of lists that groups the labels of all phages per host.
    """
    return sum([1 for query in queries if sum(query[:k]) > 0]) / len(queries)

def eval_report(predictions, results_path, suffix=''):
    """
    Computes the hit ratio @ K curve, the ROC AUC and the PR AUC for the predictions.
    """
    labels = predictions['label']
    scores = predictions['prediction_score']

    # Hit ratio @ k
    queries = []
    for host in set(predictions['host']):
        subpreds = predictions[predictions['host']==host]
        if sum(subpreds['label']) > 0:
            sublabels = subpreds['label']
            subscores = subpreds['prediction_score']
            sortpreds = [label for _, label in sorted(zip(subscores, sublabels), reverse=True)]
            queries.append(sortpreds)
    ks = np.linspace(1, 30, 30)
    hits = [hitratio(queries, int(k)) for k in ks]

    # ROC AUC
    fpr, tpr, thrs = roc_curve(labels, scores)
    rocauc = round(auc(fpr, tpr), 3)

    # PR AUC
    precision, recall, thrs = precision_recall_curve(labels, scores)
    prauc = round(auc(recall, precision), 3)

    # make plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,8))
    ax1.plot(fpr, tpr, c='#124559', linewidth=2.5)
    ax1.set_xlabel('False positive rate', size=20)
    ax1.set_ylabel('True positive rate', size=20)
    ax1.grid(True, linestyle=':')
    #ax1.yaxis.set_tick_params(labelsize = 14); ax1.xaxis.set_tick_params(labelsize = 14)
    ax1.set_title('ROC curve (AUC = '+str(rocauc)+')', size=20)
    # ---
    ax2.plot(recall, precision, c='#124559', linewidth=2.5)
    ax2.set_xlabel('Recall', size=20)
    ax2.set_ylabel('Precision', size=20)
    ax2.grid(True, linestyle=':')
    #ax2.yaxis.set_tick_params(labelsize = 14); ax2.xaxis.set_tick_params(labelsize = 14)
    ax2.set_title('PR curve (AUC = '+str(prauc)+')', size=20)
    # ---
    ax3.plot(ks, hits, c='#124559', linewidth=2.5)
    ax3.set_xlabel('k', size=20)
    ax3.set_ylabel('Hit ratio @ k', size=20)
    ax3.grid(True, linestyle=':')
    ax3.set_title('Hit ratio @ k', size=20)
    ax3.set_ylim(0.05, 1.05)
    fig.savefig(results_path+'/performance_curves'+suffix+'.pdf', dpi=500)
    return

def predict_interactions(features, interactions, model_file, path, suffix='', mode='scores'):
    """
    This function make predictions for the input data using a trained PhageHostLearn (XGBoost) model.
    As an output, it both returns the interactions dataframe with the predictions, and saves it to a csv file.
    If mode is 'ranking', the interactions are sorted by prediction score.

    INPUTS:
    - features: the input features for the model (output of construct_inputframe)
    - interactions: the interactions dataframe (output of construct_inputframe)
    - model_file: the name of the trained model file
    - path: path to the general data folder
    - data suffix to optionally add to the saved file name (default='')
    - mode: 'scores' or 'ranking' (default='scores').
    OUTPUTS: (ranked) interactions with predictions
    """
    # load the model
    xgb = XGBClassifier()
    xgb.load_model(path+'/'+model_file)
    interactions.reset_index(drop=True, inplace=True)

    # make predictions
    scores_xgb = xgb.predict_proba(features)[:,1]
    if mode == 'scores':
        interactions = pd.concat([interactions, pd.DataFrame({'prediction_score': scores_xgb})], axis=1)
        interactions.to_csv(path+'/predictions'+suffix+'.csv', index=False)
        return interactions
    elif mode == 'ranking':
        interactions = pd.concat([interactions, pd.DataFrame({'prediction_score': scores_xgb})], axis=1)
        interactions = interactions.sort_values(by='prediction_score', ascending=False)
        interactions.to_csv(path+'/predictions'+suffix+'.csv', index=False)
        return interactions


