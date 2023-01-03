"""
Created on 03/01/23

@author: dimiboeckaerts

PhageHostLearn DATA PROCESSING
"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import json
import subprocess
import numpy as np
import pandas as pd
import processing_utils as pu
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm.notebook import tqdm
from os import listdir
from xgboost import XGBClassifier
from bio_embeddings.embed import ProtTransBertBFDEmbedder


# 1 - FUNCTIONS
# --------------------------------------------------
def phanotate_processing(general_path, phage_genomes_path, phanotate_path, data_suffix=''):
    """
    This function loops over the genomes in the phage genomes folder and processed those to
    genes with PHANOTATE.

    INPUTS:
    - general path of the PhageHostLearn framework and data
    - phage genomes path to the folder containing the phage genomes as separate FASTA files
    - phanotate path to the phanotate.py file
    - data suffix to add to the phage_genes.csv file from PHANOTATE (default='')
    OUTPUT: phage_genes.csv containing all the phage genes.
    """
    phage_files = listdir(phage_genomes_path)
    phage_files.remove('.DS_Store')
    #record = SeqIO.read(phages_dir+'/'+phage_files[0], 'fasta')
    bar = tqdm(total=len(phage_files), position=0, leave=True)
    name_list = []; gene_list = []; gene_ids = []

    for file in phage_files:
        count = 1
        # access PHANOTATE
        file_dir = phage_genomes_path+'/'+file
        raw_str = phanotate_path + ' ' + file_dir
        process = subprocess.Popen(raw_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        std_splits = stdout.split(sep=b'\n')
        std_splits = std_splits[2:] #std_splits.pop(0)
        
        # Save and reload TSV
        temp_tab = open(general_path+'/phage_results.tsv', 'wb')
        for split in std_splits:
            split = split.replace(b',', b'') # replace commas for pandas compatibility
            temp_tab.write(split + b'\n')
        temp_tab.close()
        results_orfs = pd.read_csv(general_path+'/phage_results.tsv', sep='\t', lineterminator='\n', index_col=False)
        
        # fill up lists accordingly
        name = file.split('.fasta')[0]
        sequence = str(SeqIO.read(file_dir, 'fasta').seq)
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
            
        # update progress
        bar.update(1)
    bar.close()
    os.remove(general_path+'/phage_results.tsv')

    # Export final genes database
    genebase = pd.DataFrame(list(zip(name_list, gene_ids, gene_list)), columns=['phage_ID', 'gene_ID', 'gene_sequence'])
    genebase.to_csv(general_path+'/phage_genes'+data_suffix+'.csv', index=False)
    return


def compute_protein_embeddings(general_path, data_suffix=''):
    """
    This function computes protein embeddings -> SLOW ON CPU! Alternatively, can be done
    in the cloud, using the separate notebook (compute_embeddings_cloud).
    """
    genebase = pd.read_csv(general_path+'/phage_genes'+data_suffix+'.csv')
    embedder = ProtTransBertBFDEmbedder()
    names = list(genebase['gene_ID'])
    sequences = [str(Seq(sequence).translate())[:-1] for sequence in genebase['gene_sequence']]
    embeddings = [embedder.reduce_per_protein(embedder.embed(sequence)) for sequence in tqdm(sequences)]
    embeddings_df = pd.concat([pd.DataFrame({'ID':names}), pd.DataFrame(embeddings)], axis=1)
    embeddings_df.to_csv(general_path+'/phage_protein_embeddings'+data_suffix+'.csv', index=False)
    return


def phageRBPdetect(general_path, pfam_path, hmmer_path, xgb_path, gene_embeddings_path, data_suffix=''):
    """
    This function loops over the phage genes detected by PHANOTATE and uses the PhageRBPdetect pipeline
    to detect the RBPs in those genes.

    INPUTS:
    - general path of the PhageHostLearn framework and data
    - pfam path to the pfam database file of phageRBP HMMs (.hmm)
    - hmmer path to the HMMER software, e.g. '/Users/Sally/hmmer-3.3.1'
    - xgb path to the trained XGBoost model for RBP detection (.json)
    - gene embeddings path to the file with ProtTransBertBFD embeddings of the phage genes (see separate notebook!)
    - data suffix corresponding to the PHANOTATE output, can be used to designate a test set for example (default='')
    OUTPUT: RBPbase.csv containing all the detected RBPs.
    """
    genebase = pd.read_csv(general_path+'/phage_genes'+data_suffix+'.csv')

    # define all the blocks (HMMs) we want scores for
    new_blocks = ['Phage_T7_tail', 'Tail_spike_N', 'Prophage_tail', 'BppU_N', 'Mtd_N', 
                'Head_binding', 'DUF3751', 'End_N_terminal', 'phage_tail_N', 'Prophage_tailD1', 
                'DUF2163', 'Phage_fiber_2', 'unknown_N0', 'unknown_N1', 'unknown_N2', 'unknown_N3', 'unknown_N4', 
                'unknown_N6', 'unknown_N10', 'unknown_N11', 'unknown_N12', 'unknown_N13', 'unknown_N17', 'unknown_N19', 
                'unknown_N23', 'unknown_N24', 'unknown_N26','unknown_N29', 'unknown_N36', 'unknown_N45', 'unknown_N48', 
                'unknown_N49', 'unknown_N53', 'unknown_N57', 'unknown_N60', 'unknown_N61', 'unknown_N65', 'unknown_N73', 
                'unknown_N82', 'unknown_N83', 'unknown_N101', 'unknown_N114', 'unknown_N119', 'unknown_N122', 
                'unknown_N163', 'unknown_N174', 'unknown_N192', 'unknown_N200', 'unknown_N206', 'unknown_N208', 
                'Lipase_GDSL_2', 'Pectate_lyase_3', 'gp37_C', 'Beta_helix', 'Gp58', 'End_beta_propel', 
                'End_tail_spike', 'End_beta_barrel', 'PhageP22-tail', 'Phage_spike_2', 
                'gp12-short_mid', 'Collar', 
                'unknown_C2', 'unknown_C3', 'unknown_C8', 'unknown_C15', 'unknown_C35', 'unknown_C54', 'unknown_C76', 
                'unknown_C100', 'unknown_C105', 'unknown_C112', 'unknown_C123', 'unknown_C179', 'unknown_C201', 
                'unknown_C203', 'unknown_C228', 'unknown_C234', 'unknown_C242', 'unknown_C258', 'unknown_C262', 
                'unknown_C267', 'unknown_C268', 'unknown_C274', 'unknown_C286', 'unknown_C292', 'unknown_C294', 
                'Peptidase_S74', 'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']

    # optionally press database first if not done already
    output, err = pu.hmmpress_python(hmmer_path, pfam_path)

    # get domains & scores
    phage_genes = genebase['gene_sequence']
    phage_ids = genebase['phage_ID']

    hmm_scores = {item:[0]*len(phage_genes) for item in new_blocks}
    bar = tqdm(total=len(phage_genes), desc='Scanning the genes', position=0, leave=True)
    for i, sequence in enumerate(phage_genes):
        hits, scores, biases, ranges = pu.gene_domain_scan(hmmer_path, pfam_path, [sequence])
        for j, dom in enumerate(hits):
            hmm_scores[dom][i] = scores[j]
        bar.update(1)
    bar.close()
    hmm_scores_array = np.asarray(pd.DataFrame(hmm_scores))

    # load protein embeddings to make predictions for and concat them with the HMM scores
    embeddings_df = pd.read_csv(gene_embeddings_path)
    embeddings = np.asarray(embeddings_df.iloc[:, 1:]) # zeroth column is the name!
    features = np.concatenate((embeddings, hmm_scores_array), axis=1)

    # load trained model
    xgb_saved = XGBClassifier()
    xgb_saved.load_model(xgb_path)

    # make predictions with the XGBoost model
    score_xgb = xgb_saved.predict_proba(features)[:,1]
    preds_xgb = (score_xgb > 0.5)*1

    # construct RBPbase with all the information
    RBPbase = {'phage_ID':[], 'protein_ID':[], 'protein_sequence':[], 'dna_sequence':[], 
            'xgb_score':[]}
    for i, dna_sequence in enumerate(genebase['gene_sequence']):
        if preds_xgb[i] == 1:
            RBPbase['phage_ID'].append(genebase['phage_ID'][i])
            RBPbase['protein_ID'].append(genebase['gene_ID'][i])
            RBPbase['protein_sequence'].append(str(Seq(dna_sequence).translate())[:-1])
            RBPbase['dna_sequence'].append(dna_sequence)
            RBPbase['xgb_score'].append(score_xgb[i])
    RBPbase = pd.DataFrame(RBPbase)

    # filter for length and save dataframe
    to_delete = [i for i, protein_seq in enumerate(RBPbase['protein_sequence']) if (len(protein_seq)<200 or len(protein_seq)>1500)]
    RBPbase = RBPbase.drop(to_delete)
    RBPbase = RBPbase.reset_index(drop=True)
    RBPbase.to_csv(general_path+'/RBPbase'+data_suffix+'.csv', index=False)
    return


def process_bacterial_genomes(general_path, bact_genomes_path, database_path, kaptive_path, data_suffix=''):
    """
    This function processes the bacterial genomes with Kaptive, into a dictionary of K-locus proteins.

    INPUTS:
    - general path of the PhageHostLearn framework and data
    - bact genomes path to the folder of the bacterial genomes as individual FASTA files
    - database path to the Kaptive K-locus reference database (.gbk)
    - kaptive path to the kaptive.py file
    - data suffix corresponding to the PHANOTATE output, can be used to designate a test set for example (default='')
    OUTPUT: Locibase.json containing all the K-locus proteins for each bacterium.
    """
    Locibase, seros = pu.compute_kaptive_from_directory(kaptive_path, database_path, bact_genomes_path, general_path)
    pd.DataFrame(seros, columns=['sero']).to_csv(general_path+'/serotypes'+data_suffix+'.csv', index=False)
    dict_file = open(general_path+'/Locibase'+data_suffix+'.json', 'w')
    json.dump(Locibase, dict_file)
    dict_file.close()
    return


def process_interactions(general_path, interactions_xlsx_path, data_suffix=''):
    """
    This function processes the interactions matrix in a given .xlsx file to a filtered Pandas dataframe (.csv).

    INPUTS:
    - general path to the PhageHostLearn framework and data
    - interactions_xlsx_path to the xslx file
    - data suffix corresponding to the PHANOTATE output, can be used to designate a test set for example (default='')
    """
    # process xlsx file into a Pandas dataframe
    output = general_path+'/phage_host_interactions'+data_suffix
    pu.xlsx_database_to_csv(interactions_xlsx_path, output)

    # filter identical interactions (at phage level and loci nt level -> PSI-CD-HIT)
    # ... (could be interesting but not crucial, as we train on experimentally validated data, trust in good sample collection)

    return



def process_data(general_path, phage_genomes_path, bact_genomes_path, phanotate_path, pfam_path, hmmer_path, xgb_path, kaptive_database_path, 
                    kaptive_path, interactions_xlsx_path, data_suffix):
    """
    This function is a wrapper for all previous functions.
    """
    # process phage genomes with PHANOTATE
    print('Processing phage genomes with PHANOTATE...')
    phanotate_processing(general_path, phage_genomes_path, phanotate_path, data_suffix=data_suffix)

    # compute phage protein embeddings
    print('Computing protein embeddings... (this can take a while on a CPU)')
    compute_protein_embeddings(general_path, data_suffix=data_suffix)

    # detect phage RBPs
    print('Detecting RBPs with PhageRBPdetect...')
    gene_embeddings_path = general_path+'/phage_protein_embeddings'+data_suffix+'.csv'
    phageRBPdetect(general_path, pfam_path, hmmer_path, xgb_path, gene_embeddings_path, data_suffix=data_suffix)

    # process bacterial genomes with Kaptive
    print('Processing bacterial genomes with Kaptive...')
    process_bacterial_genomes(general_path, bact_genomes_path, kaptive_database_path, kaptive_path, data_suffix=data_suffix)

    # process the interactions matrix
    print('Processing the interaction matrix...')
    process_interactions(general_path, interactions_xlsx_path, data_suffix=data_suffix)

    # finish
    print('Done!')
    return

