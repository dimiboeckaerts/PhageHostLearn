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
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SearchIO import HmmerIO
from tqdm.notebook import tqdm
from os import listdir
from xgboost import XGBClassifier
from bio_embeddings.embed import ProtTransBertBFDEmbedder


# 1 - FUNCTIONS
# --------------------------------------------------
def hmmpress_python(hmm_path, pfam_file):
    """
    Presses a profiles database, necessary to do scanning.
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    press_str = 'hmmpress ' + pfam_file
    command = cd_str+'; '+press_str
    press_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    press_out, press_err = press_process.communicate()

    return press_out, press_err


def single_hmmscan_python(hmm_path, pfam_file, fasta_file):
    """
    Does a hmmscan for a given FASTA file of one (or multiple) sequences,
    against a given profile database. Assuming an already pressed profiles
    database (see function above).
    
    INPUT: all paths to the hmm, profiles_db, fasta and results file given as strings.
            results_file should be a .txt file
    OUPUT: ...
    """

    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_str = cd_process.communicate()

    # scan the sequences
    scan_str = 'hmmscan ' + pfam_file + ' ' + fasta_file + ' > hmmscan_out.txt'
    scan_process = subprocess.Popen(scan_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    scan_out, scan_err = scan_process.communicate()
    
    # process output
    results_handle = open('hmmscan_out.txt')
    scan_res = HmmerIO.Hmmer3TextParser(results_handle)
    os.remove('hmmscan_out.txt')
    
    return scan_out, scan_err, scan_res


def hmmscan_python(hmm_path, pfam_file, sequences_file, threshold=18):
    """
    Expanded version of the function above for a file with multiple sequences,
    where the results are parsed one by one to fetch the names of domains that
    we're interested in. Assumes an already pressed profiles database (see 
    funtion above).
    
    HMMSCAN = scan a (or multiple) sequence(s) for domains.
    """
    
    domains = []
    scores = []
    biases = []
    ranges = []
    for sequence in SeqIO.parse(sequences_file, 'fasta'):
        # make single-sequence FASTA file
        temp_fasta = open('single_sequence.fasta', 'w')
        temp_fasta.write('>'+sequence.id+'\n'+str(sequence.seq)+'\n')
        temp_fasta.close()
        
        # scan HMM
        _, _, scan_res = single_hmmscan_python(hmm_path, pfam_file, 'single_sequence.fasta')
        
        # fetch domains in the results
        for line in scan_res:   
            try:   
                for hit in line.hits:
                    hsp = hit._items[0] # highest scoring domain
                    aln_start = hsp.query_range[0]
                    aln_stop = hsp.query_range[1]
        
                    if (hit.bitscore >= threshold) & (hit.id not in domains):
                        domains.append(hit.id)
                        scores.append(hit.bitscore)
                        biases.append(hit.bias)
                        ranges.append((aln_start,aln_stop))
            except IndexError: # some hits don't have an individual domain hit
                pass
    
    # remove last temp fasta file
    os.remove('single_sequence.fasta')

    return domains, scores, biases, ranges


def gene_domain_scan(hmmpath, pfam_file, gene_hits, threshold=18):
    """
    This function does a hmmscan on the gene_hit(s) after translating them to
    protein sequences and saving it in one FASTA file.
    """

    # write the protein sequences to file
    hits_fasta = open('protein_hits.fasta', 'w')
    for i, gene_hit in enumerate(gene_hits):
        protein_sequence = str(Seq(gene_hit).translate())[:-1]
        hits_fasta.write('>'+str(i)+'_proteindomain_hit'+'\n'+protein_sequence+'\n')
    hits_fasta.close()
        
    # fetch domains with hmmscan
    domains, scores, biases, ranges = hmmscan_python(hmmpath, pfam_file, 'protein_hits.fasta', threshold)

    # remove last temp fasta file
    os.remove('protein_hits.fasta')
    
    return domains, scores, biases, ranges


def kaptive_python(database_path, file_path, output_path):
    """
    This function is a wrapper for the Kaptive Python file to be executed from another Python script.
    This wrapper runs on a single FASTA file (one genome) and also produces a single FASTA file.
    
    Input:
    - kaptive_directory: directory with kaptive.py in
    - database_path: path (string) to the database (.gbk file)
    - file_path: path (string) to the file (FASTA)
    - output_path: path for output
    
    Output:
    - a single fasta file of the locus (single piece or multiple ones) per genome
    """
    #cd_command = 'cd ' + kaptive_directory
    
    command = 'python kaptive.py -a ' + file_path + ' -k ' + database_path + ' -o ' + output_path + '/ --no_table'
    #command = cd_command + '; ' + kaptive_command
    ssprocess = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ssout, sserr = ssprocess.communicate()
    
    return ssout, sserr


def xlsx_database_to_csv(xlsx_file, save_path, index_col=0, header=0, export=True):
    """
    This function processes an xlsx file with interactions into a Pandas dataframe
    that is saved as an .csv file. 
    
    Input:
    * xlsx_file: path to file, file must have a column (index_col) with bacteria names and a 
        row (header) with phage names.
    * save_path: file path (and name) to save .csv file
    * index_col: number of the column in which the bacteria names are (starts at 0! - default)
    * header: number of the row in which the phage names are (starts at O! - default)
    * export: whether or not to export to csv, default = true (duh)
    """
    IM = pd.read_excel(xlsx_file, index_col=index_col, header=header)
    #bacteria_names = list(IM.index)
    #phage_names = list(IM.columns)
    if export==True:
        IM.to_csv(save_path+'.csv')
        return
    else:
        return IM


def add_to_database(old_database, new_xlsx_file, save_path, index_col=0, header=0):
    """
    This function adds data from a new xlsx file to an already existing database.
    
    Input:
    * old_database: path to csv file of old database (by default index_col and header are 0)
    * new_xlsx_file: path to file, file must have a column (index_col) with bacteria names and a 
        row (header) with phage names.
    * save_path: file path to save .csv file of the NEW MERGED DATABASE
    * index_col: number of the column in which the bacteria names are (starts at 0! - default)
    * header: number of the row in which the phage names are (starts at O! - default)
    """
    # load old data (first is index_col)
    old_database = pd.read_csv(old_database, index_col=0)
    
    # process new data
    new_database = xlsx_database_to_csv(new_xlsx_file, '', index_col=index_col, header=header, export=False)
    phage_overlap = [phage for phage in list(old_database.columns) if phage in list(new_database.columns)]
    bacteria_overlap = [bacterium for bacterium in list(old_database.index) if bacterium in list(new_database.index)]
    if (len(phage_overlap) > 0) or (len(bacteria_overlap) > 0):
        print('Oops, there seem to be duplicate(s) in the added phages and or bacteria...')
        print('Phage name duplicates:', phage_overlap)
        print('Bacteria name duplicates:', bacteria_overlap)
        return
    else:
        # merge and save
        merged_database = pd.concat([old_database, new_database])
        merged_database.to_csv(save_path+'.csv')
        return
    

def phanotate_processing(general_path, phage_genomes_path, phanotate_path, data_suffix='', add=False, test=False):
    """
    This function loops over the genomes in the phage genomes folder and processed those to
    genes with PHANOTATE.

    INPUTS:
    - general path of the PhageHostLearn framework and data
    - phage genomes path to the folder containing the phage genomes as separate FASTA files
    - phanotate path to the phanotate.py file
    - data suffix to add to the phage_genes.csv file from PHANOTATE (default='')
    - add: bool whether or not we are just adding new data or processing from scratch (default=False)
    - test: bool whether or not we want to test the function (default=False)
    OUTPUT: phage_genes.csv containing all the phage genes.
    """
    phage_files = listdir(phage_genomes_path)
    phage_files.remove('.DS_Store')
    if add == True:
        RBPbase = pd.read_csv(general_path+'/RBPbase'+data_suffix+'.csv')
        phage_ids = list(set(RBPbase['phage_ID']))
        phage_files = [x for x in phage_files if x.split('.fasta')[0] not in phage_ids]
        print('Processing ', len(phage_files), ' more phages (add=True)')
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

    # remove .tsv file if we're not in test mode
    if test == False:
        os.remove(general_path+'/phage_results.tsv')

    # Export final genes database
    genebase = pd.DataFrame(list(zip(name_list, gene_ids, gene_list)), columns=['phage_ID', 'gene_ID', 'gene_sequence'])
    if add == True:
        old_genebase = pd.read_csv(general_path+'/phage_genes'+data_suffix+'.csv')
        genebase = pd.concat([old_genebase, genebase], axis=0)
    genebase.to_csv(general_path+'/phage_genes'+data_suffix+'.csv', index=False)
    return


def compute_protein_embeddings(general_path, data_suffix='', add=False):
    """
    This function computes protein embeddings -> SLOW ON CPU! Alternatively, can be done
    in the cloud, using the separate notebook (compute_embeddings_cloud).
    """
    genebase = pd.read_csv(general_path+'/phage_genes'+data_suffix+'.csv')
    embedder = ProtTransBertBFDEmbedder()
    if add == True:
        old_embeddings_df = pd.read_csv(general_path+'/phage_protein_embeddings'+data_suffix+'.csv')
        protein_ids = list(old_embeddings_df['ID'])
        sequences = []; names = []
        for i, sequence in enumerate(genebase['gene_sequence']):
            if genebase['gene_ID'][i] not in protein_ids:
                sequences.append(str(Seq(sequence).translate())[:-1])
                names.append(genebase['gene_ID'][i])
    else:
        names = list(genebase['gene_ID'])
        sequences = [str(Seq(sequence).translate())[:-1] for sequence in genebase['gene_sequence']]
    
    embeddings = [embedder.reduce_per_protein(embedder.embed(sequence)) for sequence in tqdm(sequences)]
    embeddings_df = pd.concat([pd.DataFrame({'ID':names}), pd.DataFrame(embeddings)], axis=1)
    if add == True:
        embeddings_df = pd.DataFrame(np.vstack([old_embeddings_df, embeddings_df]), columns=old_embeddings_df.columns)
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
    output, err = hmmpress_python(hmmer_path, pfam_path)
    print(output)

    # get domains & scores
    phage_genes = genebase['gene_sequence']
    phage_ids = genebase['phage_ID']

    hmm_scores = {item:[0]*len(phage_genes) for item in new_blocks}
    bar = tqdm(total=len(phage_genes), position=0, leave=True)
    for i, sequence in enumerate(phage_genes):
        hits, scores, biases, ranges = gene_domain_scan(hmmer_path, pfam_path, [sequence])
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


def process_bacterial_genomes(general_path, bact_genomes_path, database_path, data_suffix='', add=False):
    """
    This function processes the bacterial genomes with Kaptive, into a dictionary of K-locus proteins.

    INPUTS:
    - general path of the PhageHostLearn framework and data
    - bact genomes path to the folder of the bacterial genomes as individual FASTA files
    - database path to the Kaptive K-locus reference database (.gbk)
    - data suffix corresponding to the PHANOTATE output, can be used to designate a test set for example (default='')
    - add: bool whether or not we are just adding new data or processing from scratch (default=False)
    OUTPUT: Locibase.json containing all the K-locus proteins for each bacterium.
    REMARK: only downside here is that the big fasta file gets overwritten.
    """
    # get fasta files
    fastas = listdir(bact_genomes_path)
    try:
        fastas.remove('.DS_Store')
    except:
        pass
    if add == True:
        dict_file = open(general_path+'/Locibase'+data_suffix+'.json')
        old_locibase = json.load(dict_file)
        loci_accessions = list(old_locibase.keys())
        fastas = [x for x in fastas if x.split('.fasta')[0] not in loci_accessions]
        print('Processing ', len(fastas), ' more bacteria (add=True)')

    # run Kaptive
    accessions = [file.split('.fasta')[0] for file in fastas]
    serotypes = []
    loci_results = {}
    pbar = tqdm(total=len(fastas))
    big_fasta = open(general_path+'/kaptive_results_all_loci.fasta', 'w')
    for i, file in enumerate(fastas):
        # kaptive
        file_path = bact_genomes_path+'/'+file
        out, err = kaptive_python(database_path, file_path, general_path)
        
        # process json -> proteins in dictionary
        results = json.load(open(general_path+'/kaptive_results.json'))
        serotypes.append(results[0]['Best match']['Type'])
        for gene in results[0]['Locus genes']:
            try:
                protein = gene['tblastn result']['Protein sequence']
                protein = protein.replace('-', '')
                protein = protein.replace('*', '')
            except KeyError:
                protein = gene['Reference']['Protein sequence']
            if accessions[i] in list(loci_results.keys()):
                loci_results[accessions[i]].append(protein[:-1])
            else:
                loci_results[accessions[i]] = [protein[:-1]]

        # write big fasta file with loci
        loci_sequence = ''
        for record in SeqIO.parse(general_path+'/kaptive_results_'+file, 'fasta'):
            loci_sequence = loci_sequence + str(record.seq)
        big_fasta.write('>'+accessions[i]+'\n'+loci_sequence+'\n')

        # delete temp kaptive files
        os.remove(file_path+'.ndb')
        os.remove(file_path+'.not')
        os.remove(file_path+'.ntf')
        os.remove(file_path+'.nto')
        os.remove(general_path+'/kaptive_results.json')
        os.remove(general_path+'/kaptive_results_'+file)

        # update progress
        pbar.update(1)
    pbar.close()
    big_fasta.close()

    # save to dictionary in .json file
    sero_df = pd.DataFrame(serotypes, columns=['sero'])
    if add == True:
        loci_results = {**old_locibase, **loci_results}
        old_seros = pd.read_csv(general_path+'/serotypes'+data_suffix+'.csv')
        sero_df = pd.concat([old_seros, sero_df], axis=0)
    sero_df.to_csv(general_path+'/serotypes'+data_suffix+'.csv', index=False)
    dict_file = open(general_path+'/Locibase'+data_suffix+'.json', 'w')
    json.dump(loci_results, dict_file)
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
    xlsx_database_to_csv(interactions_xlsx_path, output)

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
    #print('Computing protein embeddings... (this can take a while on a CPU)')
    #compute_protein_embeddings(general_path, data_suffix=data_suffix)

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

