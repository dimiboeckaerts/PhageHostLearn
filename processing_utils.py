"""
Created on 07/09/21

@author: dimiboeckaerts

PROCESSING UTILS FOR THE RBP-R PREDICTION FRAMEWORK
"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import json
import math
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SearchIO import HmmerIO


# 1 - FUNCTIONS
# --------------------------------------------------
def kaptive_python(project_dir, data_dir, database_name, file_names, results_dir):
    """
    This function is a wrapper for the Kaptive Python file to be executed from another Python script.
    
    Input:
    - project_directory: the location of kaptive.py (preferrably in the same project folder)
    - data_directory: location of the database and sequence file(s) to loop over
    - database_name: string of the name of the database (.gbk file)
    - file_names: list of one or multiple strings of the file name(s)
    
    Output:
    - a single fasta file of the locus per (a single piece or multiple ones)
    """
    cd_command = 'cd ' + project_dir
    kaptive_file_names = []
    
    for name in tqdm(file_names):
        kaptive_command = 'python kaptive.py -a ' + data_dir + '/' + name + ' -k ' + data_dir + '/' + database_name + ' -o ' + results_dir + '/ --no_table --no_json'
        command = cd_command + '; ' + kaptive_command
        ssprocess = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ssout, sserr = ssprocess.communicate()
        kaptive_file_names.append('kaptive_results_'+name)
        
    return kaptive_file_names, ssout, sserr

def hmmpress_python(hmm_path, pfam_file):
    """
    Presses a profiles database, necessary to do scanning.
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_err = cd_process.communicate()
    
    # compress the profiles db
    press_str = 'hmmpress ' + pfam_file
    press_process = subprocess.Popen(press_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    press_out, press_err = press_process.communicate()

    return press_out, press_err

def hmmbuild_python(hmm_path, output_file, msa_file):
    """
    Build a profile HMM from an input multiple sequence alignment (Stockholm format).
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_err = cd_process.communicate()
    
    # compress the profiles db
    press_str = 'hmmbuild ' + output_file + ' ' + msa_file
    press_process = subprocess.Popen(press_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    count_dict = {}
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
                        count_dict[hit.id] = 1
                    elif (hit.bitscore >= threshold) & (hit.id in domains):
                        count_dict[hit.id] += 1
            except IndexError: # some hits don't have an individual domain hit
                pass
    
    # remove last temp fasta file
    os.remove('single_sequence.fasta')
    
    return domains, scores, biases, ranges
    

def hmmfetch_python(hmm_path, pfam_file, domains, output_file):
    """
    Fetches the HMM profiles for given domains. Necessary to do hmmsearch with
    afterwards.
    
    INPUT: paths to files and domains as a list of strings.
    """
    
    # save domains as txt file
    domain_file = open('selected_domains.txt', 'w')
    for domain in domains:
        domain_file.write(domain+'\n')
    domain_file.close()
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_str = cd_process.communicate()
    
    # fetch selected domains in new hmm
    fetch_str = 'hmmfetch -f ' + pfam_file + ' selected_domains.txt'
    fetch_process = subprocess.Popen(fetch_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fetch_out, fetch_err = fetch_process.communicate()
    
    # write to specified output file
    hmm_out = open(output_file, 'wb')
    hmm_out.write(fetch_out)
    hmm_out.close()
    
    return fetch_out, fetch_err


def hmmsearch_python(hmm_path, selected_profiles_file, sequences_db):
    """
    HMMSEARCH = search (selected) domain(s) in a sequence database.
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_str = cd_process.communicate()
    
    # search domains in sequence database
    search_str = 'hmmsearch ' + selected_profiles_file + ' ' + sequences_db + ' > hmmsearch_out.txt'
    search_process = subprocess.Popen(search_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    search_out, search_err = search_process.communicate()
    
    # process output
    results_handle = open('hmmsearch_out.txt')
    search_res = HmmerIO.Hmmer3TextParser(results_handle)
    os.remove('hmmsearch_out.txt')
    
    sequence_hits = []
    sequence_scores = []
    sequence_bias = []
    sequence_range = []
    for line in search_res:
        for hit in line:
            try:
                hsp = hit._items[0]
                aln_start = hsp.query_range[0]
                aln_stop = hsp.query_range[1]
                if (hit.bitscore >= 25):
                    sequence_hits.append(hit.id)
                    sequence_scores.append(hit.bitscore)
                    sequence_bias.append(hit.bias)
                    sequence_range.append((aln_start,aln_stop))
            except IndexError:
                pass
                
    return sequence_hits, sequence_scores, sequence_bias, sequence_range


def hmmsearch_thres_python(hmm_path, selected_profiles_file, sequences_db, threshold):
    """
    HMMSEARCH = search (selected) domain(s) in a sequence database.
    """
    
    # change directory
    cd_str = 'cd ' + hmm_path
    cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cd_out, cd_str = cd_process.communicate()
    
    # search domains in sequence database
    search_str = 'hmmsearch ' + selected_profiles_file + ' ' + sequences_db + ' > hmmsearch_out.txt'
    search_process = subprocess.Popen(search_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    search_out, search_err = search_process.communicate()
    
    # process output
    results_handle = open('hmmsearch_out.txt')
    search_res = HmmerIO.Hmmer3TextParser(results_handle)
    os.remove('hmmsearch_out.txt')
    
    sequence_hits = []
    sequence_scores = []
    sequence_bias = []
    sequence_range = []
    for line in search_res:
        for hit in line:
            try:
                hsp = hit._items[0]
                aln_start = hsp.query_range[0]
                aln_stop = hsp.query_range[1]
                if (hit.bitscore >= threshold):
                    sequence_hits.append(hit.id)
                    sequence_scores.append(hit.bitscore)
                    sequence_bias.append(hit.bias)
                    sequence_range.append((aln_start,aln_stop))
            except IndexError:
                pass
                
    return sequence_hits, sequence_scores, sequence_bias, sequence_range


def gene_domain_search(hmmpath, selected_profiles_file, gene_sequence):
    """
    This function translates a given gene sequence to its protein sequence, 
    saves it as a FASTA file and searches the given domain(s) in the protein
    sequence.
    """
    # translate
    protein_sequence = str(Seq(gene_sequence).translate())[:-1] # don't need stop codon

    # write fasta
    temp_fasta = open('protein_sequence.fasta', 'w')
    temp_fasta.write('>protein_sequence'+'\n'+protein_sequence+'\n')
    temp_fasta.close()
    
    # search domain
    hits, scores, biases, ranges = hmmsearch_python(hmmpath, selected_profiles_file, 'protein_sequence.fasta')

    # delete fasta
    os.remove('protein_sequence.fasta')
    
    return hits, scores, biases, ranges


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
    
    return domains, scores, biases, ranges


def all_domains_scan(path, pfam_file, gene_sequences):
    """
    scan all sequences and make dictionary of results
    """
    domain_dict = {'gene_sequence': [], 'domain_name': [], 'position': [], 'score': [], 'bias': [], 'aln_range': []}
    bar = tqdm(total=len(gene_sequences), desc='Scanning the genes', position=0, leave=True)
    for gene in gene_sequences:
        hits, scores, biases, ranges = gene_domain_scan(path, pfam_file, [gene])
        for i, dom in enumerate(hits):
            domain_dict['gene_sequence'].append(gene)
            domain_dict['domain_name'].append(dom)
            domain_dict['score'].append(scores[i])
            domain_dict['bias'].append(biases[i])
            domain_dict['aln_range'].append(ranges[i])
            if ranges[i][1] > 200:
                domain_dict['position'].append('C')
            else:
                domain_dict['position'].append('N')
            
        bar.update(1)
    bar.close()
    
    return domain_dict


def domain_RBP_predictor(path, pfam_file, sequences, structural=[], binding=[], chaperone=[], archive=[]):
    """
    This function predicts whether or not a sequence is a phage RBP based on known
    related phage RBP protein domains in Pfam. Predictions are made based on the
    knowledge that RBPs are modular, consisting of a structural (N-terminal) domain,
    a C-terminal binding domain and optionally a chaperone domain at the C-end.
    
    Inputs: 
        - structural, binding, chaperone: curated lists of phage-related domains
            in Pfam.
        - path: path to HMM software for detection of the domains
        - pfam_file: link to local Pfam database file (string)
        - sequences: 
            * link to FASTA file of gene sequences to predict (string)
            OR 
            * list of sequences as string
            OR 
            * dictionary of domains (cfr. function 'all_domains_scan')
    Output:
        - a pandas dataframe of sequences in which at least one domain has been
            detected.
    """
    
    # initializations
    predictions_dict = {'gene_sequence': [], 'structural_domains': [], 
                        'binding_domains': [], 'chaperone_domains': [],
                        'archived_domains': []}
    if type(sequences) == 'string':
        # make list of sequences
        sequence_list = []
        for record in SeqIO.parse(sequences, 'fasta'):
            sequence_list.append(str(record.seq))
        # make domain dictionary
        domains_dictionary = all_domains_scan(path, pfam_file, sequence_list)
        
    elif type(sequences) == 'list':
        # make domain dictionary
        domains_dictionary = all_domains_scan(path, pfam_file, sequences)
    else: # dict
        domains_dictionary = sequences
    
    # make predictions based on the domains_dictionary
    domain_sequences = list(set(domains_dictionary['gene_sequence']))
    for sequence in domain_sequences:
        # detect all domains at correct indices: every line in domains_dictionary
        # corresponds to a gene sequence and a domain in it (one sequence can
        # have multiple domains, thus multiple lines in the dict).
        domains = []
        indices = [i for i, gene in enumerate(domains_dictionary['gene_sequence']) 
                    if sequence == gene]
        for index in indices:
            OM_score = math.floor(math.log(domains_dictionary['score'][index], 10)) # order of magnitude
            OM_bias = math.floor(math.log(domains_dictionary['bias'][index]+0.00001, 10))
            if (OM_score > OM_bias):
                domains.append(domains_dictionary['domain_name'][index])
               
        # detect the domains of interest
        struct_detected = [dom for dom in domains if dom in structural]
        binding_detected = [dom for dom in domains if dom in binding]
        chaperone_detected = [dom for dom in domains if dom in chaperone]
        archive_detected = [dom for dom in domains if dom in archive]
        
        # append results dictionary
        if (len(struct_detected) > 0) or (len(binding_detected) > 0) or (len(chaperone_detected) > 0):
            predictions_dict['gene_sequence'].append(sequence)
            predictions_dict['structural_domains'].append(struct_detected)
            predictions_dict['binding_domains'].append(binding_detected)
            predictions_dict['chaperone_domains'].append(chaperone_detected)
            predictions_dict['archived_domains'].append(archive_detected)
    
    return predictions_dict


def cdhit_python(cdhit_path, input_file, output_file, c=0.50, n=3):
    """
    This function executes CD-HIT clustering commands from within Python. To install
    CD-HIT, do so via conda: conda install -c bioconda cd-hit. By default, CD-HIT
    works via a global alignment approach, which is good for our application as
    we cut the sequences to 'one unknown domain' beforehand.
    
    Input:
        - cdhit_path: path to CD-HIT software
        - input_file: FASTA file with protein sequences
        - output file: path to output (will be one FASTA file and one .clstr file)
        - c: threshold on identity for clustering
        - n: word length (3 for thresholds between 0.5 and 0.6)
    """
    
    cd_str = 'cd ' + cdhit_path # change directory
    raw_str = './cd-hit -i ' + input_file + ' -o ' + output_file + ' -c ' + str(c) + ' -n ' + str(n) + ' -d 0'
    command = cd_str+'; '+ raw_str
    #cd_process = subprocess.Popen(cd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #cd_out, cd_err = cd_process.communicate()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    
    return stdout, stderr


def psi_cdhit_python(cdhit_path, input_file, output_file, c=0.30):
    """
    This function executes PSI CD-HIT clustering commands from within Python. To install
    CD-HIT, do so via conda: conda install -c bioconda cd-hit. By default, CD-HIT
    works via a global alignment approach, which is good for our application as
    we cut the sequences to 'one unknown domain' beforehand.
    
    Input:
        - cdhit_path: path to CD-HIT software
        - input_file: FASTA file with protein sequences
        - output file: path to output (will be one FASTA file and one .clstr file)
        - c: threshold on identity for clustering
    """
    cd_str = 'cd ' + cdhit_path # change directory
    raw_str = './psi-cd-hit.pl -i ' + input_file + ' -o ' + output_file + ' -c ' + str(c)
    command = cd_str+'; '+ raw_str
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    
    return stdout, stderr


def RBPbase_species_filter(rbp_data, data_dir, results_dir, species):
    """
    This function creates a subset of RBPbase for a specific host species for further processing.

    Input:
    - rbp_data: name of the RBP database (string)
    - data_dir: location of the database and where the fasta files will be stored (string)
    - species: the single species to create a subset for

    Output:
    - subset RBPbase
    """
    if data_dir != '':
        data_dir = data_dir+'/'
    if results_dir != '':
        results_dir = results_dir+'/'
    RBPbase = pd.read_csv(data_dir+rbp_data)
    to_delete = []
    for i, host in enumerate(RBPbase['host']):
        if host != species:
            to_delete.append(i)
        elif len(RBPbase['protein_seq'][i]) < 250:
            to_delete.append(i)
        elif len(RBPbase['protein_seq'][i]) > 1500:
            to_delete.append(i)
    RBPbase = RBPbase.drop(to_delete, axis=0)
    RBPbase = RBPbase.reset_index(drop=True)
    filepieces = rbp_data.split('.')
    RBPbase.to_csv(results_dir+filepieces[0]+'_'+species+'.'+filepieces[1], index=False)
    
    return

def kaptive_python(project_dir, data_dir, database_name, file_name):
    """
    This function is a wrapper for the Kaptive Python file to be executed from another Python script.
    This wrapper runs on a single FASTA file (one genome) and also produces a single FASTA file.
    
    Input:
    - project_directory: the location of kaptive.py (preferrably in the same project folder)
    - data_directory: location of the database and sequence file(s) to loop over
    - database_name: string of the name of the database (.gbk file)
    - file_name: string of the file name (FASTA)
    
    Output:
    - a single fasta file of the locus (single piece or multiple ones) per genome
    """
    cd_command = 'cd ' + project_dir
    
    kaptive_command = 'python kaptive.py -a ' + data_dir + '/' + file_name + ' -k ' + data_dir + '/' + database_name + ' -o ' + data_dir + '/ --no_table'
    command = cd_command + '; ' + kaptive_command
    ssprocess = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ssout, sserr = ssprocess.communicate()
    kaptive_file_name = 'kaptive_results_'+file_name
    
    return kaptive_file_name, ssout, sserr

def compute_loci(klebsiella_genomes, project_dir, data_dir, database_name):
    """
    This function uses kaptive_python to loop over all klebsiella_genomes, construct FASTA files for
    each of them and identify their loci. Importantly, the file_names are later used to construct embeddings,
    so need to be identifiable (accession numbers).
        
    Input:
    - klebsiella_genomes: Pandas DataFrame of genomes w/ following column names:
        'accession', 'sequence' and 'number_of_prophages'
    - project_directory: the location of kaptive.py (preferrably in the same project folder)
    - data_directory: location of the database and sequence file(s) to loop over
    - results_directory: location to store results
    - database_name: string of the name of the database (.gbk file)
    """
    kaptive_file_names = []
    serotypes = []
    pbar = tqdm(total=klebsiella_genomes.shape[0])
    
    # loop over klebsiella_genomes
    for i, genome in enumerate(klebsiella_genomes['sequence']):
        #if klebsiella_genomes['number_of_prophages'][i] > 0: # no filter: rows/columns consistent down the road
        acc = list(klebsiella_genomes['accession'])[i]

        # make FASTA file
        file_name = acc+'.fasta'
        fasta = open(data_dir+'/'+file_name, 'w')
        fasta.write('>'+acc+'\n'+genome+'\n')
        fasta.close()

        # run Kaptive
        kaptive_file, _, _ = kaptive_python(project_dir, data_dir, database_name, file_name)
        kaptive_file_names.append(kaptive_file)

        # process json -> proteins in fasta
        results = json.load(open(data_dir+'/kaptive_results.json'))
        serotypes.append(results[0]['Best match']['Type'])
        protein_results = open(data_dir+'/kaptive_results_proteins_'+acc+'.fasta', 'w')
        for gene in results[0]['Locus genes']:
            try:
                name = gene['Reference']['Product']
            except KeyError:
                name = 'unknown'
            protein = gene['Reference']['Protein sequence']
            protein_results.write('>'+name+'\n'+protein[:-1]+'\n')
        protein_results.close()

        # delete FASTA & temp KAPTIVE files
        os.remove(data_dir+'/'+file_name)
        os.remove(data_dir+'/'+file_name+'.ndb')
        os.remove(data_dir+'/'+file_name+'.not')
        os.remove(data_dir+'/'+file_name+'.ntf')
        os.remove(data_dir+'/'+file_name+'.nto')
        os.remove(data_dir+'/kaptive_results.json')

        # update progress
        pbar.update(1)
            
    pbar.close()
    return kaptive_file_names, serotypes


def RBPbase_fasta_processing(rbp_data, data_dir):
    """
    This function processes the RBP database from a Pandas DataFrame to individual fasta files that can be
    looped over to compute protein embeddings.

    Input:
    - rbp_data: name of the RBP DataFrame (string) in the data_dir, with protein sequences in column 'protein_seq'
    - data_dir: location of the database and where the fasta files will be stored (string)

    Output:
    - fasta files of each of the RBP sequences in the database
    - big fasta file of all sequences together (for pairwise alignments later)
    """
    if data_dir != '':
        data_dir = data_dir+'/'

    rbp_file_names = []
    RBPbase = pd.read_csv(data_dir+rbp_data)
    big_fasta = open(data_dir+rbp_data.split('.')[0]+'.fasta', 'w')
    for i, sequence in enumerate(RBPbase['protein_seq']):
        unique_id = RBPbase['unique_ID'][i]
        rbp_file_names.append(unique_id+'.fasta')
        
        # write individual fasta
        fasta = open(data_dir+unique_id+'.fasta', 'w')
        fasta.write('>'+unique_id+'\n'+sequence+'\n')
        fasta.close()
        
        # write big fasta
        big_fasta.write('>'+unique_id+'\n'+sequence+'\n')
    big_fasta.close()

    return rbp_file_names


def pairwise_alignment_julia(file_name, align_type, project_dir, n_threads='4'):
    """
    Input:
    - file_name: string of path to the FASTA file to loop over
    - align_type: type of alignment to execute ('DNA' or 'protein')
    - project_dir: project directory with julia file in it
    - n_threads: number of threads to use for multithreading (as string; default=4)
    
    Output:
    - a score matrix of pairwise ID%, named file_name + '_score_matrix.txt'

    Remark: first run the alias command once in therminal to enable julia from command line!
    """
    #alias_command = 'sudo ln -fs julia="/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia" /usr/local/bin/julia'
    threads_command = 'export JULIA_NUM_THREADS=' + n_threads
    cd_command = 'cd ' + project_dir
    pw_command = 'julia pairwise_alignment.jl ' + file_name + ' ' + align_type
    command = threads_command + '; ' + cd_command + '; ' + pw_command

    ssprocess = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ssout, sserr = ssprocess.communicate()
    
    return ssout, sserr


def get_kaptive_file_names(klebsiella_genomes):
    """
    This function is a support function to recollect all kaptive file names without
    having to recompute all loci.
    
    Input: klebsiella genomes DataFrame (columns: 'accession', 'sequence' and 'number_of_prophages')
    Output: list kaptive_file_names
    """
    kaptive_file_names = []
    for acc in klebsiella_genomes['accession']:
        file_name = 'kaptive_results_'+acc+'.fasta'
        kaptive_file_names.append(file_name)
    return kaptive_file_names


def kaptive_fasta_processing(kaptive_file_names, data_dir):
    """
    This function processes all the separate bacterial loci FASTA files into one merged FASTA file for 
    further processing (pairwise alignments).
    
    Input:
    - kaptive_file_names: list of fasta filenames of bacterial loci (output of Kaptive)
    - data_dir: location of the sequence file(s) to loop over
    """
    kaptive_fasta_all = data_dir+'/kaptive_results_all.fasta'
    big_fasta = open(kaptive_fasta_all, 'w')
    for name in kaptive_file_names:
        kaptive_id = name.split('.fasta')[0].split('kaptive_results_')[1]
        locus_sequence = ''
        for record in SeqIO.parse(data_dir+'/'+name, 'fasta'):
            locus_sequence += str(record.seq)   
        big_fasta.write('>'+kaptive_id+'\n'+locus_sequence+'\n')
    big_fasta.close()

    return kaptive_fasta_all


def cdhit_est_python(cdhit_path, data_dir, kaptive_file_names, kaptive_fasta, output_file, c=0.90, n=7):
    """
    This function executes CD-HIT-EST (DNA sequences) clustering commands from within Python. 
    To install CD-HIT, do so via conda: conda install -c bioconda cd-hit.
    
    Input:
        - cdhit_path: path to CD-HIT software
        - kaptive_file_names: list of file names for ordering
        - kaptive_fasta: FASTA file with locus sequences
        - output file: path to output (will be one FASTA file and one .clstr file)
        - c: threshold on identity for clustering
        - n: word length (7 for thresholds between 0.88 and 0.9, DNA level)
    """
    
    # perform clustering
    cd_command = 'cd ' + cdhit_path
    cluster_command = './cd-hit-est -i ' + kaptive_fasta + ' -o ' + output_file + ' -c ' + str(c) + ' -n ' + str(n) + ' -d 0'
    command = cd_command + '; ' + cluster_command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    
    # load results and process
    score_matrix = np.zeros((len(kaptive_file_names), len(kaptive_file_names)))
    clusters = open(output_file+'.clstr')
    cluster_iter = 0
    cluster_accessions = []
    for line in clusters.readlines():
        # new cluster
        if line[0] == '>':
            # finish old cluster if not first one
            if (cluster_iter > 0) and (len(cluster_accessions) > 1):
                indices = [kaptive_file_names.index('kaptive_results_'+acc+'.fasta') for acc in cluster_accessions]
                for i in range(len(indices)-1):
                    for j in range(i, len(indices)):
                        if indices[i] != indices[j]:
                            score_matrix[indices[i],indices[j]], score_matrix[indices[j],indices[i]] = c, c
                
            # initiate new cluster
            cluster_accessions = []
            cluster_iter += 1
            
        # in a cluster
        else:
            acc = line.split('>')[1].split('...')[0]
            cluster_accessions.append(acc)
            
    # finish last cluster
    if len(cluster_accessions) > 1:
        indices = [kaptive_file_names.index('kaptive_results_'+acc+'.fasta') for acc in cluster_accessions]
        for i in range(len(indices)-1):
            for j in range(i, len(indices)):
                score_matrix[i,j], score_matrix[j,i] = c, c
    
    # assess identicals
    np.fill_diagonal(score_matrix, 1)
    for i in range(len(kaptive_file_names)-1):
        for j in range(i, len(kaptive_file_names)):
            seq_li = ''; seq_lj = ''
            for record in SeqIO.parse(data_dir+'/'+kaptive_file_names[i], 'fasta'):
                seq_li += str(record.seq)
            for record in SeqIO.parse(data_dir+'/'+kaptive_file_names[j], 'fasta'):
                seq_lj += str(record.seq)
            if seq_li == seq_lj:
                score_matrix[i,j], score_matrix[j,i] = 1, 1
           
    np.savetxt(output_file+'_score_matrix.txt', score_matrix, fmt='%.3f')
            
    return stdout, stderr