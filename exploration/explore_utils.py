"""
Created on Mon Jan 11 10:36:31 2021

@author: dimiboeckaerts
"""

# 0 - LIBRARIES
# --------------------------------------------------
import time
import urllib
import os
import math
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
from Bio import SeqIO
from Bio import Entrez
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SearchIO import HmmerIO
import matplotlib.pyplot as plt
from Bio.Blast import NCBIWWW, NCBIXML
from plots2 import CircosPlot



# 1 - FUNCTIONS RBP DETECTION
# --------------------------------------------------
"""
Remarks:
    - we can use hmmscan on one big sequences file, but then that might complicate
        parsing the output easily, so we'll do one sequence at the time. For 
        this reason, I've implemented two functions (single_hmmscan and hmmscan).
"""

def sequence_split(fasta_file, start=0, end=200):
    # get splits & save in new FASTA file
    temp_fasta = open(fasta_file+'N_term.fasta', 'w')
    for sequence in SeqIO.parse(fasta_file, 'fasta'):
        split = str(sequence.seq)[start:end]
        temp_fasta.write('>'+sequence.id+'\n'+split+'\n')
    temp_fasta.close()
    print('sequences split.')
    
    return


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
                if (hit.bitscore >= 18):
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


def graph_dictionary(dict, save=False):
    """
    This function makes a network plot of a dictionary. We want to connect
    N-terminal domains to C-terminal, which can be plotted as a bipartite 
    graph. For this, we need to add nodes to a specific set (argument: bipartite
    = 0 or 1).
    
    Info: 
    https://www.python-course.eu/networkx.php
    https://networkx.org/documentation/stable/reference/algorithms/bipartite.html
    
    Input:
        - dict: the dictionary to be plotted
        - (optional) save: handle to file to save the fig
    """
    
    # define the graph
    G=nx.Graph()
    
    # add nodes
    dict_values = list(set([item for sub in list(dict.values()) for item in sub
                            if item not in list(dict.keys())]))
    G.add_nodes_from(list(dict.keys())) # N-term
    G.add_nodes_from(dict_values) # C-term
    
    # make tuples for edges
    edge_list = []
    for key in dict.keys():
        for value in list(dict[key]):
            if (key != value):
                edge_list.append((key, value))
    G.add_edges_from(edge_list)

    # get colors
    color_map = []
    node_count = 0
    for node in G:
        node_count += 1
        if node in dict.keys():
            color_map.append('#E6AA68') # N-terminals in orange
        else:
            color_map.append('#5DA9E9') # C-terminals in blue
    
    # plot the graph
    size = max(node_count/6, 8)
    plt.figure(figsize=(round(size*4/3),size))
    plt.title('Domains graph')
    pos = nx.spring_layout(G, scale=5, k=5/math.sqrt(G.order()))
    nx.draw(G, pos=pos, with_labels=True, node_color=color_map)
    plt.tight_layout()
    
    #top = nx.bipartite.sets(G)[0]
    #pos = nx.bipartite_layout(G, top)
    #for p in pos:  # raise text positions
    #    pos[p][1] += 0.05
    #nx.draw_networkx_labels(G, pos)
    
    if save != False:
        plt.savefig(save, dpi=300)
        
    #plt.show()
    return


def define_groups(matrix, threshold):
    # make groups of sequences under identity threshold
    groups_dict = {}
    groups_list = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            if (matrix[i,j] >= threshold) & (i not in groups_list) & (j not in groups_list) & (i not in groups_dict.keys()):
                groups_dict[i] = [j]
                groups_list.append(j)
            elif (matrix[i,j] >= threshold) & (i not in groups_list) & (j not in groups_list) & (i in groups_dict.keys()):
                groups_dict[i].append(j)
                groups_list.append(j)

    # assign group numbers to clusters
    groups_array = np.zeros(matrix.shape[0])
    groups_index = 1
    for key in groups_dict.keys():
        groups_array[key] = groups_index
        for item in groups_dict[key]:
            groups_array[item] = groups_index
        groups_index += 1

    # assign group numbers to leftover sequences not in any cluster
    for i,item in enumerate(groups_array):
        if item == 0:
            groups_array[i] = groups_index
            groups_index += 1
    
    return groups_array


def domain_cluster_graph(hmm_path, pfam_db, sequences, filename='domains_circos.png'):
    """
    Construct a Circosplot for the RBP domains and their combinations 
    that occur within a certain group of sequences.
    
    Dependencies: gene_domain_scan, numpy, math
    
    Inputs:
    - hmm_path to HMMER software
    - pfam_db: file link to Pfam domains database (string)
    - list of sequences (DNA level)
    - filename: directory and name to save plot (e.g. '/mydir/filename.png')
    
    Output: Circosplot
    """
    # define dictionaries for counting
    count_dictN = {'Phage_T7_tail':0,'Tail_spike_N':0,'Prophage_tail':0,'BppU_N':0,'Mtd_N':0,'Head_binding':0,
                    'DUF3751':0,'End_N_terminal':0,'phage_tail_N':0,'Prophage_tailD1':0,'DUF2163':0,'Phage_fiber_2':0,
                    'phage_RBP_N1':0,'phage_RBP_N4':0,'phage_RBP_N28a':0,'phage_RBP_N28b':0,'phage_RBP_N34':0,
                    'phage_RBP_N45':0, 'other_N':0, 'unknown_N':0}
    count_dictC = {'Lipase_GDSL_2':0,'Pectate_lyase_3':0,'gp37_C':0,'Beta_helix':0,'Gp58':0,'End_beta_propel':0,
                    'End_tail_spike':0, 'End_beta_barrel':0,'PhageP22-tail':0,'Phage_spike_2':0,'gp12-short_mid':0,
                    'Collar':0,'phage_RBP_C2':0,'phage_RBP_C10':0,'phage_RBP_C24':0,'phage_RBP_C43':0,'phage_RBP_C59':0,
                    'phage_RBP_C60':0,'phage_RBP_C62':0,'phage_RBP_C67':0,'phage_RBP_C79':0,'phage_RBP_C97':0,
                    'phage_RBP_C111':0,'phage_RBP_C115':0,'phage_RBP_C120':0,'phage_RBP_C126':0,'phage_RBP_C138':0,
                    'phage_RBP_C143':0,'phage_RBP_C157':0,'phage_RBP_C164':0,'phage_RBP_C175':0,'phage_RBP_C180':0,
                    'phage_RBP_C205':0,'phage_RBP_C217':0,'phage_RBP_C220':0,'phage_RBP_C221':0,'phage_RBP_C223':0,
                    'phage_RBP_C234':0,'phage_RBP_C235':0,'phage_RBP_C237':0,'phage_RBP_C249':0,'phage_RBP_C259':0,
                    'phage_RBP_C267':0,'phage_RBP_C271':0,'phage_RBP_C277':0,'phage_RBP_C281':0,'phage_RBP_C292':0,
                    'phage_RBP_C293':0,'phage_RBP_C296':0,'phage_RBP_C300':0,'phage_RBP_C301':0,'phage_RBP_C319':0,
                    'phage_RBP_C320':0,'phage_RBP_C321':0,'phage_RBP_C326':0,'phage_RBP_C331':0, 'phage_RBP_C337':0,
                    'phage_RBP_C338':0,'phage_RBP_C340':0,'Peptidase_S74':0,'Phage_fiber_C':0,'S_tail_recep_bd':0,
                    'CBM_4_9':0, 'DUF1983':0, 'DUF3672':0, 'other_C':0, 'unknown_C':0}
    total_length = len(count_dictN.keys())+len(count_dictC.keys())
    adjacency_matrix = np.zeros((total_length, total_length))
    
    # loop over the sequences and fill up the dictionaries and adjacency matrix
    for sequence in tqdm(sequences):
        domains, scores, biases, ranges = gene_domain_scan(hmm_path, pfam_db, [sequence])
        unknown_N = 0
        unknown_C = 0
        list_N = []
        list_C = []
        # loop over hits and count the significants in the correct key of the dictionary
        for j, dom in enumerate(domains):
            OM_score = math.floor(math.log(scores[j], 10))
            OM_bias = math.floor(math.log(biases[j]+0.00001, 10))
            
            # add known RBP domains
            if (OM_score > OM_bias) and (dom in count_dictN.keys()):
                count_dictN[dom] += 1
                unknown_N += 1
                list_N.append(dom)
            elif (OM_score > OM_bias) and (dom in count_dictC.keys()):
                count_dictC[dom] += 1
                unknown_C += 1
                list_C.append(dom)

            # add other_N, other_C domain
            elif (OM_score > OM_bias) and (dom not in count_dictN.keys()) and (ranges[j][1] < 200):
                count_dictN['other_N'] += 1
                unknown_N += 1
                list_N.append('other_N')
            elif (OM_score > OM_bias) and (dom not in count_dictC.keys()) and (ranges[j][1] >= 200):
                count_dictC['other_C'] += 1
                unknown_C += 1
                list_C.append('other_C')
        # add unknowns
        if unknown_N == 0:
            count_dictN['unknown_N'] += 1
            list_N.append('unknown_N')
        elif unknown_C == 0:
            count_dictC['unknown_C'] += 1   
            list_C.append('unknown_C')
        
        # fill up adjacency matrix
        list_N = list(set(list_N))
        list_C = list(set(list_C))
        for domN in list_N:
            indexN = list(count_dictN.keys()).index(domN)
            for domC in list_C:
                indexC = list(count_dictC.keys()).index(domC)
                adjacency_matrix[indexN, indexC] += 1

    # make a graph
    G=nx.Graph(name='RBP domain combinations')

    # add nodes & fill color_map
    color_map = []
    for domN in count_dictN.keys():
        if count_dictN[domN] > 0:
            G.add_nodes_from([(domN, {'class':'N-terminal domain'})])
            color_map.append('#508AA8')
    for domC in count_dictC.keys():
        if count_dictC[domC] > 0:
            G.add_nodes_from([(domC, {'class':'C-terminal domain'})])
            color_map.append('#FFCF9C')

     # rescale adjacency matrix
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            adjacency_matrix[i,j] = (100-len(color_map))*((adjacency_matrix[i,j] - np.min(adjacency_matrix)) / (np.max(adjacency_matrix)-np.min(adjacency_matrix)))
    
    # add appropriate edges
    for domN in count_dictN.keys():
        indexN = list(count_dictN.keys()).index(domN)
        for domC in count_dictC.keys():
            indexC = list(count_dictC.keys()).index(domC)
            weight = adjacency_matrix[indexN, indexC]
            if weight > 0:
                G.add_weighted_edges_from([(domN, domC, weight)])

    # plot the graph as circosplot
    if len(color_map) > 30:
        c = CircosPlot(graph=G, figsize=(10,10), node_labels=True, edge_width='weight', node_colors=color_map,
            node_label_layout='rotation', group_legend=True)
    else:
        c = CircosPlot(graph=G, figsize=(10,10), node_labels=True, edge_width='weight', node_colors=color_map, group_legend=True)
    c.draw()
    plt.savefig(filename, dpi=400)
    
    return


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

def cdhit_est_python(cdhit_path, input_file, output_file, c=0.90, n=7):
    """
    This function executes CD-HIT-EST (DNA sequences) clustering commands from within Python. 
    To install CD-HIT, do so via conda: conda install -c bioconda cd-hit.
    
    Input:
        - cdhit_path: path to CD-HIT software
        - input_file: FASTA file with protein sequences
        - output file: path to output (will be one FASTA file and one .clstr file)
        - c: threshold on identity for clustering
        - n: word length (7 for thresholds between 0.88 and 0.9, DNA level)
    """
    
    # perform clustering
    cd_command = 'cd ' + cdhit_path
    cluster_command = './cd-hit-est -i ' + input_file + ' -o ' + output_file + ' -c ' + str(c) + ' -n ' + str(n) + ' -d 0'
    command = cd_command + '; ' + cluster_command
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
        - input_file: FASTA file with protein/DNA sequences
        - output file: path to output (will be one FASTA file and one .clstr file)
        - c: threshold on identity for clustering
    """
    cd_str = 'cd ' + cdhit_path # change directory
    raw_str = './psi-cd-hit.pl -i ' + input_file + ' -o ' + output_file + ' -c ' + str(c)
    command = cd_str+'; '+ raw_str
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    
    return stdout, stderr

def clustalo_python(clustalo_path, input_file, output_file, out_format='fa'):
    """
    This function executes the basic command to run a local Clustal Omega MSA.
    You need to install Clustal Omega locally first, see http://www.clustal.org/omega/.
    The basic command is: clustalo -i my-in-seqs.fa -o my-out-seqs.fa -v
    
    Dependencies: subprocess
    
    Input:
        - clustalo_path: path to clustalo software
        - input_file: FASTA file with (protein) sequences
        - output_file: path to output file for MSA
        - out_format: format of the output (fa[sta],clu[stal],msf,phy[lip],selex,
                        st[ockholm],vie[nna]; default: fasta) as string
        
    Output: stdout and stderr are the output from the terminal. Results are saved 
            as given output_file.
    """
    
    cd_str = 'cd ' + clustalo_path # change dir
    raw_str = './clustalo -i ' + input_file + ' -o ' + output_file + ' -v --outfmt ' + out_format
    command = cd_str+'; '+ raw_str
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    
    return stdout, stderr