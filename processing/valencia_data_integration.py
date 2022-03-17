"""
DEPRECATED: SEE phagehostpredict_processing notebook

VALENCIA DATA INTEGRATION

Created on 25/11/21

@author: dimiboeckaerts
"""


#%% 0 - LIBRARIES & DIRECTORIES
# --------------------------------------------------
import re
import json
import math
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from os import listdir
import processing_utils as pu

valencia_dir = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/Valencia_data'
phages_dir = valencia_dir+'/phages_genomes'
klebsiella_dir = valencia_dir+'/klebsiella_genomes'
phanotate_loc = '/opt/homebrew/Caskroom/miniforge/base/envs/ML1/bin/phanotate.py'


#%% 1 - PHANOTATE
# --------------------------------------------------
phage_files = listdir(valencia_dir+'/phages_genomes')
phage_files.remove('.DS_Store')
record = SeqIO.read(phages_dir+'/'+phage_files[0], 'fasta')
bar = tqdm(total=len(phage_files), position=0, leave=True)
name_list = []; gene_list = []

for file in phage_files:
    # access PHANOTATE
    file_dir = phages_dir+'/'+file
    raw_str = phanotate_loc + ' ' + file_dir
    process = subprocess.Popen(raw_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    std_splits = stdout.split(sep=b'\n')
    std_splits = std_splits[2:] #std_splits.pop(0)
    
    # Save and reload TSV
    temp_tab = open('/Users/Dimi/Desktop/phage_results.tsv', 'wb')
    for split in std_splits:
        split = split.replace(b',', b'') # replace commas for pandas compatibility
        temp_tab.write(split + b'\n')
    temp_tab.close()
    results_orfs = pd.read_csv('/Users/Dimi/Desktop/phage_results.tsv', sep='\t', lineterminator='\n', index_col=False)
    
    # fill up lists accordingly
    name = file.split('_')[0]
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
        
    # update progress
    bar.update(1)
bar.close()
#%%        
# MAKE ONE FINAL DATABASE
genebase = pd.DataFrame(list(zip(name_list, gene_list)), columns=['phage_ID', 'gene_sequence'])
genebase.to_csv(valencia_dir+'/phage_genes.csv', index=False)


# %% 2 - RBP DETECT
# --------------------------------------------------
# define paths and press new database
path = '/Users/Dimi/hmmer-3.3.1'
new_pfam = '/Users/dimi/GoogleDrive/PhD/3_PHAGEBASE/32_DATA/RBP_detection/Pfam-A_extended.hmm'
#output, err = pu.hmmpress_python(path, new_pfam)

#%%
genebase = pd.read_csv(valencia_dir+'/phage_genes.csv')
unique_genes = genebase['gene_sequence']

# identify domains in the sequences
valencia_phage_domains = pu.all_domains_scan(path, new_pfam, list(unique_genes))
domaindump = json.dumps(valencia_phage_domains)
domfile = open(valencia_dir+'/valencia_phage_domains.json', 'w')
domfile.write(domaindump)
domfile.close()

#%% define updated domain lists (structs, bindings, chaps) & make predictions
structs = ['Phage_T7_tail', 'Tail_spike_N', 'Prophage_tail', 'BppU_N', 'Mtd_N', 
           'Head_binding', 'DUF3751', 'End_N_terminal', 'phage_tail_N', 'Prophage_tailD1', 
           'DUF2163', 'Phage_fiber_2', 'phage_RBP_N1', 'phage_RBP_N4', 'phage_RBP_N26', 
           'phage_RBP_N28', 'phage_RBP_N34', 'phage_RBP_N45']
bindings = ['Lipase_GDSL_2', 'Pectate_lyase_3', 'gp37_C', 'Beta_helix', 'Gp58', 'End_beta_propel', 
            'End_tail_spike', 'End_beta_barrel', 'PhageP22-tail', 'Phage_spike_2', 
            'gp12-short_mid', 'Collar', 'phage_RBP_C2', 'phage_RBP_C10', 'phage_RBP_C24',
            'phage_RBP_C43', 'phage_RBP_C59', 'phage_RBP_C60', 'phage_RBP_C62', 'phage_RBP_C67',
            'phage_RBP_C79', 'phage_RBP_C97', 'phage_RBP_C111', 'phage_RBP_C115', 'phage_RBP_C120'
            'phage_RBP_C126', 'phage_RBP_C138', 'phage_RBP_C43', 'phage_RBP_C157', 'phage_RBP_C164', 
            'phage_RBP_C175', 'phage_RBP_C180', 'phage_RBP_C205', 'phage_RBP_C217', 'phage_RBP_C220', 
            'phage_RBP_C221', 'phage_RBP_C223', 'phage_RBP_C234', 'phage_RBP_C235', 'phage_RBP_C237',
            'phage_RBP_C249', 'phage_RBP_C259', 'phage_RBP_C267', 'phage_RBP_C271', 'phage_RBP_C277',
            'phage_RBP_C281', 'phage_RBP_C292', 'phage_RBP_C293', 'phage_RBP_C296', 'phage_RBP_C300', 
            'phage_RBP_C301', 'phage_RBP_C319', 'phage_RBP_C320', 'phage_RBP_C321', 'phage_RBP_C326', 
            'phage_RBP_C331', 'phage_RBP_C337', 'phage_RBP_C338', 'phage_RBP_C340']
chaps = ['Peptidase_S74', 'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']
archive = ['Exo_endo_phos', 'NosD', 'SASA', 'Peptidase_M23', 'Phage_fiber']

# identify RBPs based on the scanned domains and the list of known RBP domains
domains_file = open(valencia_dir+'/valencia_phage_domains.json')
valencia_phage_domains = json.load(domains_file)
preds = pu.domain_RBP_predictor(path, new_pfam, valencia_phage_domains, structs, bindings, chaps, archive)
preds_df = pd.DataFrame.from_dict(preds)


# %% 3 - RBPbase_Valencia & merge
# --------------------------------------------------
genebase = pd.read_csv(valencia_dir+'/phage_genes.csv')
indices = [i for i, sequence in enumerate(genebase['gene_sequence']) if sequence in list(preds_df['gene_sequence'])]
RBPs = genebase.iloc[indices]
RBPs.reset_index(drop=True, inplace=True)

#%% define the paths and RBP building blocks (first all N-term, then C-term, then chaps)
N_blocks = ['Phage_T7_tail','Tail_spike_N','Prophage_tail','BppU_N','Mtd_N','Head_binding','DUF3751','End_N_terminal', 
           'phage_tail_N','Prophage_tailD1','DUF2163','Phage_fiber_2','phage_RBP_N1','phage_RBP_N4','phage_RBP_N26',
           'phage_RBP_N28','phage_RBP_N34','phage_RBP_N45']       
C_blocks = ['Lipase_GDSL_2','Pectate_lyase_3','gp37_C','Beta_helix','Gp58','End_beta_propel','End_tail_spike', 
           'End_beta_barrel','PhageP22-tail','Phage_spike_2','gp12-short_mid','Collar', 'phage_RBP_C2', 
           'phage_RBP_C10','phage_RBP_C24','phage_RBP_C43','phage_RBP_C59','phage_RBP_C60','phage_RBP_C62',
            'phage_RBP_C67','phage_RBP_C79','phage_RBP_C97','phage_RBP_C111','phage_RBP_C115','phage_RBP_C120',
            'phage_RBP_C126','phage_RBP_C138','phage_RBP_C143','phage_RBP_C157','phage_RBP_C164','phage_RBP_C175',
            'phage_RBP_C180','phage_RBP_C205','phage_RBP_C217','phage_RBP_C220','phage_RBP_C221','phage_RBP_C223',
            'phage_RBP_C234','phage_RBP_C235','phage_RBP_C237','phage_RBP_C249','phage_RBP_C259','phage_RBP_C267',
            'phage_RBP_C271','phage_RBP_C277','phage_RBP_C281','phage_RBP_C292','phage_RBP_C293','phage_RBP_C296', 
            'phage_RBP_C300','phage_RBP_C301','phage_RBP_C319','phage_RBP_C320','phage_RBP_C321','phage_RBP_C326', 
            'phage_RBP_C331','phage_RBP_C337','phage_RBP_C338','phage_RBP_C340',
           'Peptidase_S74', 'Phage_fiber_C', 'S_tail_recep_bd', 'CBM_4_9', 'DUF1983', 'DUF3672']

bar = tqdm(total=len(RBPs['gene_sequence']), leave=True)
N_list = []; C_list = []
rangeN_list = []; rangeC_list = []
unique_ids = []; proteins = []
for i, sequence in enumerate(RBPs['gene_sequence']):
    N_sequence = []; C_sequence = []
    rangeN_sequence = []; rangeC_sequence = []
    domains, scores, biases, ranges = pu.gene_domain_scan(path, new_pfam, [sequence], threshold=18)
    for j, dom in enumerate(domains):
        OM_score = math.floor(math.log(scores[j], 10)) # order of magnitude
        OM_bias = math.floor(math.log(biases[j]+0.00001, 10))
        if (OM_score > OM_bias) and (dom in N_blocks): # N-terminal block
            N_sequence.append(dom)
            rangeN_sequence.append(ranges[j])
        elif (OM_score > OM_bias) and (dom in C_blocks) and (scores[j] >= 25): # C-terminal block
            C_sequence.append(dom)
            rangeC_sequence.append(ranges[j])
        elif (OM_score > OM_bias) and (dom not in N_blocks) and (dom not in C_blocks): # other block
            if ranges[j][1] <= 200:
                N_sequence.append('other')
                rangeN_sequence.append(ranges[j])
            elif (ranges[j][1] > 200) and (scores[j] >= 25):
                C_sequence.append('other')
                rangeC_sequence.append(ranges[j])
            
    # add to the global list
    N_list.append(N_sequence)
    C_list.append(C_sequence)
    rangeN_list.append(rangeN_sequence)
    rangeC_list.append(rangeC_sequence)

    # get protein sequence
    proteins.append(str(Seq(sequence).translate())[:-1])

    # get unique ID
    check = 0; count = 0
    while check == 0:
        unique = RBPs['phage_ID'][i]+'_RBP'+str(count)
        if unique not in unique_ids:
            unique_ids.append(unique)
            check = 1
        else:
            count += 1

    # update bar
    bar.update(1)
bar.close()

# add to dataframe
RBPsV = pd.DataFrame({'phage_nr': RBPs['phage_ID'], 'host': ['klebsiella_pneumoniae']*len(RBPs['phage_ID']), 
                'host_accession': ['-']*len(RBPs['phage_ID']), 'sequence': RBPs['gene_sequence'], 
                'protein_seq': proteins, 'N_blocks': N_list, 'C_blocks': C_list, 'N_ranges': rangeN_list, 
                'C_ranges': rangeC_list, 'unique_ID': unique_ids})

# save dataframe
RBPsV.to_csv(valencia_dir+'/RBPbaseValencia.csv', index=False)

# %% Integrate with RBPbase
pu.RBPbase_identifiers('RBPbase_250621.csv', '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA')
RBPbaseOriginal = pd.read_csv('/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/RBPbase_250621.csv')
RBPbaseOriginal['origin'] = ['prophage']*RBPbaseOriginal.shape[0]
RBPbaseValencia = pd.read_csv(valencia_dir+'/RBPbaseValencia.csv')
RBPbaseValencia['origin'] = ['valencia']*RBPbaseValencia.shape[0]
RBPbase = pd.concat([RBPbaseOriginal, RBPbaseValencia])
RBPbase.reset_index(drop=True, inplace=True)
RBPbase.to_csv('/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/RBPbase_011221.csv', index=False)


# %% 4 - Bacterial genomes merge
# --------------------------------------------------
# construct Klebsiella Valencia set
klebs_files = listdir(klebsiella_dir+'/fasta_files')
klebs_files.remove('.DS_Store')
ERS_list = []; desc_list = []; genome_list = []; strain_list = []

for file in klebs_files:
    file_dir = klebsiella_dir+'/fasta_files/'+file
    genome = ''
    descr = ''
    for record in SeqIO.parse(file_dir, 'fasta'):
        descr = descr + '+' + record.description
        genome = genome + str(record.seq)

    ERS_list.append(file.split('_')[0])
    desc_list.append(descr[1:])
    genome_list.append(genome)
    strain_list.append(file.split('_')[1].split('.')[0])

klebsiella_valencia = pd.DataFrame({'GI': ['-']*len(klebs_files),'accession': ERS_list,'description':desc_list,
                        'organism': ['Klebsiella pneumoniae']*len(klebs_files),'sequence': genome_list,
                        'sequencing_info': ['-']*len(klebs_files),'strain': strain_list,
                        'number_of_prophages': ['-']*len(klebs_files)})
klebsiella_valencia.to_csv(valencia_dir+'/klebsiellaGenomesValencia.csv', index=False)
# %% Merge with prophage data and keep origin
klebsiellaOriginal = pd.read_csv('/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/Klebsiella_RBP_data/phagebase1_klebsiella_pneumoniae.csv')
klebsiellaValencia = pd.read_csv(valencia_dir+'/klebsiellaGenomesValencia.csv')
klebsiellaOriginal['origin'] = ['prophage']*klebsiellaOriginal.shape[0]
klebsiellaValencia['origin'] = ['valencia']*klebsiellaValencia.shape[0]
klebsiella_genomes = pd.concat([klebsiellaOriginal, klebsiellaValencia])
klebsiella_genomes.reset_index(drop=True, inplace=True)
klebsiella_genomes.to_csv('/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/Klebsiella_RBP_data/klebsiella_genomes_031221.csv', index=False)


# %% 5 - Process interaction matrix
# --------------------------------------------------
IM = pd.read_excel(valencia_dir+'/klebsiella_phage_host_interactions.xlsx', index_col=0, header=0)
IM_klebsiella = list(IM.index)
IM_phages = [phage.replace(' ', '') for phage in IM.columns]
klebsiellaValencia = pd.read_csv(valencia_dir+'/klebsiellaGenomesValencia.csv')
RBPbaseValencia = pd.read_csv(valencia_dir+'/RBPbaseValencia.csv')

row_names = klebsiellaValencia['accession']
column_names = RBPbaseValencia['unique_ID']
interactions = np.zeros((len(row_names), len(column_names)))
for i, row in enumerate(row_names):
    strain = klebsiellaValencia['strain'][i]
    row_index = IM_klebsiella.index(strain) # index in IM
    for j, col in enumerate(column_names):
        phage = RBPbaseValencia['phage_nr'][j]
        col_index = IM_phages.index(phage) # index in IM
        interactions[i,j] = IM.iloc[row_index, col_index]

IMValencia = pd.DataFrame(interactions, index=row_names, columns=column_names)
IMValencia.to_csv(valencia_dir+'/interactionsValencia.csv')

# %%
