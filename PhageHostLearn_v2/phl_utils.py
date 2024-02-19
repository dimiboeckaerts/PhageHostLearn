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
import subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

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

def phagerbpdetect(path, suffix=''):
    """
    This function runs the RBPdetect v3 model on the phage proteins
    Inputs:
    - path: the path to the folder containing the phage proteins
    Output: rbps.csv 
    """
    # initiation the model
    model_path = path+'/RBPdetect_v3_ESMfineT33'
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

def kaptive_py(files, path, kaptive_path, suffix=''):
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
        db_path = path+'/Klebsiella_k_locus_primary_reference.gbk'
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
                loci_names_list.append(accessions[i]+'_'+count)
                proteins_list.append(protein)
                count += 1
        loci_sequence = ''
        for record in SeqIO.parse(path+'/results'+suffix+'_'+accessions[i]+'.fasta', 'fasta'):
            loci_sequence = loci_sequence + str(record.seq)
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
    model.eval()  # disables dropout for deterministic results

    # loop over data and embed (batch size = 1)
    proteins = pd.read_csv(path+'/'+file)
    bar = tqdm(total=len(proteins['protein_sequence']), position=0, leave=True)
    sequence_representations = []
    for i, sequence in enumerate(proteins['protein_sequence']):
        data = [(proteins['protein_ID'][i], sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
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

# --------------------------------------------------

def process_phages(files, path, phanotate_path, suffix=''):
    """
    Process the phage genomes with Phanotate and PhageRBPdetect v3.
    Outputs: phage_genes.csv, rbps.csv
    """
    # collect the files
    files = [file for file in os.listdir(path+'/phage_genomes') if (file.endswith('.fasta')) or (file.endswith('.fna'))]
    # run Phanotate
    phanotate_py(files, path, phanotate_path, suffix)
    # run PhageRBPdetect v3
    phagerbpdetect(path, suffix)
    return

def process_bacteria(files, path, kaptive_path, suffix=''):
    """
    Process the bacterial genomes with Kaptive.
    Outputs: locis.csv, loci_summary.csv
    """
    # collect the files
    files = [file for file in os.listdir(path+'/bacterial_genomes') if (file.endswith('.fasta')) or (file.endswith('.fna'))]
    # run Kaptive
    kaptive_py(files, path, kaptive_path, suffix)
    return

def construct_trainingframe(rbp_file, loci_file, path, suffix=''):
    """
    """
    # compute feature representations (compute for all proteins)

    # construct the training dataframe (here make averages)