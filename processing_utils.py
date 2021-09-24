"""
Created on 07/09/21

@author: dimiboeckaerts

PROCESSING UTILS FOR THE RBP-R PREDICTION FRAMEWORK
"""

# 0 - LIBRARIES
# --------------------------------------------------
import subprocess
from tqdm import tqdm

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

def locus_embedding
