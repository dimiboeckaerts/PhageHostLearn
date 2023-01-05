import pytest
import phagehostlearn_processing as phlp

def test_phanotate():
    phlp.phanotate_processing(general_path, phage_genomes_path, phanotate_path, data_suffix='', test=True)
    # assert the .tsv file if it is correct

# PHANOTATE: check if only fasta files in folder

# PhageRBPdetect: check file existence of pressed HMM database -> does pressing work?

# Kaptive: check if only FASTA files in folder