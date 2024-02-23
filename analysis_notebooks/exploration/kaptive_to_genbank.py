import os
import subprocess
from Bio import SeqIO, SeqFeature, SeqRecord
from Bio.Alphabet import generic_dna

# Path to Kaptive executable
KAPTIVE_PATH = '/path/to/kaptive'

# Directory containing FASTA files
INPUT_DIR = '/path/to/fasta/files'

# Directory to save GenBank files
OUTPUT_DIR = '/path/to/genbank/files'

def extract_capsule_locus(fasta_file):
    # Run Kaptive on the FASTA file
    cmd = [KAPTIVE_PATH, '--no-run-webserver', '--mlst', '1', '--output-fna', '1', fasta_file]
    subprocess.run(cmd, check=True)

    # Load the Kaptive output file
    output_file = fasta_file + '.out'
    with open(output_file, 'r') as f:
        kaptive_output = f.read()

    # Parse the capsule locus from the Kaptive output
    for line in kaptive_output.split('\n'):
        if line.startswith('#CAPSULE LOCUS'):
            # Extract the capsule locus sequence
            start, end = map(int, line.split()[-2:])
            record = next(SeqIO.parse(fasta_file, 'fasta'))
            locus_seq = record.seq[start-1:end]

            # Extract the gene annotations from the Kaptive output
            features = []
            for feature_line in kaptive_output.split('#GENE')[1:]:
                feature_lines = feature_line.split('\n')
                gene_name = feature_lines[0].strip()
                start, end, strand = map(int, feature_lines[1].strip().split())
                gene_seq = locus_seq[start-1:end]
                feature = SeqFeature.SeqFeature(
                    SeqFeature.FeatureLocation(start, end, strand),
                    type='CDS',
                    id=gene_name,
                    qualifiers={
                        'gene': [gene_name],
                        'product': ['hypothetical protein']  # Replace with actual gene product name
                    }
                )
                features.append(feature)

            # Create a SeqRecord with the capsule locus and gene annotations
            record = SeqRecord.SeqRecord(locus_seq, id=record.id, name=record.name, description=record.description, features=features)
            record.seq.alphabet = generic_dna
            return record

    return None

# Iterate over all the FASTA files in the input directory
for fasta_file in os.listdir(INPUT_DIR):
    if fasta_file.endswith('.fasta'):
        # Extract the capsule locus and gene annotations
        record = extract_capsule_locus(os.path.join(INPUT_DIR, fasta_file))
        if record:
            # Save the locus as a GenBank file
            genbank_file = os.path.join(OUTPUT_DIR, fasta_file.replace('.fasta', '.gb'))
            SeqIO.write(record, genbank_file, 'genbank')
