# --------------------------------------------------
# HYPERDIMENSIONAL COMPUTING IN JULIA: PHAGE-HOST PREDICTIONS
#
# @author: dimiboeckaerts
# --------------------------------------------------

# LIBRARIES & DIRECTORIES
# --------------------------------------------------
push!(LOAD_PATH, "/Users/dimi/Documents/GitHub/HyperdimensionalComputing.jl/src/")
using HyperdimensionalComputing
using DataFrames
using ProgressMeter
using CSV
using FASTX
using BioAlignments
#using Pluto

data_dir = "/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/klebsiella_RBP_data"

# FUNCTIONS
# --------------------------------------------------
function file_to_array(file)
    """
    Function that reads a FASTA file and puts its sequences in an array.
    """
    sequences = []
    reader = FASTA.Reader(open(file, "r"))
    for record in reader
        seq = FASTA.sequence(record)
        push!(sequences, seq)
    end
    return sequences
end


# PhageHostLearning
# --------------------------------------------------
# load data and set names
IM = DataFrame(CSV.File(data_dir*"/interactions_klebsiella_mono.csv"))
interaction_matrix = Matrix(IM[1:end, 2:end])
rbps = DataFrame(CSV.File(data_dir*"/RBPbase_031221_klebsiella_pneumoniae.csv"))
loci_names = IM.accession
rbp_names = names(IM)

loci_sequences = Dict(accession=>
                file_to_array(data_dir*"/kaptive_results_proteins_"*accession*".fasta")
                for accession in loci_names)
rbp_sequences = rbps.protein_seq

# define protein alphabet
alphabet = "GAVLIFPSTYCMKRHWDENQX"
basis = Dict(c=>RealHDV() for c in alphabet)


# compute loci embeddings w/ proteins
loci_embeddings = Array{RealHDV}(undef, length(loci_sequences))
for (i, (name, proteins)) in enumerate(loci_sequences)
    # bind within one sequence, then aggregate the different sequences
    protein_hdvs = [sequence_embedding(string(sequence), basis, 3) for sequence in proteins]
    loci_hdv = HyperdimensionalComputing.aggregate(protein_hdvs)
    loci_embeddings[i] = loci_hdv
end

# compute rbp embeddings
rbp_embeddings = Array{RealHDV}(undef, length(rbp_sequences))
for (i, sequence) in enumerate(rbp_sequences)
    rbp_embeddings[i] = sequence_embedding(string(sequence), basis, 3)
end

# compute rbp-receptor signatures (bind)
signatures_pos = []
signatures_neg = []
for (i, loci_embedding) in enumerate(loci_embeddings)
    for (j, rbp_embedding) in enumerate(rbp_embeddings)
        if isequal(interaction_matrix[i,j], 1)
            push!(signatures_pos, bind([loci_embedding, rbp_embedding]))
        elseif isequal(interaction_matrix[i,j], 0)
            push!(signatures_neg, bind([loci_embedding, rbp_embedding]))
        end
    end
end
println("pos: ", length(signatures_pos), " neg: ", length(signatures_neg))


# shuffle and split in train-test
sign_pos = shuffle(signatures_pos)
sign_neg = shuffle(signatures_neg)
cutoff_pos = Int(round(length(signatures_pos)*0.75))
cutoff_neg = Int(round(length(signatures_neg)*0.75))
training_pos = sign_p[1:cutoff_pos]


# train classes (aggregate)


# train layer 1
layer1 = train(labels, loci_embeddings)
retrain!(layer1, labels, loci_embeddings, niters=10)
