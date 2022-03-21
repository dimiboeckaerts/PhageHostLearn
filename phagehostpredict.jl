# --------------------------------------------------
# PhageHostPredict (Klebsiella)
#
# An AI-based Phage-Host interaction predictor framework with receptors and receptor-binding proteins at its core. 
# This particular PhageHostPredict is for *Klebsiella pneumoniae* related phages.
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
using JSON
using FASTX
using BioAlignments
using Random
using Plots
using StatsBase
#using Pluto

general_dir = "/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/Valencia_data" # general directory
results_dir = "/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/43_RESULTS/models"
data_suffix = "Valencia" # choose a suffix for the created data files


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


# PhageHostLearning: computing embeddings
# --------------------------------------------------
# load data and set names
RBPbase = DataFrame(CSV.File(general_dir*"/RBPbase"*data_suffix*".csv"))
LociBase = JSON.pasefile(general_dir*"/Locibase"*data_suffix*".json")
IM = DataFrame(CSV.File(general_dir*"/interactions_mono"*data_suffix*".csv"))
interaction_matrix = Matrix(IM[1:end, 2:end])
loci_names = IM.accession
rbp_names = names(IM)

# define protein alphabet
alphabet = "GAVLIFPSTYCMKRHWDENQX"
basis = Dict(c=>BipolarHDV() for c in alphabet)

# compute loci embeddings w/ proteins (multi-instance)
loci_embeddings = Array{BipolarHDV}(undef, length(LociBase))
for (i, (name, proteins)) in enumerate(LociBase)
    # bind within one sequence, then aggregate the different sequences
    protein_hdvs = [sequence_embedding(string(sequence), basis, 3) for sequence in proteins]
    loci_hdv = HyperdimensionalComputing.aggregate(protein_hdvs)
    loci_embeddings[i] = loci_hdv
end

# compute rbp embeddings
rbp_embeddings = Array{BipolarHDV}(undef, length(RBPbase.ProteinSeq))
for (i, sequence) in enumerate(RBPbase.ProteinSeq)
    rbp_embeddings[i] = sequence_embedding(string(sequence), basis, 3)
end


# PhageHostLearning: first test
# --------------------------------------------------
# compute rbp-receptor signatures (bind)
signatures_pos = []
signatures_neg = []
for (i, loci_embedding) in enumerate(loci_embeddings)
    for (j, rbp_embedding) in enumerate(rbp_embeddings)
        if isequal(interaction_matrix[i,j], 1)
            push!(signatures_pos, HyperdimensionalComputing.bind([loci_embedding, rbp_embedding]))
        elseif isequal(interaction_matrix[i,j], 0)
            push!(signatures_neg, HyperdimensionalComputing.bind([loci_embedding, rbp_embedding]))
        end
    end
end
signatures_pos = convert(Array{BipolarHDV}, signatures_pos)
signatures_neg = convert(Array{BipolarHDV}, signatures_neg)
println("pos: ", length(signatures_pos), " neg: ", length(signatures_neg))

# shuffle and split in train-test
sign_pos = shuffle(signatures_pos)
sign_neg = shuffle(signatures_neg)
cutoff_pos = Int(round(length(signatures_pos)*0.75))
cutoff_neg = Int(round(length(signatures_neg)*0.75))
training_pos = sign_pos[1:cutoff_pos]
training_neg = sign_neg[1:cutoff_neg]
testing_pos = sign_pos[cutoff_pos+1:end]
testing_neg = sign_neg[cutoff_neg+1:end]

# train classes (aggregate)
training_pos_agg = HyperdimensionalComputing.aggregate(training_pos)
training_neg_agg = HyperdimensionalComputing.aggregate(training_neg)

# compute distances (Cosine sim for RealHDVs)
dist_pos_test = [cos_sim(training_pos_agg, x) for x in testing_pos]
dist_neg_test = [cos_sim(training_pos_agg, x) for x in testing_neg]

# make plots
histogram(dist_pos_test, label="positive test", alpha=0.7, legend=:topleft, nbins=40)
histogram!(dist_neg_test, label="negative test", alpha=0.7, nbins=30)

plot(sort!(dist_pos_test), label="positive test", alpha=0.7, legend=:bottomright, xlabel="rank", ylabel="Cosine sim")
plot!(sort!(dist_neg_test), label="negative test", alpha=0.7)


# PhageHostLearning: cross-validation
# --------------------------------------------------
"""
Here, we perform a 10-fold CV over the loci, just like we do to evaluate the
binary classifiers in Python.
"""
loci_known = [x for x in range(1, length=length(LociBase)) 
                if (any(isequal.(interaction_matrix[x,:], 0))) 
                    || (any(isequal.(interaction_matrix[x,:], 1)))]

# shuffle loci
loci_shuffle = shuffle(loci_known)

# divide into 10 groups
group_size = div(length(loci_shuffle), 10) + 1
get_groups(x, n) = [x[i:min(i+n-1,length(x))] for i in 1:n:length(x)]
loci_groups = get_groups(loci_shuffle, group_size)

# loop over groups
loci_nr = []; rbp_nr = []; scores = []; scores_pos = []; labels = []
for group in loci_groups
    # compute signatures for training and testing parts (group = test)
    signatures_train_pos = []
    signatures_train_neg = []
    signatures_test = []
    for (i, loci_embedding) in enumerate(loci_embeddings)
        for (j, rbp_embedding) in enumerate(rbp_embeddings)
            # training pos interaction
            if isequal(interaction_matrix[i,j], 1) && i ∉ group
                push!(signatures_train_pos, HyperdimensionalComputing.bind([loci_embedding, rbp_embedding]))
            # training neg interaction
            elseif isequal(interaction_matrix[i,j], 0) && i ∉ group
                push!(signatures_train_neg, HyperdimensionalComputing.bind([loci_embedding, rbp_embedding]))
            # test interaction
            elseif isequal(interaction_matrix[i,j], 1) && i in group
                push!(signatures_test, HyperdimensionalComputing.bind([loci_embedding, rbp_embedding]))
                push!(loci_nr, i-1) # -1 to cope with indexing python
                push!(rbp_nr, j-1)
                push!(labels, interaction_matrix[i,j])
            elseif isequal(interaction_matrix[i,j], 0) && i in group
                push!(signatures_test, HyperdimensionalComputing.bind([loci_embedding, rbp_embedding]))
                push!(loci_nr, i-1)
                push!(rbp_nr, j-1)
                push!(labels, interaction_matrix[i,j])
            end
        end
    end

    # convert signatures
    signatures_train_pos = convert(Array{BipolarHDV}, signatures_train_pos)
    signatures_train_neg = convert(Array{BipolarHDV}, signatures_train_neg)
    signatures_test = convert(Array{BipolarHDV}, signatures_test)
    println("train size:", length(signatures_train_pos)+length(signatures_train_neg))
    println("test size:", length(signatures_test))
    
    # aggregate training signatures
    signatures_pos_agg = HyperdimensionalComputing.aggregate(signatures_train_pos)
    signatures_neg_agg = HyperdimensionalComputing.aggregate(signatures_train_neg)

    # compute distance/similarity to test signatures
    for test in signatures_test
        score_pos_agg = cos_sim(signatures_pos_agg, test)
        score_neg_agg = cos_sim(signatures_neg_agg, test)
        push!(scores, score_pos_agg/score_neg_agg) # > 1 then pos, < 1 then neg
        push!(scores_pos, score_pos_agg)
    end
end

# results pos vs. neg
results = DataFrame(locus=loci_nr, rbps=rbp_nr, scores=scores, label=labels)
CSV.write(results_dir*"/results_HDC_grouped10CV_alldata.csv", results)

# results pos only
results = DataFrame(locus=loci_nr, rbps=rbp_nr, scores=scores_pos, label=labels)
CSV.write(results_dir*"/results_HDCpos_grouped10CV_alldata.csv", results)


# examine scores
histogram(scores, xlabel="score", ylabel="count")
sum(scores .< 1)/length(scores)
"""
NOTES

* scores
We notice that only 0.4% of all scores are below 1. A score of 1 signals that the
similarities toward positive and negative are equal. So everything below 1 should below
a negative (as we have constructed the ration with pos in numerator). Only 0.4% is below
that threshold of 1, so does that make sense? No. Because we would actually expect most
of the samples (most are negative), to be below 1...
"""
