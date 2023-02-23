# --------------------------------------------------
# HD VECTORIZATION IN JULIA
#
# @author: dimiboeckaerts
# --------------------------------------------------

push!(LOAD_PATH, "./HyperdimensionalComputing/src/")
using CSV
using JSON
using DataFrames
using DelimitedFiles
using HyperdimensionalComputing

function hd_vectorization(sequence, tokens; layers=3, k=(6,6,6))
    """
    Convolutional sequence vectorization with hyperdimensional vectors.
    
    Inputs:
    - sequence: the sequence to embed
    - tokens: the starting tokens to begin from, a dictionary {token->hyperdimensional_vector}
    - layers: how many convolutional layers to embed with
    - k: the sliding window for each of the convolution, should correspond to # layers
    
    Output: 
    a vector that represents the 'embedded' sequence
    
    To do: add padding and strides? add check: min length sequence.
    """
    # preprocess the tokens (layer 0)
    sequence_hdvs = [tokens[item] for item in sequence]
    
    # do convolutions
    for layer in 1:layers
        layer_hdvs = []
        layer_k = k[layer]
        for i in 1:length(sequence_hdvs)-layer_k+1
            # select sliding window
            window = sequence_hdvs[i:i+layer_k-1]
            # shift the hdvs
            shifted_hdvs = [circshift(window[j], layer_k-j) for (j, item) in enumerate(window)]
            # bind the hdvs
            push!(layer_hdvs, HyperdimensionalComputing.bind(shifted_hdvs))
        end
        # update sequence hdvs
        sequence_hdvs = layer_hdvs
    end
    return HyperdimensionalComputing.aggregate(convert(Vector{BipolarHDV}, sequence_hdvs))
end

function compute_hdc_rep(args)
    # parse arguments
    path = args[1] # general or test path depending on the mode
    suffix = args[2] # general or test suffix depending on the mode
    locibase_path = args[3] # provide full path to general data folder
    rbpbase_path = args[4] # provide the used data suffix for the data
    mode = args[5] # 'train' or 'test'. Test mode doesn't use an IM.

    # load data
    println("Loading data...")
    LociBase = JSON.parsefile(locibase_path)
    RBPbase =  DataFrame(CSV.File(rbpbase_path))
    if mode == "train"
        IM = DataFrame(CSV.File(path*"/phage_host_interactions"*suffix*".csv"))
        rename!(IM,:Column1 => :Host)
        interaction_matrix = Matrix(IM[1:end, 2:end])
    end

    # create alphabet
    alphabet = "GAVLIFPSTYCMKRHWDENQX"
    tokens = Dict(c=>BipolarHDV() for c in alphabet)

    # compute loci embeddings w/ proteins (multi-instance)
    println("Computing loci representations...")
    loci_embeddings = Array{BipolarHDV}(undef, length(LociBase))
    for (i, (name, proteins)) in enumerate(LociBase)
        # bind within one sequence, then aggregate the different sequences
        protein_hdvs = [hd_vectorization(string(sequence), tokens, layers=3, k=(6,6,6)) for sequence in proteins]
        loci_hdv = HyperdimensionalComputing.aggregate(protein_hdvs)
        loci_embeddings[i] = loci_hdv
    end

    # compute multi-rbp embeddings
    println("Computing RBP representations...")
    rbp_embeddings = Array{BipolarHDV}(undef, length(unique(RBPbase.phage_ID)))
    for (i, phageid) in enumerate(unique(RBPbase.phage_ID))
        subset = filter(row -> row.phage_ID == phageid, RBPbase)
        protein_hdvs = [hd_vectorization(string(sequence), tokens, layers=3, k=(6,6,6)) for sequence in subset.protein_sequence]
        multirbp_hdv = HyperdimensionalComputing.aggregate(protein_hdvs)
        rbp_embeddings[i] = multirbp_hdv
    end

    # compute sigatures for loci x RBP embeddings by BINDING
    features_bind = []
    groups_loci = []
    groups_phage = []
    pairs = []
    for (i, accession) in enumerate(collect(keys(LociBase)))
        for (j, phage_id) in enumerate(unique(RBPbase.phage_ID))
            if mode == "train" # only compute sigs for known interactions
                subset = filter(row -> row.Host == accession, IM)
                interaction = subset[!, phage_id][1]
                if isequal(interaction, 1) || isequal(interaction, 0)
                    signature = HyperdimensionalComputing.bind([loci_embeddings[i], rbp_embeddings[j]])
                    push!(features_bind, signature)
                    push!(groups_loci, i)
                    push!(groups_phage, j)
                    push!(pairs, (accession, String(phage_id)))
                end
            elseif mode == "test" # compute all signatures
                signature = HyperdimensionalComputing.bind([loci_embeddings[i], rbp_embeddings[j]])
                push!(features_bind, signature)
                push!(groups_loci, i)
                push!(groups_phage, j)
                push!(pairs, (accession, String(phage_id)))
            end
        end
    end

    # put the signatures in a matrix for sklearn
    features_b = zeros(Int64, length(features_bind), 10000)
    for i in range(1, length=length(features_bind))
        features_b[i,:] = features_bind[i]
    end

    # write output
    full_mat = hcat(pairs, features_b)
    writedlm(path*"/hdc_features"*suffix*".txt", full_mat)
    println("Done!")
end

compute_hdc_rep(ARGS)

# test
#pwalign(["/Users/Dimi/Desktop/kaptive_test.fasta", "DNA"])

