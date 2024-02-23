# --------------------------------------------------
# PAIRWISE BLAST IN JULIA
#
# @author: dimiboeckaerts
# --------------------------------------------------

#using FASTX
using BioSequences
using DelimitedFiles
using LinearAlgebra
using ProgressMeter
using BioTools.BLAST

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

function pwblast(args)
    """
    Function that does pairwise blast of DNA/protein sequences in given FASTA file
    and returns a score matrix.
    """

    # parse the file & initiate variables
    file = args[1]
    align_type = args[2]
    thres = parse(Int64, args[3])
    sequence_list = file_to_array(file)
    score_matrix = zeros(length(sequence_list), length(sequence_list))
    p = Progress(Int64(round((length(sequence_list)^2)/2, digits=0)))

    # do pairwise alignments
    for i in 1:length(sequence_list)
        Threads.@threads for j in i:length(sequence_list)
            res = align_type == "DNA" ? blastn(sequence_list[i], sequence_list[j], ["-perc_identity", thres]) : blastp(sequence_list[i], sequence_list[j], ["-perc_identity", thres])
            if length(res) > 0
                score_matrix[i,j] = thres
            end
            next!(p)
        end
    end

    score_matrix = (score_matrix + score_matrix') 
    score_matrix = score_matrix - Diagonal(score_matrix)/2
    writedlm(file * "_score_matrix.txt", score_matrix)
end

pwblast(ARGS)