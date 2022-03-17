# --------------------------------------------------
# PAIRWISE ALIGNMENTS IN JULIA
#
# @author: dimiboeckaerts
# --------------------------------------------------

using FASTX
#using BioSequences
using BioAlignments
using DelimitedFiles
using LinearAlgebra
using ProgressMeter

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

function pwalign(args)
    """
    Function that does pairwise alignments of sequences in given FASTA file
    and returns a score matrix.
    """

    # parse the file & initiate variables
    file = args[1]
    align_type = args[2]
    sequence_list = file_to_array(file)
    score_matrix = zeros(length(sequence_list), length(sequence_list))
    score_model = align_type == "DNA" ? AffineGapScoreModel(EDNAFULL, gap_open=-5, gap_extend=-1) : AffineGapScoreModel(BLOSUM62, gap_open=-5, gap_extend=-1)
    p = Progress(Int64(round((length(sequence_list)^2)/2, digits=0)))

    # do pairwise alignments
    for i in 1:length(sequence_list)
        Threads.@threads for j in i:length(sequence_list)
            res = pairalign(LocalAlignment(), sequence_list[i], sequence_list[j], score_model);
            aln = alignment(res)
            score_matrix[i,j] = count_matches(aln) / count_aligned(aln)
            next!(p)
        end
    end

    score_matrix = (score_matrix + score_matrix') 
    score_matrix = score_matrix - Diagonal(score_matrix)/2
    writedlm(file * "_score_matrix.txt", score_matrix)
end

pwalign(ARGS)

# test
#pwalign(["/Users/Dimi/Desktop/kaptive_test.fasta", "DNA"])
