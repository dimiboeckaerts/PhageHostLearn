# --------------------------------------------------
# PAIRWISE ALIGNMENTS IN JULIA
#
# @author: dimiboeckaerts
# --------------------------------------------------

using FASTX
using BioAlignments
using DelimitedFiles
using LinearAlgebra
using ProgressMeter

function pwalign(args)
    """
    Function that does a pairwise alignment of two sequences in given FASTA file.
    """
    sequence1 = args[1]
    sequence2 = args[2]
    align_type = args[3]
    score_model = align_type == "DNA" ? AffineGapScoreModel(EDNAFULL, gap_open=-5, gap_extend=-1) : AffineGapScoreModel(BLOSUM62, gap_open=-5, gap_extend=-1)
    res = pairalign(LocalAlignment(), sequence1, sequence2, score_model);
    aln = alignment(res)
    score = count_matches(aln) / count_aligned(aln)
    println(score)
end

pwalign(ARGS)

# test
#pwalign(["/Users/Dimi/Desktop/kaptive_test.fasta", "DNA"])
