"""
PW KERNEL COMPUTATION FOR BIOLOGICAL SEQUENCES

@author: dimiboeckaerts

In this script, BioSequences and BioAlignments packages are used to compute 
pairwise alignments for the detected RBP sequences (as DNA), both in its entirity
as well as for the N-terminus and C-terminus separately. A matrix of pairwise 
identity percentages is computed.
"""

# IMPORT LIBRARIES
# --------------------------------------------------
# import Pkg
# Pkg.add("BioSequences")
# Pkg.add("BioAlignments")
using BioSequences
using BioAlignments
using DelimitedFiles
using LinearAlgebra
using ProgressMeter
using DataFrames
using CSV


# FUNCTIONS FOR ALIGNMENT SCORE COMPUTATION
# --------------------------------------------------
# convert file to a list
function file_to_array(file; cut="")
    """
    Function that reads a .csv dataframe and puts its sequences in an array.
    The dataframe should contain a column named 'sequence' to extract
    sequences from (as DNA).

    Input:
    - file: a string to the dataframe that will be used to extraxt sequences from.
        The dataframe should contain a column named 'sequence'
    - cut: "N", "C" or "" depending on if you want to cut the sequences.
    """
    sequences = []
    reader = DataFrame(CSV.File(file))
    for record in reader.sequence
        protein_seq = translate(convert(LongRNASeq, LongDNASeq(record)))[1:(end-1)]
        if (cut == "N") & (length(protein_seq) >= 200)
            protein_seq = protein_seq[1:200]
        elseif (cut == "C") & (length(protein_seq) >= 250)
            protein_seq = protein_seq[200:end]
        elseif (cut == "C") & (200 <= length(protein_seq) >= 250)
            protein_seq = protein_seq[(end-50):end]
        end

        push!(sequences, protein_seq)
    end
    return sequences
end

# calculate alingment and its identity/match score
function calculate_perc_identity(sequence1, sequence2)
    """
    This function calculates the percentage of matches between two aligned protein sequences.
    """
    scoremodel = AffineGapScoreModel(BLOSUM62, gap_open=-5, gap_extend=-1)
    res = pairalign(LocalAlignment(), sequence1, sequence2, scoremodel);
    aln = alignment(res)

    return count_matches(aln) / count_aligned(aln)
end

# construct alignment score matrix
function compute_pw_matrix(file; cutoff="")
    """
    Function that constructs a kernel matrix based on pairwise sequence alignment.

    Input: a file handle to a .CSV file containing a DataFrame with column 'sequence'
    Output: a kernel matrix
    """
    # for simple alignment: AffineGapScoreModel(match=1, mismatch=0, gap_open=0, gap_extend=0)
    # for BLOSUM alignment: AffineGapScoreModel(BLOSUM62, gap_open=-10, gap_extend=-1)
    # by adding the transposed matrix, we add the diagonal twice... this should be corrected.

    sequence_list = file_to_array(file, cut=cutoff)
    kernel_matrix = zeros(length(sequence_list), length(sequence_list))
    p = Progress(Int64(round((length(sequence_list)^2)/2, digits=0)))

    for i in 1:length(sequence_list), j in i:length(sequence_list)
        kernel_matrix[i,j] = calculate_perc_identity(sequence_list[i], sequence_list[j])
        next!(p)
    end

    kernel_matrix = kernel_matrix + kernel_matrix'
    kernel_matrix = kernel_matrix - Diagonal(kernel_matrix)/2
    return kernel_matrix
end


# TIMING TEST
# --------------------------------------------------
test_seq1 = aa"MTDIITNVVIGMPSQLFTMARSFKAVANGKIYIGKIDTDPVNPENQIQVYVENEDGSHVPASQPIVINAAGYPVYNGQIVKFVTEQGHSMAVYDAYGSQQFYFQNVLKYDPDQFGPDLIEQLAQSGKYSQDNTKGDAMIGVKQPLPKAVLRTQHDKNKEAISILDFGV"
test_seq2 = aa"MAITKIILQQMVTMDQNSITASKYPKYTVVLSNSISSITAADVTSAIESSKASGPAAKQSEINAKQSELNAKDSENEAEISATSSQQSATQSASSATASANSAKAAKTSETNANNSKNAAKTSETNAASSASSASSFATAAENSARAAKTSETNAGNSAQAADASKTA"

scoremodel = AffineGapScoreModel(BLOSUM62, gap_open=-10, gap_extend=-1)
res = pairalign(LocalAlignment(), test_seq1, test_seq2, scoremodel);
aln = alignment(res)
sc = score(res)
count_matches(aln)

smallfile = "/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/proFiberBase_mini.csv"
f = file_to_array(smallfile, cut="")
m = compute_pw_matrix(smallfile, cutoff="C")

for i in f
    println(length(i))
end


# COMPUTE FULL MATRIX
# --------------------------------------------------
RBPdf = "/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/proFiberBase_230221.csv"
m = compute_pw_matrix(RBPdf, cutoff="")
size(m)
writedlm("/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/RBP_alignmentmatrix.txt", m)
matrix = readdlm("/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/RBP_alignmentmatrix.txt")


# COMPUTE N/C-TERMINAL CUTOFFS
# --------------------------------------------------
RBPdf = "/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/proFiberBase_230221.csv"
mN = compute_pw_matrix(RBPdf, cutoff="N")
size(mN)
writedlm("/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/RBP_alignmentmatrix_Nterm.txt", mN)
matrix = readdlm("/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/RBP_alignmentmatrix_Nterm.txt")

RBPdf = "/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/proFiberBase_230221.csv"
mC = compute_pw_matrix(RBPdf, cutoff="C")
size(mC)
writedlm("/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/RBP_alignmentmatrix_Cterm.txt", mC)
matrix = readdlm("/Users/Dimi/GoogleDrive/PhD/4_WP2_PWLEARNING/42_DATA/RBP_alignmentmatrix_Cterm.txt")
