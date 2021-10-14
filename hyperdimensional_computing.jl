# --------------------------------------------------
# HYPERDIMENSIONAL COMPUTING IN JULIA
#
# @author: dimiboeckaerts
# --------------------------------------------------

# LIBRARIES
# --------------------------------------------------
using Random, Distances


# FUNCTIONS
# --------------------------------------------------
function encode_alphabet(alphabet::Vector{String}; dim=10000::Int)
    """
    This function encodes an alphabet of characters into hyperdimensional 
    vectors (HVs) as a starting point to encode text or sequences into HVs.
    The vectors are bipolar vectors.

    Input: an alphabet as vector of strings; e.g. ["A", "B", "C"]
    Output: a dictionary with HVs for each of the characters
    """
    encodings = Dict()
    for character in alphabet
        encodings[character] = rand([-1 1], dim)
    end
    return encodings
end


function encode_sequence(sequence::String, encoded_alphabet::Dict; k=1::Int, dim=10000::Int)
    """
    This function creates a hyperdimensional vector for a given input sequence.
    It uses the encoded alphabet to go over the sequence elementwise or in k-mer blocks.

    Input:
    - sequence to encode as string
    - encoded_alphabet: dictionary of encoded characters
    - k: k that defines the length of a k-mer 
    Output: an encoding for the sequence

    Addition: kmer_encoding with a sum instead of mutiplication?
    Addition: circshift over entire sequence with k=1
    Addition: bipolarize vector again at the end?
    """
    kmers = [sequence[i:i+k-1] for i in 1:length(sequence)-k+1]
    encoding = zeros(Int, dim)
    for kmer in kmers
        kmer_encoding = ones(Int, dim)
        for i in 1:k
            kmer_encoding .*= circshift(encoded_alphabet[string(kmer[i])], i-1)
        end
        encoding += kmer_encoding
    end
    return encoding
end


function encode_classes(encoding_matrix::Array, classes; max_iterations=25::Int)
    """
    This function loops over a matrix of hyperdimensional vectors and its associated
    classes and constructs a profile for each class by summing the corresponding HVs.

    Compared to traditional machine learning, this encoding of classes constitutes the
    'learning', as hyperdimensional vectors of all classes are combined via elementwise
    addition to learn a representation of the class a a whole.

    Input:
    - encoding_matrix: matrix with encodings (#encodings x dim)
    - classes: corresponding class labels (# encodings)
    - max_iterations: # of max iterations for retraining
    Output: dictionary of HVs for each of the classes

    Addition: don't subtract from all classes, only wrong one?
    Addition: rethink retraining for multiclass, more complex measure?
    """
    # initial encodings
    class_encodings = Dict()
    for row in 1:size(encoding_matrix)[1]
        if classes[row] in keys(class_encodings)
            class_encodings[classes[row]] += encoding_matrix[row,:]
        else
            class_encodings[classes[row]] = encoding_matrix[row,:]
        end
    end

    # retraining
    count_wrong = 10000
    stop = 0
    iteration = 0
    while (iteration <= max_iterations) & (count_wrong > 0) & (stop == 0)
        # loop over matrix
        count_wrong_iter = 0
        for row in 1:size(encoding_matrix)[1]
            distances = Dict()
            actual_class = classes[row]
            for (class, class_vector) in class_encodings # compute distances
                distances[class] = cosine_dist(encoding_matrix[row,:], class_vector)
            end
            minimal_class = findmin(distances)[2]

            if minimal_class != actual_class # if wrong, adjust
                count_wrong_iter += 1
                for key in keys(class_encodings)
                    if key != actual_class
                        class_encodings[key] -= encoding_matrix[row,:]
                    else
                        class_encodings[key] += encoding_matrix[row,:]
                    end
                end
            end
        end
        #println("it: ", iteration, "wrong: ", count_wrong, "wrong it: ", count_wrong_iter)

        # check convergence
        if count_wrong_iter < count_wrong
            count_wrong = count_wrong_iter
            iteration += 1
            stop = 0
        elseif count_wrong_iter > count_wrong
            count_wrong = count_wrong_iter
            iteration += 1
            stop = 0
        else
            stop = 1
        end
    end

    println("number of iterations: ", iteration)
    println("number of wrong class assignments: ", count_wrong)
    return class_encodings
end


function make_predictions(encoding_matrix::Array, class_encodings::Dict)
    """
    This function makes predictions of the given encodings by comparing them to the
    class encodings. The cosine distance is computed for each of the class encodings 
    and the class is returned as prediction for which the distance is the smallest.

    Input:
    - encoding_matrix: HVs of 'testsamples' in a matrix to make predictions for
    - class_encodings: HVs of the classes to compare with
    Output: predictions for each of the rows of the matrix
    """
    predictions = []
    for row in 1:size(encoding_matrix)[1]
        distances = Dict()
        test_sample = encoding_matrix[row, :]
        for (class, class_vector) in class_encodings # compute distances
            distances[class] = cosine_dist(test_sample, class_vector)
        end
        prediction = findmin(distances)[2]
        push!(predictions, prediction)
    end
    return predictions
end


# TESTS
# --------------------------------------------------
# test 1
encoded_alph = encode_alphabet(["A", "C", "T", "G"])

# test 2
encoded_alp = Dict("A" => [0, 1, 2, 0], "B" => [1, 2, 2, 0])
sequence = "ABA"
expected_output = [0, 1, 2, 0] + [1, 2, 2, 0] + [0, 1, 2, 0]
encode_sequence(sequence, encoded_alp, dim=4) == expected_output

encoded_alp = Dict("A" => [0, 1, 2, 0], "B" => [1, 2, 2, 0])
sequence = "ABBA"
expected_output = ([0, 1, 2, 0] .* [0, 1, 2, 2]) + ([1, 2, 2, 0].*[0, 1, 2, 2]) + ([1, 2, 2, 0].*[0, 0, 1, 2])
encode_sequence(sequence, encoded_alp, k=2, dim=4) == expected_output

# test 3
sequences = ["GGTCTGGGATC", "AATCGGATC", "AAGGCTCTA", "AACGCATTGG", "AATCGAATCA",
 "AAGGTCTTT", "GGATCTTAC", "GGTTACCA", "GGGTTCATAA", "GGATTCTAA"]
 classes = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
 encoded_alph = encode_alphabet(["A", "C", "T", "G"])
 encos = zeros(10, 10000)
 for (i, sequence) in enumerate(sequences)
    encos[i,:] = encode_sequence(sequence, encoded_alph, k=3)
 end
 class_enco = encode_classes(encos, classes)

 # test 4
 preds = make_predictions(encos, class_enco)