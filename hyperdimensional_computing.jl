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

# small tests
encoded_alp = Dict("A" => [0, 1, 2, 0], "B" => [1, 2, 2, 0])
sequence = "ABA"
expected_output = [0, 1, 2, 0] + [1, 2, 2, 0] + [0, 1, 2, 0]
encode_sequence(sequence, encoded_alp, dim=4) == expected_output

encoded_alp = Dict("A" => [0, 1, 2, 0], "B" => [1, 2, 2, 0])
sequence = "ABBA"
expected_output = ([0, 1, 2, 0] .* [0, 1, 2, 2]) + ([1, 2, 2, 0].*[0, 1, 2, 2]) + ([1, 2, 2, 0].*[0, 0, 1, 2])
encode_sequence(sequence, encoded_alp, k=2, dim=4) == expected_output


function encode_classes(encoding_matrix, classes, encoded_alphabet::Dict; max_iterations=20)
    """
    This function loops over a matrix of hyperdimensional vectors and its associated
    classes and constructs a profile for each class by summing the corresponding HVs.

    Input:
    - encoding_matrix: matrix with encodings (#encodings x dim)
    - classes: corresponding class labels (# encodings)
    Output: dictionary of HVs for each of the classes

    Addition: don't subtract from all classes, only wrong one?
    """
    # initial encodings
    class_encodings = Dict()
    for row in 1:size(encoding_matrix)[1]
        if classes[row] in keys(class_encodings):
            class_encodings[classes[row]] += encoding_matrix[row,:]
        else
            class_encodings[classes[row]] = encoding_matrix[row,:]
        end
    end

    # retraining
    for iteration in 1:max_iterations
        count_wrong_iter = 0

        # loop over matrix
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
                    if key != actual class
                        class_encodings[key] -= encoding_matrix[row,:]
                    else
                        class_encodings[key] += encoding_matrix[row,:]
                    end
                end
            end
        end

        # check convergence
        if count_wrong_iter < count_wrong
            count_wrong = count_wrong_iter
            ...

    end

    return class_encodings
end



"""
"""
function predict(data::Vector{Vector{Int}}, reference::Dict{String,Vector{Int}})
    predictions = Vector{String}()
    for hv in data
        closest_distance = 1
        match = missing
        for (k, v) in reference
            dist = cosine_dist(hv, v)
            if (dist < closest_distance)
                closest_distance = dist
                match = k
            end
        end
        push!(predictions, match)
    end
    return predictions
end