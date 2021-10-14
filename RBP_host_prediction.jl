# --------------------------------------------------
# HYPERDIMENSIONAL COMPUTING IN JULIA: HOST CLASSIFICATION
#
# @author: dimiboeckaerts
# --------------------------------------------------

# LIBRARIES
# --------------------------------------------------
using CSV
using MLJ
using DataFrames
using ProgressMeter
using ScikitLearn.CrossValidation: KFold
include("hyperdimensional_computing.jl")


# DATA PROCESSING
# --------------------------------------------------
fibers = DataFrame(CSV.File("/Users/Dimi/Documents/GitHub_Local/BacteriophageHostPrediction/RBP_database887.csv"))
sequences = fibers.protein_sequence;
classes = fibers.class;
classes = coerce(classes, Multiclass);


# HYPERDIMENSIONAL COMPUTING
# --------------------------------------------------
#dimensions = LinRange(5000, 10000, 6)
dimension = 10000
nfolds = 5
kvalues = LinRange(5, 10, 6)
encoded_alphabet = encode_alphabet(["A", "G", "I", "L", "P", "V", "F", "W", "Y", "D",
                                    "E", "R", "H", "K", "S", "T", "C", "M", "N", "Q"],
                                    dim=dimension)

# kfold over the sequences: use 80% to build the class HVs and 20% to make predictions for
for kval in kvalues
    predictions = zeros(length(sequences))
    pb = Progress(nfolds)
    for (train_indices, test_indices) in KFold(length(sequences), n_folds=nfolds)
        # define matrices for hypervectors & labels
        x_train = zeros(length(train_indices), dimension)
        x_test = zeros(length(test_indices), dimension)
        y_train = classes[train_indices]; y_test = classes[test_indices]

        # encode sequences
        for (i, train_index) in enumerate(train_indices)
            x_train[i,:] = encode_sequence(sequences[train_index], encoded_alphabet, k=Int64.(round(kval, digits=1)))
        end
        for (i, test_index) in enumerate(test_indices)
            x_test[i,:] = encode_sequence(sequences[test_index], encoded_alphabet, k=Int64.(round(kval, digits=1)))
        end

        # encode class vectors
        class_vectors = encode_classes(x_train, y_train)

        # make predictions
        preds = make_predictions(x_test, class_vectors)

        # save predictions
        for (i, test_index) in enumerate(test_indices)
            predictions[test_index] = preds[i]
        end
        next!(pb)
    end

    # compute performance
    predictions = Int64.(predictions)
    predictions = coerce(predictions, Multiclass)
    F1_it = MLJ.macro_f1score(predictions, classes)
    println("F1 score for k-mer value of ", kval, " : ", F1_it)
end

# compute performance
predictions = Int64.(predictions)
predictions = coerce(predictions, Multiclass)
MLJ.macro_f1score(predictions, classes)
MLJ.confmat(predictions, classes)

re = MLJ.multiclass_recall(predictions, classes)
pre = MLJ.multiclass_positive_predictive_value(predictions, classes)
Fm = (2*pre*re)/(pre+re)

# compute weighted multiclass F1 score
class_weights = [count(i->(i==x), classes) for x in levels(classes)]
total_F1 = 0
for (i, level) in enumerate(levels(classes))
    # binarize arrays
    binary_predictions = coerce(Int64.(predictions .== level), Multiclass)
    binary_classes = coerce(Int64.(classes .== level), Multiclass)
    # compute binary F1
    F1 = MLJ.f1score(binary_predictions, binary_classes)
    println("F1 score for class ", level, " : ", F1)
    # refactor for class weights
    total_F1 += F1*(class_weights[i]/sum(class_weights))
end
println("Weighted average F1 score: ", total_F1) # 80.05%