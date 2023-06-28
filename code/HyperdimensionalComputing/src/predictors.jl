#=
predictors.jl; This file implements methods to compare encoded strings to references and to make predictions from this comparison.
=#


### Cosine distance predictor
"""
This function makes predictions for the given encodings by finding the closest class encoding in terms of cosine distance.\n
Input: 
    - a vector of encoded test sequences.\n
    - a dictionary of encoded reference sequences with their class a key.\n
Output: a vector of class predictions in the same order as the input test encodings.\n
    
DISCLAIMER: Currently every test encoding given to this function will be classified to the closest class, even when the true class is not in the train encodings.
=> ALLE KLASSES EVEN VER VAN EXAMPLE => UNCLASSIFIED
"""
function cosine_predict(test_encodings::Vector, train_encodings::Dict)
    predictions = Vector{String}()
    for hv in test_encodings
        closest_distance = Inf
        match = missing
        for (k, v) in train_encodings
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


### Naive bayes predictor