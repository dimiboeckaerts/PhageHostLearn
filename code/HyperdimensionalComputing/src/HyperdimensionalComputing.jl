module HyperdimensionalComputing

using Distances

export AbstractHDV, BinaryHDV, BipolarHDV,
    GradedBipolarHDV, RealHDV, GradedHDV
export offsetcombine, offsetcombine!
export aggregate, aggregate!, aggregatewith!, bind, bind!, Π, Π!, resetoffset!
export sequence_embedding, sequence_embedding!
export compute_1_grams, compute_2_grams, compute_3_grams, compute_4_grams, compute_5_grams, 
        compute_6_grams, compute_7_grams, compute_8_grams
export similarity, jacc_sim, cos_sim
export train, predict, retrain!

include("vectors.jl")
include("operations.jl")
include("encoding.jl")
include("inference.jl")
include("learning.jl")

end