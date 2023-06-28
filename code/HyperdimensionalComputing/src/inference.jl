#=
inference.jl; This file implements functions to compare two hyperdimensional vectors.
=#

using LinearAlgebra

LinearAlgebra.norm(hdv::AbstractHDV) = norm(hdv.v)

function LinearAlgebra.dot(x::AbstractHDV, y::AbstractHDV)
    nx = normalizer(x)
    ny = normalizer(y)
    if x.offset == y.offset  
        return sum(((vx,vy),)->dot(nx(vx),ny(vy)), zip(x.v, y.v))
    else
        return sum(((vx,vy),)->dot(nx(vx),ny(vy)), zip(x, y))
    end
end

cos_sim(x::AbstractVector, y::AbstractVector) = dot(x, y) / (norm(x) * norm(y))

jacc_sim(x::AbstractVector, y::AbstractVector) = dot(x, y) / sum(t->t[1]+t[2]-t[1]*t[2], zip(x,y))

# specific similarities
# for HDVs that can both be pos and neg,
# we use cosine similarity

similarity(x::BipolarHDV, y::BipolarHDV) = cos_sim(x, y)
similarity(x::GradedBipolarHDV, y::GradedBipolarHDV) = cos_sim(x, y)
similarity(x::RealHDV, y::RealHDV) = cos_sim(x, y)

similarity(x::BinaryHDV, y::BinaryHDV) = jacc_sim(x, y)
similarity(x::GradedHDV, y::GradedHDV) = jacc_sim(x, y)

#strange_fun(x, y) = sum((a, b)->max(a*b-sqrt(1-a^2)*sqrt(1-b^2),0.0), zip(x, y))