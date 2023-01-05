#=
vectors.jl; Implements the interface for HDV
=#

validindex(i, n) = i < 1 ? validindex(i + n, n) : (i > n ? validindex(i - n, n) : i)

# random numbers in an interval [l, u]
@inline function randinterval(T::Type, n, l, u)
    @assert l < u "The lower bound should be belowe the upper bound"
    return rand(T, n) .* T(u - l) .+ T(l)
end


abstract type AbstractHDV{T} <: AbstractVector{T} end

# taking the indices takes a long time=> remove!

@inline Base.getindex(hdv::AbstractHDV, i) = @inbounds hdv.v[validindex(i-hdv.offset, length(hdv))] .|> normalizer(hdv)

Base.size(hdv::AbstractHDV) = size(hdv.v)

@inline Base.setindex!(hdv::AbstractHDV, val, i) = @inbounds (hdv.v[validindex(i-hdv.offset, length(hdv))] = val)

Base.Vector(hdv::AbstractHDV) = collect(hdv)

#=
Base.iterate(hdv::AbstractHDV, state=1) = state > length(hdv) ?
                                                    nothing :
                                            (normalizer(hdv)(hdv.v[validindex(i-hdv.offset, length(hdv))]), state+1)
=#

normalizer(::AbstractHDV) = identity  # normalizer does nothing by default

function normalize!(hdv::AbstractHDV)
    hdv.v .= normalizer(hdv).(hdv.v)
    hdv.m = 1
    return hdv
end

getvector(hdv::AbstractHDV) = hdv.v

Base.sum(hdv::AbstractHDV) = sum(hdv.v)

# We always provide a constructor with optinal dimensionality (n=10,000 by default) and
# a method `similar`.

mutable struct BipolarHDV <: AbstractHDV{Int}
    v::Vector{Int}
    offset::Int
    m::Int
    BipolarHDV(v::Vector, offset=0, m=1) = new(v, offset, m)
end

BipolarHDV(n::Int=10_000) = BipolarHDV(rand((-1, 1), n))

Base.similar(hdv::BipolarHDV) = BipolarHDV(similar(hdv.v), 0, 0)

normalizer(::BipolarHDV) = vᵢ -> clamp(vᵢ, -1, 1) 

# `BinaryHDV` contain binary vectors.

mutable struct BinaryHDV <: AbstractHDV{Bool}
    v::Vector{Int}
    offset::Int
    m::Int
    BinaryHDV(v::AbstractVector, offset=0, m=1) = new(v, offset, m)
end


BinaryHDV(n::Int=10_000) = BinaryHDV(rand(0:1, n))

Base.similar(hdv::BinaryHDV) = BinaryHDV(similar(hdv.v), 0, 0)

normalizer(hdv::BinaryHDV) = vᵢ -> 2vᵢ > hdv.m 


# GradedBipolarHDV are vectors in $[-1, 1]^n$, allowing for graded relations.

mutable struct GradedBipolarHDV{T<:Real} <: AbstractHDV{T}
    v::Vector{T}
    offset::Int
    m::T
    GradedBipolarHDV(v::AbstractVector, offset=0, m=1) = new{eltype(v)}(v, offset, m)
end

GradedBipolarHDV(T::Type, n::Int=10_000; l=-0.8, u=0.8) = GradedBipolarHDV(randinterval(T, n, l, u))
GradedBipolarHDV(n::Int=10_00; l=-0.8, u=0.8) = GradedBipolarHDV(Float32, n; l, u)

Base.similar(hdv::GradedBipolarHDV) = GradedBipolarHDV(similar(hdv.v), 0, 0)

#normalizer(hdv::GradedBipolarHDV) = vᵢ -> clamp(vᵢ, -1, 1)

mutable struct GradedHDV{T<:Real} <: AbstractHDV{T}
    v::Vector{T}
    offset::Int
    m::Int
    GradedHDV(v::AbstractVector, offset=0, m=1) = new{eltype(v)}(v, offset, m)
end

GradedHDV(T::Type, n::Int=10_000; l=0.2, u=0.8) = GradedHDV(randinterval(T, n, l, u))
GradedHDV(n::Int=10_000; l=0.2, u=0.8) = GradedHDV(Float32, n; l, u)

Base.similar(hdv::GradedHDV) = GradedHDV(similar(hdv.v), 0, 0)

#normalizer(hdv::GradedHDV) = vᵢ -> clamp(vᵢ, -1, 1)


# Finally, `RealHDV` contain real values, drawn from a standard normal distribution
# by default.

mutable struct RealHDV{T<:Real} <: AbstractHDV{T}
    v::Vector{T}
    offset::Int
    m::Float64
    RealHDV(v::Vector{T}, offset=0,m=1) where {T} = new{T}(v,offset,m)
end

RealHDV(n::Int=10_000) = RealHDV(randn(n))
RealHDV(T::Type{<:Real}, n::Int=10_000) = RealHDV(T.(randn(n)), 0)

normalizer(hdv::RealHDV) = vᵢ -> vᵢ / sqrt(hdv.m)

Base.similar(hdv::RealHDV) = RealHDV(similar(hdv.v), 0, 0)