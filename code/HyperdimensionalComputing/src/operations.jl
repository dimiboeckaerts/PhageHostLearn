#=
operations.jl; This file implements operations that can be done on hypervectors to enable them to encode text-based data.
=#

# Remark: use element-wise reduce, maybe using LazyArrays?


#=

| Operation            | symbol | remark                                                                                                          |
| -------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| Bundling/aggregating | `+`    | combines the information of two vectors into a new vector similar to both                                       |
| Binding              | `*`    | mapping, combines the two vectors in something different from both, preserves distance, distributes of bundling |
| Shifting             | `Π`    | Permutation (in practice cyclic shifting), distributes over addition, conserves distance                        |
=#

"""
    grad2bipol(x::Number)

Maps a graded number in [0, 1] to the [-1, 1] interval.
"""
grad2bipol(x::Number) = 2x - one(x)


"""
bipol2grad(x::Number)

Maps a bipolar number in [-1, 1] to the [0, 1] interval.
"""
bipol2grad(x::Number) = (x + one(x)) / 2

three_pi(x, y) = abs(x-y)==1 ? zero(x) : x * y / (x * y + (one(x) - x) * (one(y) - y))
fuzzy_xor(x, y) = (one(x)-x) * y + x * (one(y)-y)

three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))  # currently just *

aggfun(::AbstractHDV) = +
aggfun(::GradedHDV) = three_pi
aggfun(::GradedBipolarHDV) = three_pi_bipol

bindfun(::AbstractHDV) = *
bindfun(::BinaryHDV) = ⊻
bindfun(::GradedHDV) = fuzzy_xor
bindfun(::GradedBipolarHDV) = fuzzy_xor_bipol

neutralbind(hdv::AbstractHDV) = one(eltype(hdv))
neutralbind(hdv::BinaryHDV) = false
neutralbind(hdv::GradedHDV) = zero(eltype(hdv))
neutralbind(hdv::GradedBipolarHDV) = -one(eltype(hdv))

function elementreduce!(f, itr, init)
    return foldl(itr; init) do acc, value
        acc .= f.(acc, value)
    end
end

# computes `r[i] = f(x[i], y[i+offset])`
# assumes postive offset (for now)
@inline function offsetcombine!(r, f, x, y, offset=0)
    @assert length(r) == length(x) == length(y)
    n = length(r)
    if offset==0
        r .= f.(x, y)
    else
        i′ = n - offset
        for i in 1:n
            i′ = i′ == n ? 1 : i′ + 1
            @inbounds r[i] = f(x[i], y[i′])
        end
    end
    return r
end

@inline function offsetcombine(f, x::V, y::V, offset=0) where {V<:AbstractVecOrMat}
    @assert length(x) == length(y)
    r = similar(x)
    n = length(r)
    if offset==0
        r .= f.(x, y)
    else
        i′ = n - offset
        for i in 1:n
            i′ = i′ == n ? 1 : i′ + 1
            @inbounds r[i] = f(x[i], y[i′])
        end
    end
    return r
end

# AGGREGATION
# -----------

aggregate(hdvs::AbstractVector{<:AbstractHDV}, args...; kwargs...) = aggregate!(similar(first(hdvs)), hdvs, args...; kwargs...)
aggregate(hdvs::NTuple{N,T}, args...; kwargs...) where {N,T<:AbstractHDV} = aggregate!(similar(first(hdvs)), hdvs, args...; kwargs...)

Base.:+(hdv1::HDV, hdv2::HDV) where {HDV<:AbstractHDV} = aggregate!(similar(hdv1), (hdv1, hdv2))

clearhdv!(r::AbstractHDV) = fill!(r.v, zero(eltype(r)))
clearhdv!(r::GradedHDV) = fill!(r.v, one(eltype(r))/2)

function aggregate!(r::AbstractHDV, hdvs; clear=true, norm=false)
    clear && clearhdv!(r)
    aggr = aggfun(r)
    foldl(hdvs, init=r.v) do acc, value
        offsetcombine!(acc, aggr, acc, value.v, value.offset)
    end
    for hdv in hdvs
        r.m += hdv.m
    end
    norm && normalize!(r)
    return r
end

function aggregate!(r::AbstractHDV, hdvs, weights; clear=true, norm=false)
    @assert length(hdvs) == length(weights) "You have to provide the same number of weights as vectors."
    clear && clearhdv!(r)
    aggr = aggfun(r)
    foldl(zip(hdvs, weights), init=r.v) do acc, (value, weight)
        offsetcombine!(acc, aggr, acc, weight .* value.v, value.offset)
    end
    for (hdv, weight) in zip(hdvs, weights)
        r.m += weight * hdv.m
    end
    norm && normalize!(r)
    return r
end

aggregatewith!(r::AbstractHDV, hdvs; kwargs...) = aggregate!(r, hdvs; clear=false, kwargs...)

# BINDING
# -------

Base.bind(hdvs::AbstractVector{<:AbstractHDV}) = bind!(similar(first(hdvs)), hdvs)
Base.bind(hdvs::NTuple{N,T}) where {N,T<:AbstractHDV} = bind!(similar(first(hdvs)), hdvs)

Base.:*(hdv1::HDV, hdv2::HDV) where {HDV<:AbstractHDV} = bind!(similar(hdv1), (hdv1, hdv2))

function bind!(r::AbstractHDV, hdvs)
    fill!(r.v, neutralbind(r))
    r.m = 1  # fresh vector
    # extract the normalizer
    nr = normalizer(r)
    binder = (x, y) -> bindfun(r)(nr(x), nr(y))
    foldl(hdvs, init=r.v) do acc, value
        offsetcombine!(acc, binder, acc, value.v, value.offset)
    end
    return r
end

# SHIFTING
# --------

function Base.circshift!(hdv::AbstractHDV, k)
    hdv.offset += k
    return hdv
end

for hdvt in [:BipolarHDV, :BinaryHDV, :GradedBipolarHDV, :GradedHDV, :RealHDV]
    eval(quote
        Base.circshift(hdv::$hdvt, k::Integer) = $(hdvt)(hdv.v, hdv.offset + k)
    end)
end

Π(hdv::AbstractHDV, k) = circshift(hdv, k)
Π!(hdv::AbstractHDV, k) = circshift!(hdv, k)

function resetoffset!(hdv::AbstractHDV)
    hdv.offset == 0 && return hdv
    v = circshift(hdv.v, -hdv.offset)
    hdv.offset = 0
    return hdv
end