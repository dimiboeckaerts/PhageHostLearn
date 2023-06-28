#=
encoding.jl; This file implements functions to encode token-based data, be it alphabets, strings or full classes of text.
=#

# SEQUENCE EMBEDDING

# precomputing of the n-grams

ngrams_recursion(d::AbstractDict{K,V}, hdv::AbstractHDV) where {K,V<:AbstractHDV} = 
    Dict(k => hdv * Π(v, 1) for (k, v) in d)

ngrams_recursion(d::AbstractDict{K,V}, hdv::AbstractHDV) where {K,V<:AbstractDict} = 
    Dict(k => ngrams_recursion(v, hdv) for (k, v) in d)

function ngrams_recursion(d::AbstractDict{K,V}, hdvs) where {K,V}
    result = Dict{K,Dict{K,V}}()
    for k in keys(d)
        result[k] = ngrams_recursion(d, hdvs[k])
    end
    return result
end

# structure to store the N-grams
struct NGrams{V,D}
    d::D
    NGrams(V::Type, d::AbstractDict) = new{V,typeof(d)}(d)
end

Base.show(io::IO, ngrams::NGrams) = print("n-gram embedding of order $(order(ngrams)) for type $(vectortype(ngrams))")

# TODO: consider using an N-dim array?

# Generates nested dictionary with ngrams 
# needs to be hard-coded for dispatch
#=
compute_1_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), Dict(c=>hdvs[c] for c in alphabet))
compute_2_grams(hdvs, alphabet=1:length(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_1_grams(hdvs, alphabet).d, hdvs))
compute_3_grams(hdvs, alphabet=1:length(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_2_grams(hdvs, alphabet).d, hdvs))
compute_4_grams(hdvs, alphabet=1:length(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_3_grams(hdvs, alphabet).d, hdvs))
compute_5_grams(hdvs, alphabet=1:length(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_4_grams(hdvs, alphabet).d, hdvs))
compute_6_grams(hdvs, alphabet=1:length(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_5_grams(hdvs, alphabet).d, hdvs))
compute_7_grams(hdvs, alphabet=1:length(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_6_grams(hdvs, alphabet).d, hdvs))
compute_8_grams(hdvs, alphabet=1:length(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_7_grams(hdvs, alphabet).d, hdvs))
=#
compute_1_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), Dict(c=>hdvs[c] for c in alphabet))
compute_2_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_1_grams(hdvs, alphabet).d, hdvs))
compute_3_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_2_grams(hdvs, alphabet).d, hdvs))
compute_4_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_3_grams(hdvs, alphabet).d, hdvs))
compute_5_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_4_grams(hdvs, alphabet).d, hdvs))
compute_6_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_5_grams(hdvs, alphabet).d, hdvs))
compute_7_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_6_grams(hdvs, alphabet).d, hdvs))
compute_8_grams(hdvs, alphabet=keys(hdvs)) = NGrams(eltype(hdvs), ngrams_recursion(compute_7_grams(hdvs, alphabet).d, hdvs))

# get the embedding on the k-gram that starts at position i
get_gram_embedding(sequence, i, embeddings::Dict{K,V}) where {K,V<:Dict} =
            get_gram_embedding(sequence, i+1, embeddings[sequence[i]])
get_gram_embedding(sequence, i, embeddings::Dict{K,V}) where {K,V<:AbstractHDV} = embeddings[sequence[i]]
get_gram_embedding(sequence, i, embeddings::NGrams) = get_gram_embedding(sequence, i, embeddings.d)

order(embeddings::Dict{K,V}) where {K,V<:Dict} = order(first(embeddings::Dict{K,V})[2]) + 1
order(embeddings::Dict) = 1
order(embeddings::NGrams) = order(embeddings.d)

vectortype(embeddings::NGrams{V,D}) where {V, D} = V

get_leaf(d::AbstractDict{K,V}) where {K,V<:AbstractDict} = get_leaf(d[first(keys(d))])
get_leaf(d::AbstractDict) = d[first(keys(d))]

similar_vector(ngrams::NGrams) = similar(get_leaf(ngrams.d))

function sequence_embedding!(result::AbstractHDV, sequence, token_vectors, w=3)
    fill!(result.v, zero(eltype(result)))
    tmp = similar(result)
    n = length(sequence)
    for i in 1:n-w
        fill!(tmp.v, neutralbind(tmp))
        for k in (w-1):-1:0
            v = token_vectors[sequence[i+k]]
            offsetcombine!(tmp.v, bindfun(tmp), tmp.v, v.v, v.offset + k)
        end
        offsetcombine!(result.v, aggfun(result), result.v, tmp.v, 0)
    end
    result.m = length(sequence)-w
    normalize!(result)
    return result
end

function sequence_embedding!(result::AbstractHDV, sequence, ngrams_embedding::NGrams)
    fill!(result.v, zero(eltype(result)))
    n = length(sequence)
    w = order(ngrams_embedding)
    for i in 1:n-w
        hdv = get_gram_embedding(sequence, i, ngrams_embedding)
        offsetcombine!(result.v, aggfun(result), result.v, hdv.v, 0)
    end
    result.m = length(sequence)-w
    normalize!(result)
    return result
end

sequence_embedding(sequence, token_vectors::AbstractVector{V}, args...) where {V<:AbstractHDV} = 
                sequence_embedding!(similar(first(token_vectors)), sequence, token_vectors, args...)
    
sequence_embedding(sequence, token_vectors::Dict{T,V}, args...) where {T,V<:AbstractHDV} = 
                sequence_embedding!(similar(first(token_vectors)[2]), sequence, token_vectors, args...)

sequence_embedding(sequence, ngrams_embedding::NGrams, args...) = 
                sequence_embedding!(similar_vector(ngrams_embedding), sequence, ngrams_embedding, args...)

