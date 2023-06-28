#=
learning.jl; Provides the basic functions to do learning.
=#

train(y, hdvs) = Dict(c=>aggregate(hdvs[y.==c]) for c in unique(y)) 

predict(v::AbstractHDV, centers) = maximum((similarity(v, xcᵢ), yᵢ) for (yᵢ, xcᵢ) in centers)[2]

predict(hdvs::Vector{<:AbstractHDV}, centers) = [maximum((similarity(v, xcᵢ), yᵢ) for (yᵢ, xcᵢ) in centers)[2] for v in hdvs]

function retrain!(centers, y, hdvs; niters=10, verbose=true)
    @assert length(y) == length(hdvs)
    n_obs = length(y)
    wrong = zeros(Bool, n_obs)
    for iter in 1:niters
        verbose && print("Iteration $iter of $niters ...")
        # check all wrongly classified classes
        for (i, (yᵢ, hdv)) in enumerate(zip(y, hdvs))
            (predict(hdv, centers) != yᵢ) && (wrong[i] = true)
        end
        n_errors = sum(wrong)
        verbose && println(" found $n_errors classification errors")
        # aggregate all the mistaken vectors again in the centers
        for (yₖ, cₖ) in centers
            aggregatewith!(cₖ, hdvs[(y.==yₖ) .& wrong])
        end
        fill!(wrong, false)
    end
    return centers
end