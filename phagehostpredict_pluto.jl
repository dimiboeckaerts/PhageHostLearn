### A Pluto.jl notebook ###
# v0.17.0

using Markdown
using InteractiveUtils

# ╔═╡ 51e2b10b-2175-4bd8-896f-ab565584201d
include("/Users/dimi/Documents/GitHub/HyperdimensionalComputing.jl/src/HyperdimensionalComputing.jl")

# ╔═╡ 7155ffc2-7d01-11ec-3a7e-efef0b462ad9
md"""
# PhageHostPredict HDC (Klebsiella)

An AI-based Phage-Host interaction predictor framework with receptors and receptor-binding proteins at its core. This particular PhageHostPredict is for *Klebsiella pneumoniae* related phages and uses hyperdimensional computing (HDC).

This notebook follows after having ran the PhageHostPredict_processing steps implemented in the accompanying Python script.

The framework follows these steps:
1. Loading an initial processing of the phage-host data
2. Computing HDC embeddings for the interactions
3. Learning interactions and making predictions
"""

# ╔═╡ ef6ff138-0857-40e9-9de8-77146d0746a9
md"""
### 1 - Libraries, directories and functions
"""

# ╔═╡ d931719a-a6cd-48c2-a46a-06c82d4b9c76
md"""
begin
	include("/Users/dimi/Documents/GitHub/HyperdimensionalComputing.jl/src/HyperdimensionalComputing.jl")
	using DataFrames
	using ProgressMeter
	using CSV
	using FASTX
	using BioAlignments
end
"""

# ╔═╡ 06a42113-26f4-4428-983e-0f6bcdf3248e
md"""
### 1 - Loading and processing data
"""

# ╔═╡ 08fa9b1e-ca5d-4a46-afc2-76f8e42c8689
@__DIR__ 

# ╔═╡ Cell order:
# ╟─7155ffc2-7d01-11ec-3a7e-efef0b462ad9
# ╟─ef6ff138-0857-40e9-9de8-77146d0746a9
# ╠═d931719a-a6cd-48c2-a46a-06c82d4b9c76
# ╟─06a42113-26f4-4428-983e-0f6bcdf3248e
# ╠═51e2b10b-2175-4bd8-896f-ab565584201d
# ╠═08fa9b1e-ca5d-4a46-afc2-76f8e42c8689
