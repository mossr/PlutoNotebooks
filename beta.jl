### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ c59d5960-f7e0-11ea-21a0-d9ced93de3f5
using Distributions, PGFPlots, Statistics

# ╔═╡ ff09b860-f7e5-11ea-21a9-f90adbc47fa2
begin
	using Random; Random.seed!(0)
	apply(xₜ) = [0.1, 0.5, 0.9][xₜ]
	estimates = thompson_sampling(ones(3), ones(3), apply)
	argmax(mean.(estimates))
end

# ╔═╡ 41494860-f7e3-11ea-27bc-1b2515638b08
include("../src/thompson_sampling.jl")

# ╔═╡ bb2af0f2-f7e0-11ea-037c-45857eea5fd7
md"# Beta Distribution"

# ╔═╡ 4f7cd780-f7e3-11ea-13f9-79626222bcd6
bt = thompson_sampling(ones(3), ones(3), xₜ->[0.1, 0.5, 0.6][xₜ]; T=10000)

# ╔═╡ c0cb5ef0-f7e0-11ea-27d4-8b0dd09e98e1
function plot_beta(B=[Beta(5,3)])
	r4 = x->round(x, digits=4)
	plots = Plots.Linear[]
	𝛍 = Float64[]
	for beta in B
		push!(𝛍, mean(beta))
		p = Plots.Linear(x->pdf(beta, x), (0,1),
					     style="solid, thick, mark=none", 
						 legendentry="\${\\rm Beta}($(r4(beta.α)),$(r4(beta.β)))\$")
		push!(plots, p)
	end
    Axis(plots, xlabel=L"\theta", ylabel=L"P(\theta)",
		 style="enlarge x limits=0, ymin=0, legend pos=north west",
		 width="10cm", height="8cm", title="\$\\mu \\approx $(string(r4.(𝛍)))\$")
end

# ╔═╡ c1200740-f7e3-11ea-0edf-d1d27c46ed57
plot_beta(bt)

# ╔═╡ Cell order:
# ╟─bb2af0f2-f7e0-11ea-037c-45857eea5fd7
# ╠═41494860-f7e3-11ea-27bc-1b2515638b08
# ╠═4f7cd780-f7e3-11ea-13f9-79626222bcd6
# ╠═c59d5960-f7e0-11ea-21a0-d9ced93de3f5
# ╠═c0cb5ef0-f7e0-11ea-27d4-8b0dd09e98e1
# ╠═c1200740-f7e3-11ea-0edf-d1d27c46ed57
# ╠═ff09b860-f7e5-11ea-21a9-f90adbc47fa2
