### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ faddc480-fd20-11ea-291f-a949393ca57e
using AddPackage

# ╔═╡ 1bb367d0-fd1e-11ea-25fc-1d02eef55e1c
@add using Statistics, LinearAlgebra, PlutoUI, RDatasets, Plots, Random

# ╔═╡ 1a6a4600-fd1e-11ea-0e9b-dde0b5b7835f
using BeautifulAlgorithms

# ╔═╡ 5888d030-084c-11eb-35e7-a35be9959eca
md"""
# K-means Clustering
"""

# ╔═╡ 400fdc10-084c-11eb-2a2e-876c4d8257ae
md"![](https://raw.githubusercontent.com/mossr/BeautifulAlgorithms.jl/7236f2a65db95ec984ead63e0d000d745f110420/img/svg/k_means_clustering.svg)"

# ╔═╡ 62f93cce-084c-11eb-297d-853d1ea628d4
md"""
## Dataset (iris)
"""

# ╔═╡ 02d87630-fd21-11ea-376a-df5f1bec80c1
iris = dataset("datasets", "iris")

# ╔═╡ 9e46878e-fd1e-11ea-249f-6770cad19550
plotly()

# ╔═╡ 070f8170-fd27-11ea-12d4-e94d41145016
xlabel, ylabel = :PetalLength, :PetalWidth

# ╔═╡ 6c332540-084c-11eb-06e2-19221a34fca0
md"""
## True labels
"""

# ╔═╡ fcffc4b0-fd21-11ea-1eb1-35b64224f59a
scatter(iris[xlabel], iris[ylabel],
	    color=map(s->Int(s.level), iris[:Species]), lab="")

# ╔═╡ 7d24c2f0-084c-11eb-0d24-ab765aaaaf68
md"""
## Learned clusters
"""

# ╔═╡ cd225250-fd24-11ea-3d69-77665023ba88
K = length(unique(iris[:Species]))

# ╔═╡ a0dec1d0-fd22-11ea-3a8b-4ba4a497b497
D = map((l,w,s)->([l,w], Int(s.level)), iris[xlabel], iris[ylabel], iris[:Species])

# ╔═╡ a7d7d010-fd24-11ea-02ad-ad2a64d9a10d
@bind T Slider(1:20, show_value=true, default=20)

# ╔═╡ d2299f80-fd22-11ea-25f3-cb8d35d54a60
(z, μ) = k_means_clustering(x->x, D, K; T=T)

# ╔═╡ 0698d230-fd24-11ea-09b5-238ea2484cfe
begin
	Random.seed!(2)
	plt = scatter()
	for k in keys(z)
		scatter!(first.(first.(D[z[k]])), last.(first.(D[z[k]])), 
			     color=k, alpha=1)
	end
	scatter!(first.(μ), last.(μ),
		     marker=:xcross, color=1:length(μ), markersize=7, alpha=0.8,
		     markerstrokewidth=3)
	plt
end

# ╔═╡ Cell order:
# ╟─5888d030-084c-11eb-35e7-a35be9959eca
# ╠═faddc480-fd20-11ea-291f-a949393ca57e
# ╠═1bb367d0-fd1e-11ea-25fc-1d02eef55e1c
# ╠═1a6a4600-fd1e-11ea-0e9b-dde0b5b7835f
# ╟─400fdc10-084c-11eb-2a2e-876c4d8257ae
# ╟─62f93cce-084c-11eb-297d-853d1ea628d4
# ╠═02d87630-fd21-11ea-376a-df5f1bec80c1
# ╠═9e46878e-fd1e-11ea-249f-6770cad19550
# ╠═070f8170-fd27-11ea-12d4-e94d41145016
# ╟─6c332540-084c-11eb-06e2-19221a34fca0
# ╠═fcffc4b0-fd21-11ea-1eb1-35b64224f59a
# ╟─7d24c2f0-084c-11eb-0d24-ab765aaaaf68
# ╠═cd225250-fd24-11ea-3d69-77665023ba88
# ╠═a0dec1d0-fd22-11ea-3a8b-4ba4a497b497
# ╠═a7d7d010-fd24-11ea-02ad-ad2a64d9a10d
# ╠═d2299f80-fd22-11ea-25f3-cb8d35d54a60
# ╠═0698d230-fd24-11ea-09b5-238ea2484cfe
