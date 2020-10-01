### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 9ef8fa00-f7c0-11ea-2b67-edc493913ec1
begin
	using LinearAlgebra

	dist_manhattan(𝐯, 𝐯′) = norm(𝐯 - 𝐯′, 1)
	dist_euclidean(𝐯, 𝐯′) = norm(𝐯 - 𝐯′, 2)
	dist_supremum(𝐯, 𝐯′)  = norm(𝐯 - 𝐯′, Inf)

	function nearest_neighbor(x′, φ, 𝒟train, dist)
		𝒟train[argmin([dist(φ(x), φ(x′)) for (x,y) in 𝒟train])][end]
	end
end

# ╔═╡ 9fc52e30-f7c1-11ea-3b59-9b5d472b5135
begin
	using ColorSchemes
	using PGFPlots
	resetPGFPlotsOptions()
	pushPGFPlotsOptions("scale=2")
	viridis_r = ColorMaps.RGBArrayMap(ColorSchemes.viridis,
		                              interpolation_levels=500, invert=true)
end;

# ╔═╡ 91ed2c90-f7c1-11ea-27ae-176ef4e7f9f2
md"# Voronoi Diagram"

# ╔═╡ ab1a0760-f7c1-11ea-3ea6-d15ea43db107
function voronoi(𝒟train, dist)
	f = (x1,x2)->nearest_neighbor([x1,x2], 𝐱->𝐱, 𝒟train, dist)

	Axis([Plots.Image(f, (0,10), (0,10), colorbar=false,
                      xbins=200, ybins=200, colormap = viridis_r),
		  Plots.Scatter(map(d->d[1][1], 𝒟train), map(d->d[1][2], 𝒟train); 
                        onlyMarks=true, mark="*", markSize=1,
                        style="mark options={fill=black}, white"),
	], width="5cm", height="5cm", style="xticklabels={,,}, yticklabels={,,},")
end

# ╔═╡ 436378d0-f7c2-11ea-03b0-5986e2aa7a0e
𝒟train = [([1,5],1), 
		  ([1,6],2), 
		  ([2,8],3), 
		  ([3,7],4), 
		  ([3,6],5), 
		  ([5,9],6), 
		  ([6,2],7),
		  ([7,5],8), 
		  ([8,3],9), 
		  ([9,9],10)];

# ╔═╡ 0e6c982e-f7c4-11ea-2463-ab7c99ff9c2c
voronoi(𝒟train, dist_manhattan)

# ╔═╡ 12970800-f7c4-11ea-35c4-058dfb1e9963
voronoi(𝒟train, dist_manhattan)

# ╔═╡ 1608af20-f7c4-11ea-1ec0-8f8ca4d23a89
voronoi(𝒟train, dist_supremum)

# ╔═╡ e420ce20-f7c3-11ea-1e96-c7492a01001a
md"##### Testing to find difference between the three distance metrics."

# ╔═╡ aea61ede-f7c2-11ea-3ed3-b900c11c1287
X = [6.1,6.5] # [9,6.5]

# ╔═╡ f1000d50-f7c2-11ea-24e3-eda93949e99c
nearest_neighbor(X, x->x, 𝒟train, dist_manhattan)

# ╔═╡ efe52770-f7c2-11ea-0524-695060959424
nearest_neighbor(X, x->x, 𝒟train, dist_euclidean)

# ╔═╡ b2bf298e-f7c2-11ea-396d-5b65a2639bf9
nearest_neighbor(X, x->x, 𝒟train, dist_supremum)

# ╔═╡ c6744980-f7c1-11ea-2696-29c13bbae2cb
𝒟train_test = [([5,9],6), 
               ([5,5],7),
               ([7,5],8), 
               ([9,9],10)];

# ╔═╡ 1ab4e400-f7c2-11ea-09d7-2d0021e7c43f
voronoi(𝒟train_test, dist_manhattan)

# ╔═╡ 21c77fa0-f7c2-11ea-33f9-f5159ed14231
voronoi(𝒟train_test, dist_euclidean)

# ╔═╡ 27d29650-f7c2-11ea-0ba8-b37c4f392023
voronoi(𝒟train_test, dist_supremum)

# ╔═╡ Cell order:
# ╠═9ef8fa00-f7c0-11ea-2b67-edc493913ec1
# ╟─91ed2c90-f7c1-11ea-27ae-176ef4e7f9f2
# ╠═9fc52e30-f7c1-11ea-3b59-9b5d472b5135
# ╠═ab1a0760-f7c1-11ea-3ea6-d15ea43db107
# ╠═436378d0-f7c2-11ea-03b0-5986e2aa7a0e
# ╠═0e6c982e-f7c4-11ea-2463-ab7c99ff9c2c
# ╠═12970800-f7c4-11ea-35c4-058dfb1e9963
# ╠═1608af20-f7c4-11ea-1ec0-8f8ca4d23a89
# ╟─e420ce20-f7c3-11ea-1e96-c7492a01001a
# ╠═aea61ede-f7c2-11ea-3ed3-b900c11c1287
# ╠═f1000d50-f7c2-11ea-24e3-eda93949e99c
# ╠═efe52770-f7c2-11ea-0524-695060959424
# ╠═b2bf298e-f7c2-11ea-396d-5b65a2639bf9
# ╠═c6744980-f7c1-11ea-2696-29c13bbae2cb
# ╠═1ab4e400-f7c2-11ea-09d7-2d0021e7c43f
# ╠═21c77fa0-f7c2-11ea-33f9-f5159ed14231
# ╠═27d29650-f7c2-11ea-0ba8-b37c4f392023
