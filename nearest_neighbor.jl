### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 6ac547f0-084d-11eb-2e0b-5df1f4ef3e0c
using LinearAlgebra

# ╔═╡ 9fc52e30-f7c1-11ea-3b59-9b5d472b5135
begin
	using ColorSchemes
	using PGFPlots
	resetPGFPlotsOptions()
	pushPGFPlotsOptions("scale=2")
	viridis_r = ColorMaps.RGBArrayMap(ColorSchemes.viridis,
		                              interpolation_levels=500, invert=true)
end;

# ╔═╡ 1ea2dac2-084f-11eb-29c0-d3a4d75c2d11
using Random

# ╔═╡ f5fbf900-084c-11eb-0082-2ff3282fff2d
md"""
# Nearest Neighbor
"""

# ╔═╡ 9ef8fa00-f7c0-11ea-2b67-edc493913ec1
function nearest_neighbor(x′, φ, 𝒟train, dist)
	𝒟train[argmin([dist(φ(x), φ(x′)) for (x,y) in 𝒟train])][end]
end

# ╔═╡ 91ed2c90-f7c1-11ea-27ae-176ef4e7f9f2
md"# Voronoi Diagram"

# ╔═╡ ab1a0760-f7c1-11ea-3ea6-d15ea43db107
function voronoi(𝒟train, dist; xlim=(0, 10), ylim=(0, 10))
	f = (x1,x2)->nearest_neighbor([x1,x2], 𝐱->𝐱, 𝒟train, dist)

	Axis([Plots.Image(f, xlim, ylim, colorbar=false,
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

# ╔═╡ 4450767e-084d-11eb-228e-5b9035731336
md"""
## Mahnattan distance

Also called the $L_1$ distance.

$$d(\mathbf v, \mathbf v^\prime) = \lVert \mathbf v - \mathbf v^\prime \rVert_1 = \sum_{i=1}^n \lvert v_i - v_i^\prime \rvert$$
"""

# ╔═╡ 6e21e070-084d-11eb-24af-418d629b4cdf
dist_manhattan(𝐯, 𝐯′) = norm(𝐯 - 𝐯′, 1)

# ╔═╡ 66dd90fe-084e-11eb-0725-29715e054e1e
voronoi(𝒟train, dist_manhattan)

# ╔═╡ ccfe3d50-084d-11eb-2345-5fa46c8cf1ca
md"""
## Euclidian distance
Also called the $L_2$ distance.

$$d(\mathbf v, \mathbf v^\prime) = \lVert \mathbf v - \mathbf v^\prime \rVert_2 = \sqrt{ \sum_{i=1}^n \left(v_i - v_i^\prime \right)^2}$$
"""

# ╔═╡ 732fd270-084d-11eb-39fa-078840bdc098
dist_euclidean(𝐯, 𝐯′) = norm(𝐯 - 𝐯′, 2)

# ╔═╡ 12970800-f7c4-11ea-35c4-058dfb1e9963
voronoi(𝒟train, dist_euclidean)

# ╔═╡ 9cef4590-084e-11eb-245e-fdb1c7ec7999
md"""
## Supremum distance
Also called the $L_\infty$ distance or the Chebyshev distance.

$$d(\mathbf v, \mathbf v^\prime) = \lVert \mathbf v - \mathbf v^\prime \rVert_\infty = \lim_{p\to\infty} \left( \sum_{i=1}^n \lvert v_i - v_i^\prime \rvert^p \right)^{\!^1/_p}$$
"""

# ╔═╡ 77bc5f70-084d-11eb-3cda-b1ae821d5bcd
dist_supremum(𝐯, 𝐯′)  = norm(𝐯 - 𝐯′, Inf)

# ╔═╡ 1608af20-f7c4-11ea-1ec0-8f8ca4d23a89
voronoi(𝒟train, dist_supremum)

# ╔═╡ e420ce20-f7c3-11ea-1e96-c7492a01001a
md"# Random datapoints"

# ╔═╡ 2087c760-084f-11eb-15ed-b31fbfdd384b
Random.seed!(0);

# ╔═╡ b6b25b00-0850-11eb-2c76-b9f7a2140522
md"""
## Sorted along $2$nd dimension
"""

# ╔═╡ 22614250-084f-11eb-3d83-cf86d51471eb
X = sort!(rand(10,2); dims=2)

# ╔═╡ 415c78f0-084f-11eb-1811-3187727eec99
Y = 1:size(X)[1]

# ╔═╡ b50a93e0-084f-11eb-02cb-8fcb3b711c96
𝒟train_rand = [(X[i,:], Y[i]) for i in 1:length(Y)]

# ╔═╡ 1ab4e400-f7c2-11ea-09d7-2d0021e7c43f
voronoi(𝒟train_rand, dist_euclidean; xlim=(0, 1), ylim=(0, 1))

# ╔═╡ 8a620e60-0850-11eb-3969-c91f0ac25a15
md"""
## Sorted along $1$st dimension
"""

# ╔═╡ dd4bf280-0850-11eb-3e78-c54ae25fca17
X1 = sort!(rand(10,2); dims=1)

# ╔═╡ eb5bf410-0850-11eb-2770-513f1614ce42
𝒟train_rand1 = [(X1[i,:], Y[i]) for i in 1:length(Y)]

# ╔═╡ 98d4f070-0850-11eb-1ae7-af5d983c9bbb
voronoi(𝒟train_rand1, dist_euclidean; xlim=(0, 1), ylim=(0, 1))

# ╔═╡ Cell order:
# ╟─f5fbf900-084c-11eb-0082-2ff3282fff2d
# ╠═6ac547f0-084d-11eb-2e0b-5df1f4ef3e0c
# ╠═9ef8fa00-f7c0-11ea-2b67-edc493913ec1
# ╟─91ed2c90-f7c1-11ea-27ae-176ef4e7f9f2
# ╠═9fc52e30-f7c1-11ea-3b59-9b5d472b5135
# ╠═ab1a0760-f7c1-11ea-3ea6-d15ea43db107
# ╠═436378d0-f7c2-11ea-03b0-5986e2aa7a0e
# ╟─4450767e-084d-11eb-228e-5b9035731336
# ╠═6e21e070-084d-11eb-24af-418d629b4cdf
# ╠═66dd90fe-084e-11eb-0725-29715e054e1e
# ╟─ccfe3d50-084d-11eb-2345-5fa46c8cf1ca
# ╠═732fd270-084d-11eb-39fa-078840bdc098
# ╠═12970800-f7c4-11ea-35c4-058dfb1e9963
# ╟─9cef4590-084e-11eb-245e-fdb1c7ec7999
# ╠═77bc5f70-084d-11eb-3cda-b1ae821d5bcd
# ╠═1608af20-f7c4-11ea-1ec0-8f8ca4d23a89
# ╟─e420ce20-f7c3-11ea-1e96-c7492a01001a
# ╠═1ea2dac2-084f-11eb-29c0-d3a4d75c2d11
# ╠═2087c760-084f-11eb-15ed-b31fbfdd384b
# ╟─b6b25b00-0850-11eb-2c76-b9f7a2140522
# ╠═22614250-084f-11eb-3d83-cf86d51471eb
# ╠═415c78f0-084f-11eb-1811-3187727eec99
# ╠═b50a93e0-084f-11eb-02cb-8fcb3b711c96
# ╠═1ab4e400-f7c2-11ea-09d7-2d0021e7c43f
# ╟─8a620e60-0850-11eb-3969-c91f0ac25a15
# ╠═dd4bf280-0850-11eb-3e78-c54ae25fca17
# ╠═eb5bf410-0850-11eb-2770-513f1614ce42
# ╠═98d4f070-0850-11eb-1ae7-af5d983c9bbb
