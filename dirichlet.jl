### A Pluto.jl notebook ###
# v0.11.14

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

# ╔═╡ 9cf1baa0-075e-11eb-1e3f-c5db708222f6
using Pkg; Pkg.add("AddPackage"); using AddPackage

# ╔═╡ 4ae1a69e-075c-11eb-2e10-8b720ad50a4e
# pkg"add https://github.com/shashankp/PlutoUI.jl#TableOfContents-element"
@add using Distributions, LinearAlgebra, PyPlot, PlutoUI

# ╔═╡ 1daaa380-075c-11eb-245c-3f7054f453ad
md"""
# Dirichlet Distribution
The *dirichlet distribution* is a generalized beta distribution to multiple dimensions.

$$\operatorname{Dir}(\theta_{1:n} \mid \alpha_{1:n}) = \frac{\Gamma(\alpha_0)}{\prod_{i=1}^n \Gamma(\alpha_i)} \prod_{i=1}^n \theta_i^{\alpha_i - 1}$$

The Dirichlet distribution has a posterior of

$$p(\theta_{1:n} \mid \alpha_{1:n}, m_{1:n}) = \operatorname{Dir}(\theta_{1:n} \mid \alpha_1 + m_1, \ldots, \alpha_n + m_n)$$


and a mean vector $𝛍$, whose $i$th component is

$$\mu_i = \frac{\alpha_i}{\sum_{j=1}^n \alpha_j}.$$
"""

# ╔═╡ e39a9b60-075f-11eb-2b10-736d5fa68716
md"""
## Required Packages
"""

# ╔═╡ 05e5897e-0771-11eb-162a-9f45eea08fac
md"""
# 2-Parameter Dirichlet

This is equivalent to a Beta distribution.
"""

# ╔═╡ acded410-0773-11eb-0090-3fbad6834a0d
md"""
## Visualization (3D and 2D)
"""

# ╔═╡ 87f93370-084a-11eb-3941-f9b1c50e6b61
PyPlot.rc("text", usetex=true)

# ╔═╡ 312cca92-0771-11eb-1eb2-afbd5589fbe9
md"""
$(@bind α1 Slider(1:20, default=5, show_value=true))
$(@bind α2 Slider(1:20, default=5, show_value=true))
"""

# ╔═╡ 0be8b0f0-0771-11eb-264e-95a9e28a1677
D₂ = Dirichlet([α1, α2])

# ╔═╡ 2715a180-0771-11eb-064f-9dcd89dca83c
θ = rand(D₂, 10_000);

# ╔═╡ c035eaf0-0771-11eb-228c-3db7e1c0c551
(α1, α2)

# ╔═╡ 2b51c490-0771-11eb-15f6-ef92c4b1cbb4
begin
	close("all")
	hist2D(θ[1,:], θ[2,:], bins=100, density=true, range=[(0,1), (0,1)])
	colorbar(fraction=0.04, pad=0.01)
	xlabel(L"\theta_1")
	ylabel(L"\theta_2")
	title(LaTeXString("{\\rm Dir}($α1, $α2)"))
	gcf()
end

# ╔═╡ 83a4da10-0771-11eb-3ada-df8c2c726f5a
begin
	close("all")
	hist(θ[1,:], bins=20)
	xlabel(L"\theta_1")
	xlim([0,1])
	gcf()
end

# ╔═╡ 847cffb0-0773-11eb-0a22-11bc2c6e4c45
md"""
# 3-Parameter Dirichlet
The 3-parameter Dirchlet can be visualized using a triangle (i.e. simplex).
"""

# ╔═╡ eb4b72d0-075f-11eb-05ad-8ba7687c9dda
md"""
## Triangulation Functions
"""

# ╔═╡ 50416220-075c-11eb-3841-65e393a7ccca
md"""
Dirichlet simplex plot, modified from:
- [http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/](http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/)
"""

# ╔═╡ 48fa2c80-075d-11eb-2d4d-bf45f5ea2b76
begin
	corners = [0 0; 1 0; 0.5 sqrt(0.75)]
	area = prod(filter(x->x!=0, corners))
	pairs = [corners[circshift(1:3, -i+1)[2:end], :] for i in 1:3]
	triangle = matplotlib.tri.Triangulation(corners[:,1], corners[:,2])
	cross2d(X) = cross([X[:,1]..., 0], [X[:,2]..., 0])
	triarea(xy, pair) = 0.5norm(cross2d(pair .- [xy[1] xy[2]]))
	xy2bc(xy, tol=1e-4) = clamp.([triarea(xy, p) for p in pairs] / area, tol, 1-tol)
	refiner = matplotlib.tri.UniformTriRefiner(triangle)
	trimesh = refiner.refine_triangulation(subdiv=8)
end;

# ╔═╡ f6b13bf0-075f-11eb-0681-ed025305d2f1
md"""
## Draw Dirichlet Probability Density Function
"""

# ╔═╡ 43f97660-075c-11eb-01e2-9f90daed5bed
function drawpdf(dir::Dirichlet, nlevels=50; newfig=false)
	newfig && (close("all"), figure())
	pdfvals = [pdf(dir, xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

	tricontourf(trimesh, pdfvals, nlevels)

	axis("equal")
	axis("off")
	xlim([0, 1])
	ylim([0, sqrt(0.75)])
	try title(LaTeXString(string("{\\rm Dir}", Int.(tuple(dir.alpha...)))))
	catch; title(LaTeXString(string("{\\rm Dir}", tuple(dir.alpha...)))) end
	gcf()
end

# ╔═╡ 02146d00-0760-11eb-0213-ed9fd951fe05
md"""
## Visualization
"""

# ╔═╡ 4dd3ca60-0855-11eb-0c9e-eb87b67d21c1
savefig("dirichlet.svg")

# ╔═╡ 2ee63920-0775-11eb-1e69-b964d14996bb
md"""
Parameters $\theta_1$, $\theta_2$, and $\theta_3$ correspond to the corners going counter-clockwise, starting at the lower left.
"""

# ╔═╡ 9d3d5480-075c-11eb-3f72-dfd2b0e6dca3
md"""
$(@bind α₁ Slider(1:50, default=1, show_value=true))
$(@bind α₂ Slider(1:50, default=2, show_value=true))
$(@bind α₃ Slider(1:50, default=3, show_value=true))
"""

# ╔═╡ 67ae3d70-075c-11eb-2539-c3f202453a78
D = Dirichlet([α₁, α₂, α₃])

# ╔═╡ 72b285a0-075c-11eb-07a1-2f803d743902
drawpdf(D; newfig=true)

# ╔═╡ b63833c0-0765-11eb-1eac-6b7e0429c138
(α₁, α₂, α₃)

# ╔═╡ f245b660-075d-11eb-38a6-1982a0a47e69
# Markdown.parse("""
# \$\$\\theta_1 = $(θ₁),\\quad \\theta_2 = $(θ₂),\\quad \\theta_3 = $(θ₃)\$\$
# """)
md"*TODO:* MathJax scrolling bugfix."

# ╔═╡ b98a65a0-075d-11eb-3655-ad1491b4a4a9
md"""
## Dirichlet Subplots
"""

# ╔═╡ 745a04fe-0765-11eb-2761-01b1ef886839
begin
    figure()
    A = ([0.99,0.99,0.99], [1,1,1], [5,5,5], [1,2,3], [2,5,10], [50,50,50])
    for (i, α) in zip(1:6, A)
        subplot(2,3,i)
        drawpdf(Dirichlet(α))
    end
    tight_layout()
    gcf()
end

# ╔═╡ 2bc66300-085b-11eb-00f8-01a91c4f1197
savefig("dirichlet_subplots.png")

# ╔═╡ d580ff70-07b8-11eb-0718-597deb8aa280
PlutoUI.TableOfContents("Dirichlet Distribution")

# ╔═╡ Cell order:
# ╟─1daaa380-075c-11eb-245c-3f7054f453ad
# ╟─e39a9b60-075f-11eb-2b10-736d5fa68716
# ╠═9cf1baa0-075e-11eb-1e3f-c5db708222f6
# ╠═4ae1a69e-075c-11eb-2e10-8b720ad50a4e
# ╟─05e5897e-0771-11eb-162a-9f45eea08fac
# ╠═0be8b0f0-0771-11eb-264e-95a9e28a1677
# ╠═2715a180-0771-11eb-064f-9dcd89dca83c
# ╟─acded410-0773-11eb-0090-3fbad6834a0d
# ╠═87f93370-084a-11eb-3941-f9b1c50e6b61
# ╠═c035eaf0-0771-11eb-228c-3db7e1c0c551
# ╟─312cca92-0771-11eb-1eb2-afbd5589fbe9
# ╠═2b51c490-0771-11eb-15f6-ef92c4b1cbb4
# ╠═83a4da10-0771-11eb-3ada-df8c2c726f5a
# ╟─847cffb0-0773-11eb-0a22-11bc2c6e4c45
# ╟─eb4b72d0-075f-11eb-05ad-8ba7687c9dda
# ╟─50416220-075c-11eb-3841-65e393a7ccca
# ╠═48fa2c80-075d-11eb-2d4d-bf45f5ea2b76
# ╟─f6b13bf0-075f-11eb-0681-ed025305d2f1
# ╠═43f97660-075c-11eb-01e2-9f90daed5bed
# ╟─02146d00-0760-11eb-0213-ed9fd951fe05
# ╠═67ae3d70-075c-11eb-2539-c3f202453a78
# ╠═72b285a0-075c-11eb-07a1-2f803d743902
# ╠═4dd3ca60-0855-11eb-0c9e-eb87b67d21c1
# ╟─2ee63920-0775-11eb-1e69-b964d14996bb
# ╠═b63833c0-0765-11eb-1eac-6b7e0429c138
# ╟─9d3d5480-075c-11eb-3f72-dfd2b0e6dca3
# ╟─f245b660-075d-11eb-38a6-1982a0a47e69
# ╟─b98a65a0-075d-11eb-3655-ad1491b4a4a9
# ╠═745a04fe-0765-11eb-2761-01b1ef886839
# ╠═2bc66300-085b-11eb-00f8-01a91c4f1197
# ╠═d580ff70-07b8-11eb-0718-597deb8aa280
