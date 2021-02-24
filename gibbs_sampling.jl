### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 0fa22180-7635-11eb-1fd9-497f5f2a360b
using Distributions

# ╔═╡ 4f35d920-7637-11eb-1d6c-75971e0648ea
using Plots; plotly()

# ╔═╡ 1f0f65f0-7631-11eb-06bd-495554489442
md"""
# Gibbs Sampling
_Gibbs sampling_ is a Markov chain Monte Carlo (MCMC) technique to indirectly sample from a joint distribution $p(\mathbf x)$ by instead drawing samples iteratively from known conditional probability distributions $p(x_i \mid \mathbf{x}_{-i})$ where $\mathbf{x}_{-i}$ denotes all variables other than $x_i$.
"""

# ╔═╡ 67cceb4e-7631-11eb-28a9-adad7b848ee6
function gibbs_sampling(𝐱₀::Vector{Float64}, cpd::Vector{Function}, M=100)
    𝐗 = Vector(undef, M)
    D = length(𝐱₀)
    𝐱 = copy(𝐱₀)
    for t in 1:M
        for i in 1:D
            𝐱₋ᵢ = 𝐱[1:D .!= i]
            𝐱[i] = rand(cpd[i](𝐱₋ᵢ))
        end
        𝐗[t] = copy(𝐱)
    end
    return 𝐗
end

# ╔═╡ d17d3192-7631-11eb-20fe-b599a1fd564d
md"""
### Example: Multivariate Gaussian Distribution
Let $\mathbf{x}$ be a random variable whose joint distribution is a multivariate Gaussian:

$$p(\mathbf{x}) = \mathcal{N}(𝛍, 𝚺)$$

Let's consider the bivariate case, where:

$$\begin{gather}
𝛍 = [\mu_1, \mu_2]\\
𝚺 = \begin{bmatrix}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{21} & \Sigma_{22}
\end{bmatrix}
\end{gather}$$
"""

# ╔═╡ f6fe9af0-7634-11eb-1dbf-bdb18cc246d9
𝛍 = [0, 0]

# ╔═╡ fc176260-7634-11eb-3c46-cd4490784878
𝚺 = [1   0.8;
     0.8   1]

# ╔═╡ 5c8a76c0-7633-11eb-1922-8dd43ec92691
p = MvNormal(𝛍, 𝚺); # Illustrative example: p(𝐱) is "hard" to compute directly.

# ╔═╡ 1a779180-7635-11eb-130f-59c7f1dae88f
md"""
For the sake of illustration, consider the case where we _cannot_ directly sample from $p(\mathbf{x})$, but we _do have_ the conditional distributions:$(html"<sup><a href='https://faculty.wcas.northwestern.edu/~lchrist/course/Korea_2016/note_on_Gibbs_sampling.pdf'>[1]</a></sup>")

$$
\begin{align}
p(x_1 \mid x_2^{(t-1)}) &= \mathcal{N}\left(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2^{(t-1)} - \mu_2),\, \frac{\Sigma_{11}\Sigma_{22} - \Sigma_{12}^2}{\Sigma_{22}}\right)\\
p(x_2 \mid x_1^{(t)}) &= \mathcal{N}\left(\mu_2 + \Sigma_{21}\Sigma_{11}^{-1}(x_1^{(t)} - \mu_1),\, \frac{\Sigma_{11}\Sigma_{22} - \Sigma_{21}^2}{\Sigma_{11}}\right)

\end{align}$$
"""

# ╔═╡ f61afc40-7635-11eb-0d7c-4d1bf60afe0c
p_x₁(x₂) =
	Normal(𝛍[1] + 𝚺[1,2]/𝚺[2,2]*(x₂ - 𝛍[2]), sqrt((𝚺[1,1]*𝚺[2,2] - 𝚺[1,2]^2)/𝚺[2,2]));

# ╔═╡ 5d06cb4e-7636-11eb-2c14-2548356935f6
p_x₁(x₂::Vector) = p_x₁(x₂...); # turn [x₂] into x₂

# ╔═╡ 750f9880-7636-11eb-145d-6767745bffcd
md"""
---
"""

# ╔═╡ 77155390-7636-11eb-0fbb-d90b0cada385
p_x₂(x₁) =
	Normal(𝛍[2] + 𝚺[2,1]/𝚺[1,1]*(x₁ - 𝛍[1]), sqrt((𝚺[1,1]*𝚺[2,2] - 𝚺[2,1]^2)/𝚺[1,1]));

# ╔═╡ d8a42640-7636-11eb-2d6b-530a8c511287
p_x₂(x₁::Vector) = p_x₂(x₁...); # turn [x₁] into x₁

# ╔═╡ fb128aa0-7636-11eb-3f37-dbcd90d77f76
md"""
### Example: Running Gibbs Sampling
"""

# ╔═╡ 032afd80-7637-11eb-0212-1b3ed5d598bb
x₀ = rand(2) # random initial state/point

# ╔═╡ bd474b10-7641-11eb-32b5-39c33a64f537
CPDs = [p_x₁, p_x₂];

# ╔═╡ 097259de-7637-11eb-0582-e94ec1278274
X = gibbs_sampling(x₀, CPDs, 500);

# ╔═╡ 206a2c40-7637-11eb-2fb1-a1c3fddccc90
function plot_gibbs(x₀, X; p=nothing, lims=[])
    # Plot Gibbs samples and chain
    if isnothing(p)
        α = 1
        plot(aspect_ratio=:equal)
    else
        α = 0.5
        # Plot true joint distribution
        f = (x,y)->pdf(p, [x,y])
        contourf(lims, lims, f, color=:viridis)
    end

    # Plot chain (including 𝐱₀)
    X = [x₀, X...]
    for i in 1:length(X)-1
        x, x′ = X[i], X[i+1]
		Xs = [x[1], x[1], x′[1]]
		Ys = [x[2], x′[2], x′[2]]
        plot!(Xs, Ys, color=:gray, alpha=α, label=i==1 ? "Markov chain" : nothing)
    end

    # Plot individual Gibbs samples
    scatter!(first.(X), last.(X), label="Gibbs Sample",
		     ms=2, color=:red, alpha=α, legend=:bottomright)

    # Plot initial state
    scatter!([x₀[1]], [x₀[2]], ms=3, color=:yellow, label="Initial x")
end

# ╔═╡ 3d266f60-7637-11eb-0ff1-0d92a95f3a19
plot_gibbs(x₀, X; p=p, lims=range(-4, 4, length=100)) # true distribution for contour

# ╔═╡ a61be590-7641-11eb-142f-7391a6423548
plot_gibbs(x₀, X) # only plot Gibbs samples

# ╔═╡ Cell order:
# ╟─1f0f65f0-7631-11eb-06bd-495554489442
# ╠═67cceb4e-7631-11eb-28a9-adad7b848ee6
# ╟─d17d3192-7631-11eb-20fe-b599a1fd564d
# ╠═0fa22180-7635-11eb-1fd9-497f5f2a360b
# ╠═f6fe9af0-7634-11eb-1dbf-bdb18cc246d9
# ╠═fc176260-7634-11eb-3c46-cd4490784878
# ╠═5c8a76c0-7633-11eb-1922-8dd43ec92691
# ╟─1a779180-7635-11eb-130f-59c7f1dae88f
# ╠═f61afc40-7635-11eb-0d7c-4d1bf60afe0c
# ╠═5d06cb4e-7636-11eb-2c14-2548356935f6
# ╟─750f9880-7636-11eb-145d-6767745bffcd
# ╠═77155390-7636-11eb-0fbb-d90b0cada385
# ╠═d8a42640-7636-11eb-2d6b-530a8c511287
# ╟─fb128aa0-7636-11eb-3f37-dbcd90d77f76
# ╠═032afd80-7637-11eb-0212-1b3ed5d598bb
# ╠═bd474b10-7641-11eb-32b5-39c33a64f537
# ╠═097259de-7637-11eb-0582-e94ec1278274
# ╠═4f35d920-7637-11eb-1d6c-75971e0648ea
# ╠═3d266f60-7637-11eb-0ff1-0d92a95f3a19
# ╠═a61be590-7641-11eb-142f-7391a6423548
# ╠═206a2c40-7637-11eb-2fb1-a1c3fddccc90
