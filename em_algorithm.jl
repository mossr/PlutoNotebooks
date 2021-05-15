### A Pluto.jl notebook ###
# v0.14.2

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

# ╔═╡ d01ec170-7574-45bc-88f9-d7183e998b52
using Distributions, LinearAlgebra

# ╔═╡ 350117c4-6b1d-4e84-ab53-90038704d7c3
using Plots, PlutoUI, Random

# ╔═╡ 57486310-b405-11eb-03d6-e11d748d9884
md"""
# The EM Algorithm
Applied to the mixture of Gaussians. Following from [CS229 lecture notes](http://cs229.stanford.edu/notes2020spring/cs229-notes7b.pdf).
"""

# ╔═╡ 58466585-80fa-41df-af1a-0015642831e7
md"""
Given a training dataset $\{x^{(1)}, \ldots, x^{(n)}\}$, we can model the data by specifying the joint distribution:

$$p(x^{(i)}, z^{(i)}) = p(x^{(i)} \mid z^{(i)}) p(z^{(i)})$$
where $z^{(i)}$ is a latent variable. We model the random variables as:

$$\begin{align}
z^{(i)} &\sim \operatorname{Multinomial}(\phi)\\
(x^{(i)} \mid z^{(i)}=j) &\sim \mathcal{N}(\mu_j, \Sigma_j)
\end{align}$$
where $\phi_j$ gives $p(z^{(i)} = j)$ and assumes our data $x^{(i)}$ given $z^{(i)}=j$ is from a normal distribution.
"""

# ╔═╡ 94162b68-dcd7-4701-ba73-00acee7ec1e9
md"""
The parameters are unknown to us (i.e., how the data was _actually_ generated). We want to learn these!
"""

# ╔═╡ 77c68aed-7c38-43c7-8386-7ff3bfb94245
begin
	ϕ = [0.2, 0.8]
	z = Multinomial(1, ϕ)
	x_z₁ = MvNormal([1, 1], [2 0; 0 2])
	x_z₂ = MvNormal([4, 4], [1 0.5; 0.5 1])
end;

# ╔═╡ d69eaa9d-c9b4-4b92-85d5-ee239a692c68
md"""
## Plotting
"""

# ╔═╡ f11b9758-939b-46e8-9b32-2aaa6c1038cf
n = 1000;

# ╔═╡ 8ca1b48a-1fbb-4658-9a2f-5b6b399c149f
begin
	x₁_samples = []
	x₂_samples = []
	for i in 1:n
		# Sample from Multinomial to determine which Gaussian to pick
		if rand(z)[1] == 1
			push!(x₁_samples, rand(x_z₁)) # Sample from (x | z = 1)
		else
			push!(x₂_samples, rand(x_z₂)) # Sample from (x | z = 2)
		end
	end
end

# ╔═╡ da1359c7-17cc-40c3-a111-b73610045b52
md"""
Here we plot the true labels (again, unknown to us).
"""

# ╔═╡ 7041994f-bf1f-4f31-8588-3a1043f3c33d
begin
	getx₁(data) = map(xᵢ->xᵢ[1], data)
	getx₂(data) = map(xᵢ->xᵢ[2], data)
	label = i -> "\$y=$i\$"
	scatter(getx₁(x₁_samples), getx₂(x₁_samples), label=label(1), c=:black)
	scatter!(getx₁(x₂_samples), getx₂(x₂_samples), label=label(2), c=:crimson, legend=:topleft)
end

# ╔═╡ d7a9ec8a-8c18-4326-be07-6f1f0669eef7
md"""
## Collect input data $x^{(i)}$
"""

# ╔═╡ 2613e239-a226-41de-b37c-5ee13b6e6c10
md"""
Combine and shuffle generated input data $x^{(i)} \in \{x^{(1)}, \ldots, x^{(n)}\}$. Shuffling is unnecessary, but is used to demonstrate that no inherent patterns exist in the data.
"""

# ╔═╡ d86dffc6-6aab-4b83-8b18-a6ba56d2b6d6
begin
	permutation = randperm(n)
	x = vcat(x₁_samples, x₂_samples)[permutation]
	y = vcat(ones(length(x₁_samples)), 2ones(length(x₂_samples)))[permutation]
end;

# ╔═╡ 1976cce1-75c4-477d-a343-12bd1acc8a6c
md"""
## E-step
For each $i$, $j$, set the weights $w_j^{(i)}$ as the expected log-likelihood of the current parameter estimates:

$$\begin{align}
w_j^{(i)} &:= p(z^{(i)} = j \mid x^{(i)}; \phi, \mu, \Sigma)\\
	      &= \frac{p(x^{(i)} \mid z^{(i)}=j; \mu, \Sigma)p(z^{(i)}=j; \phi)}{\sum_{l=1}^k p(x^{(i)} \mid z^{(i)}=l; \mu, \Sigma)p(z^{(i)}=l; \phi)}\tag{Bayes' rule}\\
		  &= \frac{\operatorname{pdf}(\mathcal{N}(\mu_j, \Sigma_j), x^{(i)})\phi_j}{\sum_{l=1}^k \operatorname{pdf}(\mathcal{N}(\mu_l, \Sigma_l), x^{(i)})\phi_l}\tag{pdf notation}
\end{align}$$
"""

# ╔═╡ dc1df6b7-27e8-401f-942a-6375d520e1a8
function e_step(θ, x)
	ϕ, μ, Σ = θ.ϕ, θ.μ, θ.Σ
	n = length(x)
	k = length(ϕ)
	w = Matrix{Real}(undef, n, k)
	for i in 1:n
		for j in 1:k
			w[i,j] = (pdf(MvNormal(μ[j], Σ[j]), x[i]) * ϕ[j]) /
				sum(pdf(MvNormal(μ[l], Σ[l]), x[i]) * ϕ[l] for l in 1:k)
		end
	end
	return w
end

# ╔═╡ cd95ec82-9503-4e28-bc53-5a6b80025121
md"""
## M-step
Update the parameters $\theta = \langle \phi, \mu, \Sigma \rangle$ using the maximum likelihood estimate (MLE):

$$\begin{align}
\phi_j &= \frac{1}{n} \sum_{i=1}^n w_j^{(i)}\\
\mu_j &= \frac{\sum_{i=1}^n w_j^{(i)} x^{(i)}}{\sum_{i=1}^n w_j^{(i)}}\\
\Sigma_j &= \frac{\sum_{i=1}^n w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^\top}{\sum_{i=1}^n w_j^{(i)}}\\
\end{align}$$
"""

# ╔═╡ 77e18da8-c41d-4a68-9825-a09658e169a1
function m_step!(θ, w, x)
	ϕ, μ, Σ = θ.ϕ, θ.μ, θ.Σ
	n = length(x)
	k = length(ϕ)
	for j in 1:k
		sum_w = sum(w[i,j] for i in 1:n)
		ϕ[j] = 1/n * sum_w
		μ[j] = sum(w[i,j]*x[i] for i in 1:n) / sum_w
		Σ[j] = Hermitian(sum(w[i,j]*(x[i]-μ[j])*(x[i]-μ[j])' for i in 1:n) / sum_w)
	end
	return θ
end

# ╔═╡ 9c99b7be-1296-44fd-8bd8-e1b00803760a
md"""
## Run to convergence
"""

# ╔═╡ d2b0e15d-fd2b-4128-821e-333bf0a4517d
md"""
## Classification
For each data point $x^{(i)}$, which bivariate Gaussian is it more likely to come from?

$$\hat{y}^{(i)} = \operatorname*{arg\;max}_{j \in \{1,\ldots,k\}} p(x^{(i)}; \mu_j, \Sigma_j)$$
"""

# ╔═╡ bb358b2e-c5a5-4d7a-98e3-c87f0ee60f26
classify(xᵢ, θ) = argmax([pdf(MvNormal(θ.μ[j], θ.Σ[j]), xᵢ) for j in 1:length(θ.μ)])

# ╔═╡ d8999eb7-086c-4a02-8323-44a0d5adc536
@bind max_iterations Slider(0:200, default=100)

# ╔═╡ 2d062f8c-8344-4a96-bc9e-00f493e20adc
begin
	Random.seed!(1)
	ϕ̂ = [0.5, 0.5] # Implied k=2
	μ̂ = [randn(2), randn(2)]
	Σ̂ = [randn(2,2), randn(2,2)]
	Σ̂[1] = Hermitian(Σ̂[1]'Σ̂[1] + I) # Ensure symmetric, PSD, and Hermitian
	Σ̂[2] = Hermitian(Σ̂[2]'Σ̂[2] + I)
	θ = (ϕ=ϕ̂, μ=μ̂, Σ=Σ̂) # Full parameters
	tolerance = 1e-12
	# max_iterations = 100_000 # Defined by the slider below

	for iter in 1:max_iterations
		θ₋₁ = deepcopy(θ)
		w = e_step(θ, x)
		m_step!(θ, w, x)

		if all([norm(θ₋₁.μ - θ.μ), norm(θ₋₁.Σ - θ.Σ), norm(θ₋₁.ϕ - θ.ϕ)] .< tolerance)
			@info "Converged at iteration $iter"
			break
		end
	end

	θ
end

# ╔═╡ ec100ecb-7a63-4dda-addd-cf07c937ac31
ŷ = map(xᵢ -> classify(xᵢ, θ), x);

# ╔═╡ fbf2598f-00fe-4eb5-85a7-34a73e41cb4a
begin
	XY_range = range(minimum(getx₁(x)), maximum(getx₂(x)), length=100)
	f₁(x,y) = pdf(MvNormal(θ.μ[1], θ.Σ[1]), [x,y])
	f₂(x,y) = pdf(MvNormal(θ.μ[2], θ.Σ[2]), [x,y])

	label_hat = i -> "\$\\hat{y}=$i\$"
	scatter(getx₁(x[ŷ .== 1]), getx₂(x[ŷ .== 1]), label=label_hat(1), c=:black)
	scatter!(getx₁(x[ŷ .== 2]), getx₂(x[ŷ .== 2]), label=label_hat(2), c=:crimson)
	contour!(XY_range, XY_range, f₁, color=:viridis, l=2)
	contour!(XY_range, XY_range, f₂, color=:cividis, l=2, legend=:topleft, cb=false)
end

# ╔═╡ 4766f4bf-f39e-40a3-98f8-78a01ef73e1b
md"""
### Accuracy
"""

# ╔═╡ 3313531a-2745-4fa7-9685-216a3daf26b2
sum(y .== ŷ) / length(y)

# ╔═╡ Cell order:
# ╟─57486310-b405-11eb-03d6-e11d748d9884
# ╠═d01ec170-7574-45bc-88f9-d7183e998b52
# ╟─58466585-80fa-41df-af1a-0015642831e7
# ╟─94162b68-dcd7-4701-ba73-00acee7ec1e9
# ╠═77c68aed-7c38-43c7-8386-7ff3bfb94245
# ╟─d69eaa9d-c9b4-4b92-85d5-ee239a692c68
# ╠═350117c4-6b1d-4e84-ab53-90038704d7c3
# ╠═f11b9758-939b-46e8-9b32-2aaa6c1038cf
# ╠═8ca1b48a-1fbb-4658-9a2f-5b6b399c149f
# ╟─da1359c7-17cc-40c3-a111-b73610045b52
# ╠═7041994f-bf1f-4f31-8588-3a1043f3c33d
# ╟─d7a9ec8a-8c18-4326-be07-6f1f0669eef7
# ╟─2613e239-a226-41de-b37c-5ee13b6e6c10
# ╠═d86dffc6-6aab-4b83-8b18-a6ba56d2b6d6
# ╟─1976cce1-75c4-477d-a343-12bd1acc8a6c
# ╠═dc1df6b7-27e8-401f-942a-6375d520e1a8
# ╟─cd95ec82-9503-4e28-bc53-5a6b80025121
# ╠═77e18da8-c41d-4a68-9825-a09658e169a1
# ╟─9c99b7be-1296-44fd-8bd8-e1b00803760a
# ╠═2d062f8c-8344-4a96-bc9e-00f493e20adc
# ╟─d2b0e15d-fd2b-4128-821e-333bf0a4517d
# ╠═bb358b2e-c5a5-4d7a-98e3-c87f0ee60f26
# ╠═ec100ecb-7a63-4dda-addd-cf07c937ac31
# ╠═d8999eb7-086c-4a02-8323-44a0d5adc536
# ╠═fbf2598f-00fe-4eb5-85a7-34a73e41cb4a
# ╟─4766f4bf-f39e-40a3-98f8-78a01ef73e1b
# ╠═3313531a-2745-4fa7-9685-216a3daf26b2
