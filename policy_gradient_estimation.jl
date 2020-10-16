### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ b039225e-0f65-11eb-0925-337baf99a1f2
using Pkg; Pkg.add("AddPackage"); using AddPackage

# ╔═╡ 7e7a7440-0f65-11eb-14e5-e1281ae10f18
@add using Distributions

# ╔═╡ 96f42b80-0f40-11eb-1fde-67a6c8b9aad4
md"""
# Policy Gradient Estimation
To make improvements to a policy $\pi$, one way to inform *how* to improve the policy is to estimate the utility gradient $\nabla U$ with respect to the policy parameters $𝛉$ to guide the optimization process.
"""

# ╔═╡ a2114652-0f55-11eb-3324-81f9b428185a
md"""
## Markov Decision Process (MDP)
The 5-tuple that defines a *Markov decision process* is the following.

$$\langle \mathcal S, \mathcal A, T, R, \gamma \rangle$$

Variable           | Description
:----------------- | :--------------
$\mathcal{S}$      | State space
$\mathcal{A}$      | Action space
$T$                | Transition model
$R$                | Reward model
$\gamma \in [0,1]$ | Discount factor

We also define $TR$ as a shorthand to sample the transition model $T$ and reward model $R$.
"""

# ╔═╡ 8c5079e0-0f40-11eb-0c79-4ddc395519a3
struct MDP
	𝒮  # state space
	𝒜  # action space
	T  # transition function
	R  # reward function
	TR # sample transition and reward
	γ  # discount factor
end

# ╔═╡ 59285fa0-0f5f-11eb-323e-03e29ded12df
md"""
## Working Example: Gridworld MDP
"""

# ╔═╡ 1612cce0-0f60-11eb-1336-2939d0577246
# using TikzPictures

# ╔═╡ d402751e-0f60-11eb-02ea-a5a92a50c5ee
md"""
Simple MDP example from [Josh Greaves](https://joshgreaves.com/reinforcement-learning/introduction-to-reinforcement-learning/).
"""

# ╔═╡ 271e0d10-0f60-11eb-0e1a-a10dd335701f
# TikzPicture(L"""
# \node[minimum size=1cm, draw=black, fill=white, circle] (s) {\shortstack{Don't\\Understand}};
# \node[minimum size=1cm, draw=black, fill=white, circle, right=3.5cm of s] (s2) {Understand};
# \node[minimum size=1.3cm, draw=black, fill=white, diamond, above=0.5cm of s] (r) {$r_t$};
# \node[minimum size=1cm, draw=black, fill=white, rectangle, above=0.5cm of r] (a) {$a_t$};


# \draw[->] (s) -- (s2) node[midway, above] {Study $(0.8)$};
# \draw[->] [loop below, label={Study $(0.2)$}] (s) to (s);
# %\path[->] (s) -- [out=180, in=200, looseness=5, label={Study $(0.2)$}] (s);

# %\path[->]
# %    (s) edge [out=0] node {Study} (s2);
# %\draw[->] (s) -- (s2) node {Study};

# \draw[->] (s) -- (r);
# \draw[->] (a) -- (r);
# \draw[->] (a) [out=0,in=135] to (s2);
# """,
# preamble="""
# \\definecolor{pastelBlue}{HTML}{1BA1EA}
# \\usetikzlibrary{shapes}
# \\usetikzlibrary{positioning}
# \\usetikzlibrary{arrows}
# \\tikzset{every picture/.style={semithick, >=stealth'}}
# """,
# options="every node/.style={scale=1.25}")

# ╔═╡ 89d06640-0f62-11eb-1e65-192bc5b08a51
md"""
### States
$$\mathcal{S} = \{\text{don't understand},\; \text{understand}\}$$
"""

# ╔═╡ 5cc50e60-0f5f-11eb-0e06-2b69b7017802
@enum States dont_understand understand

# ╔═╡ 12013c40-0f60-11eb-31a6-0733ce4a66fb
𝒮 = (dont_understand, understand);

# ╔═╡ ff3b56f0-0f63-11eb-0689-1985d62a2f7a
md"""
### Actions
$$\mathcal{A} = \{\text{don't study},\; \text{study}\}$$
"""

# ╔═╡ 01287dbe-0f60-11eb-298a-93199a217f0f
@enum Actions dont_study study

# ╔═╡ 0873b592-0f60-11eb-1961-b955934386b5
𝒜 = (dont_study, study);

# ╔═╡ 1a07d800-0f64-11eb-3d54-3994bcf81271
md"""
### Rewards
We get a $-1$ reward for studying (it's tough...) and a $+1$ reward for not studying. We also get an additive $+10$ reward for understanding and a $-10$ reward for not understanding.
"""

# ╔═╡ 20f53860-0f64-11eb-15e6-cb9384937408
R(s,a) = (a == study ? -1 : 1) + (s == understand ? 10 : -10)

# ╔═╡ 430b91b0-0f64-11eb-1d73-89c89c9bb7bb
md"""
### Transitions
We understand a concept $80\%$ of the time when we study.
"""

# ╔═╡ 8cb37710-0f64-11eb-0d21-abe7e504ff22
function T(s,a)
	if s == dont_understand && a == dont_study
		# if we don't understand and don't study, we continue to not understand.
		return Categorical([1.0, 0.0])
	elseif s == dont_understand && a == study
		# if we don't understand and study, we have an 80% chance to understand.
		return Categorical([0.2, 0.8])
	else
		# otherwise, we understand already.
		return Categorical([0.0, 1.0])
	end
end

# ╔═╡ 2baa0b90-0f65-11eb-0526-0facc81c6edd
function TR(s,a)
	s′ = 𝒮[rand(T(s,a))]
	r = R(s′,a)
	return (s′, r)
end	

# ╔═╡ 1872c3a0-0f65-11eb-02d9-757cd36e2f65
γ = 0.95 # discount factor

# ╔═╡ 216ac110-0f65-11eb-0991-f9fc5d96c2cb
𝒫 = MDP(𝒮, 𝒜, T, R, TR, γ)

# ╔═╡ d8814fe0-0f47-11eb-1b8b-699d58ab9347
md"""
## Finite Difference

$$\frac{\partial f}{\partial x}(x) \approx \frac{f(x + \delta) - f(x)}{δ}$$
"""

# ╔═╡ c8d79b80-0f47-11eb-0c2b-793cbd7ea9dd
forward_diff(f, x; δ=sqrt(eps())) = (f(x + δ) - f(x)) / δ

# ╔═╡ 2cf4ba30-0f48-11eb-260d-413f902c1460
md"""
Example approximate derivative:

$\frac{\partial \log(x^2)}{\partial x} = \frac{2x}{x^2}$
"""

# ╔═╡ bbe3c2e2-0f48-11eb-37d1-21a6be0e06ed
f(x) = log(x^2);

# ╔═╡ ec43cb60-0f48-11eb-1ca9-918754d22908
f′(x) = 2x/x^2;

# ╔═╡ 4fc84900-0f48-11eb-323a-0755b22da27f
x = 3;

# ╔═╡ 03b94eb0-0f48-11eb-125e-c51ab377cd05
dx = forward_diff(f, x)

# ╔═╡ 86342b80-0f48-11eb-1588-87b1ddba4376
isapprox(dx, f′(x), atol=1e-5)

# ╔═╡ 10d006b2-0f49-11eb-2553-97020ab3d986
md"""
### Value Gradient Estimate

In the context of policy optimization, we want to estimate the gradient

$$\nabla U(𝛉) \approx \begin{bmatrix}
\displaystyle\frac{U(𝛉 + \delta \mathbf e^{(1)}) - U(𝛉)}{δ}, \,\ldots\, , \frac{U(𝛉 + \delta\mathbf e^{(n)}) - U(𝛉)}{\delta}
\end{bmatrix}$$

where $\mathbf e^{(i)}$ is the $i$th *standard basis* vector consisting of zeros except for the $i$th component that is $1$.
"""

# ╔═╡ 9f649350-0f49-11eb-1efc-c34b0435fa68
𝐞(i,n) = Int[j==i for j in 1:n];

# ╔═╡ 20ea9df0-0f4c-11eb-2a5b-7b44804d5834
md"""
Example vector-valued function $U(𝛉)$, taking the partial derivatives numerically

$$U(𝛉) = \theta_1^2 + \theta_2^2$$

where $U: 𝛉 \to \mathbb{R}$, a real-valued function of $n$ variables.
"""

# ╔═╡ cd98d040-0f4b-11eb-2122-c72e2aaa7c5a
U(𝛉) = 𝛉[1]^2 + 𝛉[2]^2;

# ╔═╡ d9defc60-0f4d-11eb-3d65-3d1a30081be6
U([1, 2])

# ╔═╡ 6b527e50-0f54-11eb-3485-0701ca2589af
𝛉 = [1, 2]

# ╔═╡ 529b85e0-0f4b-11eb-101d-290839143cc5
∇U(U,𝛉; n=length(𝛉),δ=sqrt(eps())) = [forward_diff(U, 𝛉; δ=δ*𝐞(i,n))[i] for i in 1:n];

# ╔═╡ acd3d94e-0f4f-11eb-11af-13f83340567d
Markdown.parse("""
\$\$\\begin{equation}
\\nabla U(𝛉) = \\begin{bmatrix}
2\\theta_1\\\\
2\\theta_2
	\\end{bmatrix} \\approx \\begin{bmatrix}
$(∇U(U, 𝛉)[1])\\\\
$(∇U(U, 𝛉)[2])
\\end{bmatrix}
\\end{equation}\$\$
""")

# ╔═╡ df2e9d30-0f4b-11eb-1171-f5e3a7d1eef3
∇U(U, 𝛉)

# ╔═╡ 560cc260-0f4c-11eb-2ee6-a58f60d96694
md"""
## Simulate Rollouts
We simulate policy $\pi$ rollouts to estimate the value function $U(𝛉)$.
"""

# ╔═╡ 59e14a00-0f4c-11eb-2d2e-9d4263911877
function simulate(𝒫::MDP, s, π, d)
	τ = []
	for i in 1:d
		a = π(s)
		s′, r = 𝒫.TR(s,a)
		push!(τ, (s,a,r))
		s = s′
	end
	return τ
end

# ╔═╡ 8f5016c0-0f6b-11eb-3f16-832e1dc52a59
md"""
---
"""

# ╔═╡ 9ccbd980-0f69-11eb-3f64-c3f04deff03d
md"""
Example random policy uniformly over actions *don't study* or *study*.
"""

# ╔═╡ b8bb3910-0f50-11eb-3a1f-95fe87fb974f
π(s) = rand(𝒜); # random policy

# ╔═╡ eb6b6690-0f65-11eb-028f-73699f3e4372
τ = simulate(𝒫, dont_understand, π, 10)

# ╔═╡ 928d4060-0f6b-11eb-35ba-e16d91e52bf8
md"""
---
"""

# ╔═╡ b3259720-0f69-11eb-05f2-010da263b15c
md"""
Example policy where we *study* with a $10\%$ chance.
"""

# ╔═╡ 57955800-0f69-11eb-253f-9b9a9956038b
π_90_10(s) = 𝒜[rand(Categorical([0.9, 0.1]))];

# ╔═╡ 95264cb0-0f69-11eb-02a9-4f1dd18a47fa
τ_90_10 = simulate(𝒫, dont_understand, π_90_10, 10)

# ╔═╡ 94af0ef0-0f6b-11eb-369b-a14fc2d1bc24
md"""
---
"""

# ╔═╡ c0788a40-0f69-11eb-1d93-4f85b2e7116d
md"""
Example policy where we always study if we don't understand.
"""

# ╔═╡ 1abf2f30-0f66-11eb-1b73-93bfdbdd540f
π_study(s) = s == dont_understand ? study : dont_study;

# ╔═╡ 4ec50e32-0f66-11eb-09d7-8bcfbc32b0b5
τ_study = simulate(𝒫, dont_understand, π_study, 10)

# ╔═╡ 4243cf82-0f51-11eb-3c56-4fc7dd540660
md"""
## Regression Gradient
Instead of using the finite difference to approximate the gradient at $𝛉$, we can use *linear regression* with random perturbations from $𝛉$.

The perturbations are stored in a matrix:

$$\Delta \mathbf{\Theta} = \left[\Delta 𝛉^{(1)},\, \ldots,\, \Delta 𝛉^{(m)}\right]$$

For each perturbation, we perform a rollout and estimate the change in utility:

$$\Delta \mathbf U = \left[ U\left(𝛉 + \Delta 𝛉^{(1)}\right) - U(𝛉),\, \ldots, \, U\left(𝛉 + \Delta 𝛉^{(m)}\right) - U(𝛉)\right]$$

Then the policy gradient estimate using linear regression is:

$$\nabla U(𝛉) \approx \Delta \mathbf\Theta^+ \Delta\mathbf U$$
"""

# ╔═╡ 6ed6adf0-0f52-11eb-3680-05adcb13cbda
struct RegressionGradient
	𝒫 # problem (MDP)
	b # initial state distribution
	d # depth
	m # number of samples
	δ # step size
end

# ╔═╡ 79439820-0f52-11eb-0efd-335db997a7ad
import LinearAlgebra: normalize

# ╔═╡ 63412240-0f52-11eb-0081-b7703f2feee7
function gradient(M::RegressionGradient, π, 𝛉)
	𝒫, b, d, m, δ, γ = M.𝒫 , M.b, M.d, M.m, M.δ, M.𝒫.γ
	Δ𝚯 = [δ.*normalize(randn(length(𝛉)), 2) for i = 1:m]
	R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
	U(𝛉) = R(simulate(𝒫, rand(b), s->π(𝛉,s), d))
	ΔU = [U(𝛉 + Δ𝛉) - U(𝛉) for Δ𝛉 in Δ𝚯]
	return pinv(reduce(hcat, Δ𝚯)') * ΔU
end

# ╔═╡ 78347dd0-0f6d-11eb-06dd-cf7168b0b81d
M = RegressionGradient(𝒫, 𝒫.𝒮, 10, 20, 0.1)

# ╔═╡ cf75a260-0f52-11eb-3bc9-3d30c93d8fba
md"""
## Likelihood Ratio

The *likelihood ratio trick* is:

$$\nabla_𝛉 \log p_𝛉(\tau) = \frac{\nabla_𝛉 p_𝛉(\tau)}{p_𝛉(\tau)}$$

Which can be used to evaluate the policy gradient:

$$\begin{align}
\nabla U(𝛉) &= \mathbb{E}_\tau \left[\nabla_𝛉 \log p_𝛉(\tau) R(\tau)\right] 
\end{align}$$

We can estimate the above expectation through rolling out simulated trajectories.
"""

# ╔═╡ 1bdc2700-0f5d-11eb-1152-4bb2534f0f46
struct LikelihoodRatioGradient
	𝒫 # problem (MDP)
	b # initial state distribution
	d # depth
	m # number of samples
	∇logπ # gradient of log likelihood
end

# ╔═╡ 53f1e8f0-0f5d-11eb-0f27-677b660be61e
import Statistics: mean

# ╔═╡ 27223320-0f5d-11eb-200e-9fa83fc1db57
function gradient(M::LikelihoodRatioGradient, π, 𝛉)
	𝒫, b, d, m, ∇logπ, γ = M.𝒫, M.b, M.d, M.m, M.∇logπ, M.𝒫.γ
	π𝛉(s) = π(𝛉, s)
	R(τ) = sum(r*γ^(k-1) for (k, (s,a,r)) in enumerate(τ))
	∇U(τ) = sum(∇logπ(𝛉, a, s) for (s,a) in τ)*R(τ)
	return mean(∇U(simulate(𝒫, rand(b), π𝛉, d)) for i in 1:m)
end

# ╔═╡ 077527c0-0f54-11eb-0bb9-417408e519b5
md"""
## Reward-to-Go

The *reward-to-go* approach attempts to reduce the variance in the estimate, as compared to the likeihood ration policy gradient method.

$$\nabla U(𝛉) = \mathbb{E}_\tau \left[ \sum_{k=1}^d \nabla_𝛉\log \pi_𝛉\left(s^{(k)} \mid s^{(k)}\right) \gamma^{k-1} r_\text{to-go}^{(k)} \right]$$

with the reward-to-go from step $k$ defined as:

$$r_\text{to-go}^{(k)} = \sum_{\ell=k}^dr^{(\ell)}\gamma^{\ell-k}$$
"""

# ╔═╡ 696a6e50-0f5d-11eb-0c53-87841669a060
struct RewardToGoGradient
	𝒫 # problem
	b # initial state distribution
	d # depth
	m # number of samples
	∇logπ # gradient of log likelihood
end

# ╔═╡ 6ce76010-0f5d-11eb-39d8-37881e2d6650
function gradient(M::RewardToGoGradient, π, 𝛉)
	𝒫, b, d, m, ∇logπ, γ = M.𝒫, M.b, M.d, M.m, M.∇logπ, M.𝒫.γ
	π𝛉(s) = π(𝛉, s)
	R(τ, j) = sum(r*γ^(k-1) for (k,(s,a,r)) in zip(j:d, τ[j:end]))
	∇U(τ) = sum(∇logπ(𝛉, a, s)*R(τ,j) for (j, (s,a,r)) in enumerate(τ))
	return mean(∇U(simulate(𝒫, rand(b), π𝛉, d)) for i in 1:m)
end

# ╔═╡ 91d2e650-0f45-11eb-2baf-998fe3df3270
md"""
## Advantage

$$A(s,a) = Q(s,a) - U(s)$$

The policy gradient using the advantage is unbiases and typically has much lower variance. The gradient compuation has the form:

$$\nabla U(𝛉) = \mathbb{E}_\tau\left[ \sum_{k=1}^d \nabla_𝛉 \log \pi_𝛉\left(a^{(k)} \mid s^{(k)}\right) \gamma^{k-1} A_𝛉\left(s^{(k)}, a^{(k)}\right) \right]$$
"""

# ╔═╡ Cell order:
# ╟─96f42b80-0f40-11eb-1fde-67a6c8b9aad4
# ╠═b039225e-0f65-11eb-0925-337baf99a1f2
# ╟─a2114652-0f55-11eb-3324-81f9b428185a
# ╠═8c5079e0-0f40-11eb-0c79-4ddc395519a3
# ╟─59285fa0-0f5f-11eb-323e-03e29ded12df
# ╠═1612cce0-0f60-11eb-1336-2939d0577246
# ╟─d402751e-0f60-11eb-02ea-a5a92a50c5ee
# ╟─271e0d10-0f60-11eb-0e1a-a10dd335701f
# ╟─89d06640-0f62-11eb-1e65-192bc5b08a51
# ╠═5cc50e60-0f5f-11eb-0e06-2b69b7017802
# ╠═12013c40-0f60-11eb-31a6-0733ce4a66fb
# ╟─ff3b56f0-0f63-11eb-0689-1985d62a2f7a
# ╠═01287dbe-0f60-11eb-298a-93199a217f0f
# ╠═0873b592-0f60-11eb-1961-b955934386b5
# ╟─1a07d800-0f64-11eb-3d54-3994bcf81271
# ╠═20f53860-0f64-11eb-15e6-cb9384937408
# ╟─430b91b0-0f64-11eb-1d73-89c89c9bb7bb
# ╠═7e7a7440-0f65-11eb-14e5-e1281ae10f18
# ╠═8cb37710-0f64-11eb-0d21-abe7e504ff22
# ╠═2baa0b90-0f65-11eb-0526-0facc81c6edd
# ╠═1872c3a0-0f65-11eb-02d9-757cd36e2f65
# ╠═216ac110-0f65-11eb-0991-f9fc5d96c2cb
# ╟─d8814fe0-0f47-11eb-1b8b-699d58ab9347
# ╠═c8d79b80-0f47-11eb-0c2b-793cbd7ea9dd
# ╟─2cf4ba30-0f48-11eb-260d-413f902c1460
# ╠═bbe3c2e2-0f48-11eb-37d1-21a6be0e06ed
# ╠═ec43cb60-0f48-11eb-1ca9-918754d22908
# ╠═4fc84900-0f48-11eb-323a-0755b22da27f
# ╠═03b94eb0-0f48-11eb-125e-c51ab377cd05
# ╠═86342b80-0f48-11eb-1588-87b1ddba4376
# ╟─10d006b2-0f49-11eb-2553-97020ab3d986
# ╠═9f649350-0f49-11eb-1efc-c34b0435fa68
# ╠═529b85e0-0f4b-11eb-101d-290839143cc5
# ╟─20ea9df0-0f4c-11eb-2a5b-7b44804d5834
# ╠═cd98d040-0f4b-11eb-2122-c72e2aaa7c5a
# ╠═d9defc60-0f4d-11eb-3d65-3d1a30081be6
# ╠═6b527e50-0f54-11eb-3485-0701ca2589af
# ╟─acd3d94e-0f4f-11eb-11af-13f83340567d
# ╠═df2e9d30-0f4b-11eb-1171-f5e3a7d1eef3
# ╟─560cc260-0f4c-11eb-2ee6-a58f60d96694
# ╠═59e14a00-0f4c-11eb-2d2e-9d4263911877
# ╟─8f5016c0-0f6b-11eb-3f16-832e1dc52a59
# ╟─9ccbd980-0f69-11eb-3f64-c3f04deff03d
# ╠═b8bb3910-0f50-11eb-3a1f-95fe87fb974f
# ╠═eb6b6690-0f65-11eb-028f-73699f3e4372
# ╟─928d4060-0f6b-11eb-35ba-e16d91e52bf8
# ╟─b3259720-0f69-11eb-05f2-010da263b15c
# ╠═57955800-0f69-11eb-253f-9b9a9956038b
# ╠═95264cb0-0f69-11eb-02a9-4f1dd18a47fa
# ╟─94af0ef0-0f6b-11eb-369b-a14fc2d1bc24
# ╟─c0788a40-0f69-11eb-1d93-4f85b2e7116d
# ╠═1abf2f30-0f66-11eb-1b73-93bfdbdd540f
# ╠═4ec50e32-0f66-11eb-09d7-8bcfbc32b0b5
# ╟─4243cf82-0f51-11eb-3c56-4fc7dd540660
# ╠═6ed6adf0-0f52-11eb-3680-05adcb13cbda
# ╠═79439820-0f52-11eb-0efd-335db997a7ad
# ╠═63412240-0f52-11eb-0081-b7703f2feee7
# ╠═78347dd0-0f6d-11eb-06dd-cf7168b0b81d
# ╟─cf75a260-0f52-11eb-3bc9-3d30c93d8fba
# ╠═1bdc2700-0f5d-11eb-1152-4bb2534f0f46
# ╠═53f1e8f0-0f5d-11eb-0f27-677b660be61e
# ╠═27223320-0f5d-11eb-200e-9fa83fc1db57
# ╟─077527c0-0f54-11eb-0bb9-417408e519b5
# ╠═696a6e50-0f5d-11eb-0c53-87841669a060
# ╠═6ce76010-0f5d-11eb-39d8-37881e2d6650
# ╟─91d2e650-0f45-11eb-2baf-998fe3df3270
