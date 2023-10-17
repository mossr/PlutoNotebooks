### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ c7e947eb-a51d-4456-a636-f61b971724c7
using DiscreteValueIteration

# ╔═╡ 99914226-eb68-4cf9-a688-2b8618587461
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add([
        Pkg.PackageSpec(name="POMDPs"),
        Pkg.PackageSpec(name="QuickPOMDPs"),
        Pkg.PackageSpec(name="POMDPTools", version="0.1.3"),
        Pkg.PackageSpec(name="DiscreteValueIteration"),
        Pkg.PackageSpec(name="PlutoUI"),
        Pkg.PackageSpec(name="Plots"),
        Pkg.PackageSpec(name="ColorSchemes"),
        Pkg.PackageSpec(name="Colors"),
    ])
	using POMDPs, POMDPTools, QuickPOMDPs
	using Plots, ColorSchemes, Colors
	default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style plots

	struct State # State definition
		x::Int
		y::Int
	end

	@enum Action UP DOWN LEFT RIGHT # Action definition

	null_state = State(-1,-1)
	𝒮 = [[State(x,y) for x=1:10, y=1:10]..., null_state] # State−space
	𝒜 = [UP, DOWN, LEFT, RIGHT] # Action−space

	apply(s,a) = Dict(
		UP    => State(s.x, s.y+1),
		DOWN  => State(s.x, s.y-1),
		LEFT  => State(s.x-1, s.y),
		RIGHT => State(s.x+1, s.y))[a]


	function T(s, a) # Transition function
		R(s) != 0 && return Deterministic(null_state)
		Nₐ = length(𝒜)
		next_states = Vector{State}(undef, Nₐ + 1)
		probabilities = zeros(Nₐ + 1)
		for (i, a′) in enumerate(𝒜)
			prob = (a′ == a) ? 0.7 : (1 - 0.7) / (Nₐ - 1)
			destination = apply(s, a′)
			next_states[i+1] = destination
			if 1 ≤ destination.x ≤ 10 && 1 ≤ destination.y ≤ 10
				probabilities[i+1] += prob
			end
		end
		next_states[1] = s
		probabilities[1] = 1 - sum(probabilities)
		return SparseCat(next_states, probabilities)
	end

	function R(s, a=missing) # Reward function
		if s == State(4,3)
			return -10
		elseif s == State(4,6)
			return -5
		elseif s == State(9,3)
			return 10
		elseif s == State(8,8)
			return 3
		end
		return 0
	end

	function render(mdp, s=nothing;
			values=nothing, show_grid=true, label_rewards=true,
			cmap=ColorScheme([RGB(0.7, 0, 0), RGB(1, 1, 1), RGB(0, 0.4, 0)])) 
		gr()
		s_valid = setdiff(states(mdp), [null_state])
		U = map(s->isnothing(values) ? reward(mdp,s) : value(values,s), s_valid)
		xmax = ymax = Int(sqrt(length(U)))
		Uxy = reshape(U, xmax, ymax)
		heatmap(Uxy',leg=:none,ratio=:equal,frame=:box,tickor=:out,c=cmap.colors)
		xlims!(0.5, xmax+0.5)
		ylims!(0.5, ymax+0.5)
		xticks!(1:xmax)
		yticks!(1:ymax)
		rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
		terminal_states = filter(s->reward(mdp, s) != 0, states(mdp))
		if label_rewards
			for sT in terminal_states
				r = reward(mdp, sT)
				annotate!([(sT.x, sT.y, (r, :white, :c, 10, "Computer Modern"))])
			end
		end
		if show_grid
			for x in 1:xmax, y in 1:ymax
				plot!(rectangle(1, 1, x-0.5, y-0.5), fillalpha=0, linecolor=:gray)
			end
		end
		if !isnothing(s)
			color = (s in terminal_states) ? "yellow" : "blue"
			scatter!([s.x], [s.y], ms=8, c=color, alpha=0.9)
		end
		return title!("Grid World")
	end

	abstract type GridWorld <: MDP{State, Action} end
	
	md"> `Grid World MDP definition (unhide).`"
end

# ╔═╡ 39981cad-3875-4438-8af5-aaee151c6a0a
using PlutoUI

# ╔═╡ fa74acb0-1c29-11ec-0f1b-fd1757a25278
md"""
# Value Iteration
The _value iteration_ algorithm uses _dynamic programming_ and the _Bellman equation_ to solve for the optimal policy in discrete MDPs (i.e., discrete states and actions).$^1$
"""

# ╔═╡ 9ef949e0-caf6-47e5-88ca-ea2d75f79443
md"""
## Grid World MDP
"""

# ╔═╡ 863b9fc0-3711-4729-b730-ae184ebba0b5
mdp = QuickMDP(GridWorld,
	states     = 𝒮,
	actions    = 𝒜,
	reward     = R,
	transition = T,
	discount   = 0.95,
	isterminal = s->s==State(-1,-1),
	render     = render)

# ╔═╡ 9654f2df-5aef-4ccc-8192-10b6f5c5f829
function Tuple(mdp::MDP)
	𝒮 = states(mdp)
	𝒜 = actions(mdp)
	γ = discount(mdp)
	R(s,a) = reward(mdp, s, a)
	T(s,a,s′) = pdf(transition(mdp, s, a), s′)
	return (𝒮, 𝒜, R, T, γ)
end

# ╔═╡ bcaa3dba-db7d-41eb-9886-84b3f1449849
md"""
## Value Iteration
The _value iteration_ algorithm$^2$ iterative updates the utilities for each state using the action $a \in \mathcal{A}$ that maximizes the _Bellman equation_ $R(s,a) + \gamma\sum_{s^\prime}T(s^\prime \mid s, a)U(s^\prime)$ using _dynamic programming_ to recursively call $U(s^\prime)$ for all states $s^\prime$.
"""

# ╔═╡ 423e14a1-aaba-4e02-9e47-97f2d87d4f42
md"""
>  $\textbf{function } \text{ValueIteration}(\mathcal{M})$\
> $\qquad k\leftarrow 0$\
> $\qquad U_0(s) \leftarrow \text{ for all states } s$\
> $\qquad \textbf{repeat}$\
> $\qquad\qquad U_{k+1}(s) \leftarrow \displaystyle\max_{a \in \mathcal{A}} \left( R(s,a) + \gamma\sum_{s^\prime \in \mathcal{S}} T(s^\prime \mid s, a)U_k(s^\prime)\right) \text{ for all states } s$\
> $\qquad\qquad k \leftarrow k + 1$\
> $\qquad \textbf{until }\text{convergence}$\
> $\qquad \textbf{return } U_k$
"""

# ╔═╡ 44388665-1927-45fb-9325-256b618749f9
function value_iteration(mdp; k_max=30)
	(𝒮, 𝒜, R, T, γ) = Tuple(mdp)
	U = Dict(map(s->s=>0.0, 𝒮))

	for k in 1:k_max
		U = Dict(map(s->s=>maximum(a->R(s,a) + γ*sum(s′->T(s,a,s′)*U[s′], 𝒮), 𝒜), 𝒮))
	end

	return U
end

# ╔═╡ 1f9eee07-aa97-44b1-b619-b0d92fba6444
md"""
>  $\textbf{function } \text{ExtractPolicy}(\mathcal{M}, U)$\
> $\qquad \pi(s) = \displaystyle\operatorname*{argmax}_{a \in \mathcal{A}}\left(R(s,a) + \gamma\sum_{s^\prime \in \mathcal{S}}T(s^\prime \mid s, a)U_k(s^\prime)\right)\text{ for all states } s$\
> $\qquad \textbf{return } \pi$
"""

# ╔═╡ 5b9890d0-3f8c-4a89-9ca2-37f6802f99a7
function extract_policy(mdp, U)
	(𝒮, 𝒜, R, T, γ) = Tuple(mdp)
	π = Dict(map(s->s=>argmax(a->R(s,a) + γ*sum(s′->T(s,a,s′)*U[s′], 𝒮), 𝒜), 𝒮))
	return π
end

# ╔═╡ 2b8715ec-15b8-4453-8720-df2e225e2d75
U = value_iteration(mdp)

# ╔═╡ ceb07b8c-68d9-44a9-afc0-c124c8dd966b
md"""
### Querying policy $\pi$ given state $s$
"""

# ╔═╡ 34c6a700-e04b-4cac-96b7-ffcd2957fc3e
s = State(5,5);

# ╔═╡ c799a2bc-f4b1-4512-98c8-0cf3d3f16772
md"""
### Visualize value function $U(s)$ for all states $s$
"""

# ╔═╡ 6ae30851-49cc-4c8f-a1d9-262758882f01
render(mdp; values=U)

# ╔═╡ 7f1c6a83-9bd9-4e20-86da-7ad8e74e68ad
md"""
## Asynchronous Value Iteration
_Gauss-Seidel value iteration_ or _asyncronous value iteration_ will update $U(s)$ for  each state, where the states are in some arbitrary order (instead of updating $U$ _once all states have been iterated over_).

$$U(s) = \max_{a \in \mathcal{A}}\left(R(s,a) + \gamma\sum_{s^\prime \in \mathcal{S}} T(s^\prime \mid s, a)U(s^\prime) \right)$$
"""

# ╔═╡ f06daa0b-d5ab-440f-a22b-12b0e018805d
function async_value_iteration(mdp; k_max=30)
	(𝒮, 𝒜, R, T, γ) = Tuple(mdp)
	U = Dict(map(s->s => 0.0, 𝒮))

	for k in 1:k_max
		for s ∈ 𝒮
			U[s] = maximum(a->R(s,a) + γ*sum(s′->T(s,a,s′)*U[s′], 𝒮), 𝒜)
		end
	end

	π = Dict(map(s->s => argmax(a->R(s,a) + γ*sum(s′->T(s,a,s′)*U[s′], 𝒮), 𝒜), 𝒮))
	return U, π
end

# ╔═╡ 12f3d708-c9f0-4acf-9e0b-b933649de50c
Uₐ, πₐ = async_value_iteration(mdp)

# ╔═╡ b694255e-0958-42d9-b883-eaeea0cc047f
md"""
## Compare Policies
"""

# ╔═╡ ab270fdc-fe8b-416e-91f8-f05bfe7e1fb9
solver = ValueIterationSolver();

# ╔═╡ 94bd2d6a-480a-42a9-b2b0-665ca7d232a7
policy = solve(solver, mdp);

# ╔═╡ 4ce68975-f936-4b9d-8c6d-3447474093ff
md"""
## Other Implementations
"""

# ╔═╡ f5649e96-c6c3-4a83-8065-47e66314500c
function value_iteration_full(mdp; k_max=30)
	𝒮 = states(mdp)
	𝒜 = actions(mdp)
	γ = discount(mdp)
	R(s,a) = reward(mdp, s, a)
	T(s,a,s′) = pdf(transition(mdp, s, a), s′)
	U = Dict(map(s->s=>0.0, 𝒮)) # U(s) ← 0 for all states s
	π = Dict()

	for k in 1:k_max
		for s ∈ 𝒮
			bellman = [R(s,a) + γ*sum(s′->T(s,a,s′)*U[s′], 𝒮) for a ∈ 𝒜]
			U[s] = maximum(bellman)
			π[s] = Action(argmax(bellman)-1)
		end
	end

	return U, π
end

# ╔═╡ b575322c-5a98-4f0b-ba20-c7247b44202e
function extract_policy(U)
	𝒮 = states(mdp)
	𝒜 = actions(mdp)
	γ = discount(mdp)
	R(s,a) = reward(mdp, s, a)
	T(s,a,s′) = pdf(transition(mdp, s, a), s′)
	
	π = Dict(map(s->s=>argmax(a->R(s,a) + γ*sum(s′->T(s,a,s′)*U[s′], 𝒮), 𝒜), 𝒮))
	return π
end

# ╔═╡ bf6e0ebc-fc1d-404b-815b-a7d2bf70d3b1
π = extract_policy(mdp, U)

# ╔═╡ 13947b73-27a7-4108-8ca0-0093e7eaea07
π[s]

# ╔═╡ a64da3a8-b1b1-4608-ab32-399f25b923a5
with_terminal() do
	@info all(s->policy(s)==π[s], 𝒮)
	@info all(s->policy(s)==πₐ[s], 𝒮) # async (Gauss-Seidel)
	@info all(s->π[s]==πₐ[s], 𝒮) # VI vs. async (Gauss-Seidel)
end

# ╔═╡ f74166e9-02d6-40da-b95e-9216c33b6ad1
function value_iteration_bellman(mdp; k_max=30)
	(𝒮, 𝒜, R, T, γ) = Tuple(mdp)
	U = Dict(map(s->s=>0.0, 𝒮))
	π = Dict()
	bellman(U,s,a) = R(s,a) + γ*sum(s′->T(s,a,s′)*U[s′], 𝒮)
	
	for k in 1:k_max		
		U = Dict(map(s->s=>maximum(a->bellman(U,s,a), 𝒜), 𝒮))
		π = Dict(map(s->s=>argmax(a->bellman(U,s,a), 𝒜), 𝒮))
	end

	return U, π
end

# ╔═╡ 0300c3e2-e77d-41ca-aece-8a7e9ab03459
md"""
$$U^*(s) = \max_{a \in \mathcal{A}}\left(R(s,a) + \gamma\sum_{s^\prime \in \mathcal{S}} T(s^\prime \mid s, a)U^*(s^\prime) \right)$$
"""

# ╔═╡ e31700b7-eb2c-49f7-8683-6a30552ca712
md"""
$$\pi^*(s) = \operatorname*{argmax}_{a \in \mathcal{A}}\left(R(s,a) + \gamma\sum_{s^\prime \in \mathcal{S}} T(s^\prime \mid s, a)U^*(s^\prime) \right)$$
"""

# ╔═╡ 34d03da4-f95c-4dd2-8364-0b4b0307f875
md"""
## Helper functions
"""

# ╔═╡ b31b2b75-6bf4-4b44-839b-9043f85f0472
POMDPs.action(policy::Dict, s) = policy[s]

# ╔═╡ 76c59acf-8ad2-4c19-b34a-5917be3b75a5
POMDPs.value(U::Dict, s) = U[s]

# ╔═╡ c0db85c4-bd43-4773-8c7b-0e485c21dd55
(π::Policy)(s) = action(π, s) # π(s) sugar

# ╔═╡ ad92ca29-7738-49bc-9c64-1570b7b97846
md"""
## References
1. Richard Bellman, _A Markovian Decision Process_, Indiana University Mathematics Journal, 1957.
2. Mykel J. Kochenderfer, _Decision Making Under Uncertainty: Theory and Application_, MIT Press, 2015.
"""

# ╔═╡ 715a9553-f3a4-43dd-b93d-021eaff71067
TableOfContents(title="Value Iteration Algorithm")

# ╔═╡ Cell order:
# ╟─fa74acb0-1c29-11ec-0f1b-fd1757a25278
# ╟─9ef949e0-caf6-47e5-88ca-ea2d75f79443
# ╟─99914226-eb68-4cf9-a688-2b8618587461
# ╠═863b9fc0-3711-4729-b730-ae184ebba0b5
# ╠═9654f2df-5aef-4ccc-8192-10b6f5c5f829
# ╟─bcaa3dba-db7d-41eb-9886-84b3f1449849
# ╟─423e14a1-aaba-4e02-9e47-97f2d87d4f42
# ╠═44388665-1927-45fb-9325-256b618749f9
# ╟─1f9eee07-aa97-44b1-b619-b0d92fba6444
# ╠═5b9890d0-3f8c-4a89-9ca2-37f6802f99a7
# ╠═2b8715ec-15b8-4453-8720-df2e225e2d75
# ╠═bf6e0ebc-fc1d-404b-815b-a7d2bf70d3b1
# ╟─ceb07b8c-68d9-44a9-afc0-c124c8dd966b
# ╠═34c6a700-e04b-4cac-96b7-ffcd2957fc3e
# ╠═13947b73-27a7-4108-8ca0-0093e7eaea07
# ╟─c799a2bc-f4b1-4512-98c8-0cf3d3f16772
# ╠═6ae30851-49cc-4c8f-a1d9-262758882f01
# ╟─7f1c6a83-9bd9-4e20-86da-7ad8e74e68ad
# ╠═f06daa0b-d5ab-440f-a22b-12b0e018805d
# ╠═12f3d708-c9f0-4acf-9e0b-b933649de50c
# ╟─b694255e-0958-42d9-b883-eaeea0cc047f
# ╠═c7e947eb-a51d-4456-a636-f61b971724c7
# ╠═ab270fdc-fe8b-416e-91f8-f05bfe7e1fb9
# ╠═94bd2d6a-480a-42a9-b2b0-665ca7d232a7
# ╠═39981cad-3875-4438-8af5-aaee151c6a0a
# ╠═a64da3a8-b1b1-4608-ab32-399f25b923a5
# ╟─4ce68975-f936-4b9d-8c6d-3447474093ff
# ╟─f5649e96-c6c3-4a83-8065-47e66314500c
# ╟─b575322c-5a98-4f0b-ba20-c7247b44202e
# ╟─f74166e9-02d6-40da-b95e-9216c33b6ad1
# ╟─0300c3e2-e77d-41ca-aece-8a7e9ab03459
# ╟─e31700b7-eb2c-49f7-8683-6a30552ca712
# ╟─34d03da4-f95c-4dd2-8364-0b4b0307f875
# ╠═b31b2b75-6bf4-4b44-839b-9043f85f0472
# ╠═76c59acf-8ad2-4c19-b34a-5917be3b75a5
# ╠═c0db85c4-bd43-4773-8c7b-0e485c21dd55
# ╟─ad92ca29-7738-49bc-9c64-1570b7b97846
# ╠═715a9553-f3a4-43dd-b93d-021eaff71067
