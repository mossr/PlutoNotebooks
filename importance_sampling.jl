### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 8f20ec09-2a13-4130-989d-6edd87deb89a
using Distributions

# ╔═╡ 4ec759e2-6a6e-493e-9e7d-4b9eeba0ebdd
using Plots; default(fontfamily="Computer Modern", framestyle=:box)

# ╔═╡ b9d672c1-f71e-4813-9214-84ed490f4914
using PlutoUI

# ╔═╡ 46bcc8a0-2117-4b85-8a44-fcaa0bba6868
using ColorSchemes

# ╔═╡ 4147c74d-10bc-48aa-a725-fea135d2e304
using PDMats, Primes, Random, LinearAlgebra, StatsBase

# ╔═╡ c85db890-8bcb-11ec-3a45-2b0d44ba0abc
md"""
# Importance sampling
$$\begin{align}
\mathbb{E}_p[f(x)] &= \int p(x)f(x)\; dx\\
                 &= \int p(x)\frac{q(x)}{q(x)}f(x)\; dx\\
                 &= \int q(x)\frac{p(x)}{q(x)}f(x)\; dx\\
                 &= \mathbb{E}_q\left[\frac{p(x)}{q(x)}f(x)\right]\\
                 &\approx \frac{1}{n}\sum_{i=1}^n \frac{p(x_i)}{q(x_i)}f(x_i)
\end{align}$$

 $p(x)$ is the _target distribution_, $q(x)$ is the _importance/proposal distribution_ and $\frac{p(x)}{q(x)}$ is the _likelihood ratio_.
"""

# ╔═╡ a3f021c4-03be-49d9-b27e-f050caedfcd3
p = Normal(8,1)

# ╔═╡ 42be2255-9bb8-4740-958f-9bfe9752f27b
q = Normal(12,3)

# ╔═╡ 3fa3ba32-ee4b-4c55-a0d5-2ea9a848d76f
w(xᵢ) = pdf(p,xᵢ) / pdf(q,xᵢ) # likelihood ratio

# ╔═╡ ef39b5ca-715b-4d22-aa89-513d16c31ff2
md"""
# Rare event probability: $\mathbb{E}_q\big[\mathbb{I}\{x \ge \gamma\}\big]$
"""

# ╔═╡ 6cfada66-a72c-4eb4-9aa9-408a6d16c0ca
md"""
#### Truth
"""

# ╔═╡ 34ffdb0b-eb92-4de2-a0e1-2a38c2b15373
# @bind γ Slider(8:15, default=13, show_value=true) # rare event threshold

# ╔═╡ fbaba7a8-ca9f-445f-baf5-2c02b5029603
γ = 10.2

# ╔═╡ 47ae008a-bd0a-45d5-b09d-2b18621994a2
𝕀(b) = b ? 1 : 0 # indicator function

# ╔═╡ fe2a153b-e90f-4826-8058-cfb9a6c2b41b
g(x) = 𝕀(x ≥ γ) # probability that value is above some threshold γ

# ╔═╡ 10f24b89-0c77-4e6c-b10c-14879582c325
md"""
#### Estimate
"""

# ╔═╡ 6fc930f6-7dd0-4aa0-8ce7-30a5582012e0
n = 1000 # notice low number of samples

# ╔═╡ 32bbfa28-1a47-4b1b-b2c5-bcc0d14d2f6e
X = rand(q, n); # sample from proposal distribution q

# ╔═╡ 873ef3ba-c803-46a6-b864-8b88903de7ec
pmin, pmax = minimum(pdf.(p, X)), maximum(pdf.(p, X));

# ╔═╡ 65668688-a98b-4afa-9d06-68c03ece67b0
gradient = cgrad(:viridis, [pmin, pmax])

# ╔═╡ 4212859a-8859-4b0a-9223-ccb1afc8cfcd
cmap = map(ℓ->get(gradient, ℓ), w.(X));

# ╔═╡ 70a2a4bb-f4e6-44cb-845e-a8b33999b514
begin
	xmin2, xmax2 = 0, 25
	plot(x->pdf(p,x), xmin2, xmax2, c=:black, lw=2, label="p (target/truth)")
	scatter!(X, pdf.(q, X), ms=80pdf.(q, X), msw=0, label=false, c=cmap)
	plot!(x->pdf(q,x), xmin2, xmax2, c=:crimson, lw=2, label="q (proposal)")
	plot!([γ, γ], [0, pdf(p,γ)], c=:crimson, lw=3, label=false)
	scatter!([γ], [pdf(p,γ)], c=:crimson, lw=3, label="𝔼[𝕀(x ≥ γ)]")
	# scatter!(X, pdf.(q, X), c=:white, ms=w.(X), label=false)
	# scatter!(X, pdf.(q, X), c=:white, ms=20pdf.(p, X), label=false)
	xlims!(xmin2, xmax2)
	xlabel!("cost")
	ylabel!("likelihood")
end

# ╔═╡ 14dd7ac0-4367-4bfc-9717-ccdb10bda171
md"""
###### Importance sampling estimate
"""

# ╔═╡ 45504a50-9ce2-4457-9493-26127f9f537e
μ = 1/n * sum(xᵢ->w(xᵢ)*g(xᵢ), X) # estimated mean

# ╔═╡ 363154fb-f4e7-4a4a-a878-5e458b738cc0
σ² = 1/n * sum(xᵢ->(w(xᵢ)*g(xᵢ) - μ)^2, X) # estimated variance

# ╔═╡ ccdf4496-bee6-4701-940b-e4560149fabe
md"""
## Monte Carlo
"""

# ╔═╡ 2c2ad947-298a-431b-b9e5-c74a25b0439c
mc = g.(rand(p, 10_000_000)) # notice high number of samples!

# ╔═╡ e9e0ad28-8d67-4a1d-9798-808b0632e87c
sum(mc)

# ╔═╡ e2d1d150-40f5-41df-9bcd-131f8fb14537
mean(mc), var(mc)

# ╔═╡ 54c65259-7e7d-4bd2-9983-82a2e957d808
md"""
# Another function $f(x)$
"""

# ╔═╡ f796718c-bcdc-4d5a-be3f-fe882c2679f2
md"""
Input value $x$ has likelihood $p(x)$ and output value of $f(x)$.
"""

# ╔═╡ add76ee3-2e70-4a55-84c1-2cb20bc5e30c
f(x) = 3*pdf(Exponential(2.5), x) # 1 / (1 + exp(-x + 1.9))

# ╔═╡ 8c112c36-00c8-42d3-bb10-5966fbef81d9
# samples = filter(y->y ≥ γ, rand(p, 10_000_000))

# ╔═╡ a4876a1a-bbf1-4b7a-83e0-58c4af899c36
md"""
#### Truth
"""

# ╔═╡ 7667881d-5546-452d-a630-1c972fdec485
f(mean(p)) # true (unknown)

# ╔═╡ fff3e2e1-bee5-4a2e-9570-85f8dfac82ca
k = 5000

# ╔═╡ b26c338c-4274-42d5-b1af-b5558a584a58
inputs = rand(p, k); # hard to get

# ╔═╡ 9604d9b2-9aa2-4131-8f42-74ceb74cf0e9
samples = f.(inputs); # thus, hard to get

# ╔═╡ 2fb742a3-6d96-4332-a9aa-62db643f0811
begin
	xmin, xmax = 0, 25
	plot(x->pdf(p,x), xmin, xmax, c=:blue, ls=:dash, label="p (target/truth)")
	plot!(x->pdf(q,x), xmin, xmax, c=:red, ls=:dash, label="q (proposal)")
	plot!(f, xmin, xmax, c=:green, lw=3, label="f (e.g., risk values)")
	# vline!([γ], c=:black, lw=3, label="γ")
	# plot!([mean(p), mean(p)], [0, f(mean(p))], c=:black, lw=3, label="𝔼[f(x)]")
	# scatter!([mean(p)], [f(mean(p))], c=:black, lw=3, label=false)
	scatter!(inputs, samples, c=:gray, ms=10pdf.(p,inputs), label=false)
	scatter!(X, f.(X), c=:white, ms=20pdf.(q,X), label=false)
	xlims!(xmin, xmax)
	xlabel!("cost")
	ylabel!("likelihood")
end

# ╔═╡ 7cf56c73-5022-45c3-b995-974d122eea16
mean(samples), var(samples) # hard to get

# ╔═╡ 5c4ab4a0-caab-4670-9edd-c96bb361ffa4
md"""
###### Importance sampling estimate
"""

# ╔═╡ fc283c41-11ee-4d7f-b2d6-311dff8777bd
μ̃ = 1/n * sum(xᵢ->w(xᵢ)*f(xᵢ), X)

# ╔═╡ f5ee8b07-0edf-4fd3-8584-cbce68e8a06a
σ̃² = 1/n * sum(xᵢ->(w(xᵢ)*f(xᵢ) - μ̃)^2, X)

# ╔═╡ 8df63154-bfdb-46ab-8d71-9a67dbf0e453
md"""
---
"""

# ╔═╡ 841ec98d-7e80-4cfc-a92e-8cd700235390
w.(X)'f.(X) / n # alternate

# ╔═╡ 96349c9f-1421-4d4c-ae28-3f076da83dc6
mean(xᵢ->w(xᵢ)*f(xᵢ), X), var(map(xᵢ->w(xᵢ)*f(xᵢ), X)) # alternate

# ╔═╡ d8af9650-927e-4069-a8ce-7d1428aa8653
md"""
# Multi-variate case
"""

# ╔═╡ b895f340-a407-4637-9a4a-0be5f481ce36
𝐩 = MvNormal([0,0], [1 0; 0 1]);

# ╔═╡ 44bf340a-8f11-4a1d-ac90-9692e4adec4d
𝐪 = MvNormal([4,4], [3 0; 0 3]);

# ╔═╡ 1d332efa-30c3-4523-9c00-d7c9d9e32f63
@bind γ₁ Slider(-5:0.5:5, default=3, show_value=true) # rare event threshold

# ╔═╡ aeaff93a-84bf-4b47-90f7-fa16b9ed708c
@bind γ₂ Slider(-5:0.5:5, default=3, show_value=true) # rare event threshold

# ╔═╡ 5c7352f8-a0d5-4b58-bda9-7c3932601489
begin
	xrange = range(-10, stop=10, length=100)
	yrange = range(-10, stop=10, length=100)

	rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
	mvkwargs = (ratio=1, c=:viridis, levels=30)
	contour(xrange, yrange, (x,y)->pdf(𝐩, [x,y]); mvkwargs...)
	contour!(xrange, yrange, (x,y)->pdf(𝐪, [x,y]); mvkwargs...)
	scatter!([γ₁], [γ₂], c=:red, ms=3, label=false)
	plot!([γ₁, xrange[end]], [γ₂, γ₂], c=:red, ms=3, label=false)
	plot!([γ₁, γ₁], [γ₂, yrange[end]], c=:red, ms=3, label=false)
	scatter!([γ₁], [γ₂], c=:red, ms=3, label=false)
	plot!(rectangle(xrange[end]-γ₁,yrange[end]-γ₂,γ₁,γ₂), c=:red, opacity=0.25, label=false)

	xlims!(xrange[1], xrange[end])
	ylims!(yrange[1], yrange[end])
end

# ╔═╡ 9facd395-91cd-46f9-86a8-7c7375a9c4a7
𝛄 = [γ₁,γ₂]

# ╔═╡ 7161fab4-834e-45ec-96c3-cafe2742f96d
md"""
#### Truth
"""

# ╔═╡ 736d48ec-e251-48e5-9b69-59e596caa0ed
𝐠(𝐱) = 𝕀(all(𝐱 .≥ 𝛄)) # probability that value is above some threshold 𝛄

# ╔═╡ bb34647f-f3c9-44d7-bf91-d32f7cb0d43a
md"""
#### Estimate
"""

# ╔═╡ 4eac94f9-71f1-469a-8fc4-29b5dd39dc59
nₘᵥ = 1000;

# ╔═╡ 1314c2a6-903c-4441-bd2f-1aac3878a614
𝐗 = rand(𝐪, nₘᵥ); # sample from proposal distribution q

# ╔═╡ 75ef8e35-a80d-43b1-8363-109417107a61
md"""
###### Importance sampling estimate
"""

# ╔═╡ 5f5789a4-8dfb-4538-b7ed-4f963b5a41f1
𝐰(𝐱ᵢ) = pdf(𝐩,𝐱ᵢ) / pdf(𝐪,𝐱ᵢ) # likelihood ratio (multi-variate)

# ╔═╡ c3f44bf2-0ac2-42b3-ade4-c9a71b10c3b3
𝛍 = 1/nₘᵥ * sum(𝐱ᵢ->𝐰(𝐱ᵢ)*𝐠(𝐱ᵢ), eachcol(𝐗)) # estimated mean

# ╔═╡ 46c66d24-4196-4826-9832-10f1ce385867
𝛔² = 1/nₘᵥ * sum(𝐱ᵢ->(𝐰(𝐱ᵢ)*𝐠(𝐱ᵢ) - 𝛍)^2, eachcol(𝐗)) # estimated variance

# ╔═╡ c8157008-c6d9-4bb5-a440-dc6c02844dfe
md"""
## Monte Carlo (multi-variate case)
"""

# ╔═╡ ca0345ed-734a-4b67-b3c4-885c0679b908
mcₘᵥ = 𝐠.(eachcol(rand(𝐩, 1_000_000))) # notice high number of samples!

# ╔═╡ 3fe212a8-f689-4df8-bf2c-de46a68ebcaf
sum(mcₘᵥ)

# ╔═╡ af9572e5-360f-4096-9891-41d3ce3b6019
mean(mcₘᵥ), var(mcₘᵥ)

# ╔═╡ 0e6e4c0b-44ee-45f0-b5e6-f467285ef9b2
md"""
## Helper code (multi-variate CDF):
- **Taken from [@blackeneth](https://discourse.julialang.org/u/blackeneth/summary), implementation of the multivariate CDF for MvNormal in Julia**: [https://discourse.julialang.org/t/mvn-cdf-have-it-coded-need-help-getting-integrating-into-distributions-jl/38631/15](https://discourse.julialang.org/t/mvn-cdf-have-it-coded-need-help-getting-integrating-into-distributions-jl/38631/15)
"""

# ╔═╡ 82451f0d-5cee-474d-ae48-adc1bbb1e08b
"""
Computes permuted lower Cholesky factor \$c\$ for \$R\$ which may be singular,
also permuting integration limit vectors a and b.

# Arguments
- `r::Matrix`          Matrix for which to compute lower Cholesky matrix
                        when called by mvn_cdf(), this is a covariance matrix

- `a::Vector`          column vector for the lower integration limit
                        algorithm may permutate this vector to improve integration
                        accuracy for mvn_cdf()

- `b::Vector`          column vector for the upper integration limit
                        algorithm may pertmutate this vector to improve integration
                        accuracy for mvn_cdf()
# Output
tuple An a tuple with 3 returned arrays:
1. lower Cholesky root of r
2. lower integration limit (perhaps permutated)
3. upper integration limit (perhaps permutated)

Examples
r = [1 0.25 0.2; 0.25 1 0.333333333; 0.2 0.333333333 1]
a = [-1; -4; -2]
b = [1; 4; 2]

(c, ap, bp) = _chlrdr(r,a,b)

result:
Lower cholesky root:
c = [ 1.00 0.0000 0.0000,
0.20 0.9798 0.0000,
0.25 0.2892 0.9241 ]
Permutated upper input vector:
ap = [-1, -2, -4]
Permutated lower input vector:
bp = [1, 2, 4]

Related Functions
`mvn_cdf` - multivariate Normal CDF function makes use of this function
"""
function _chlrdr(Σ,a,b)

    # Rev 1.13

    # define constants
    # 64 bit machien error 1.0842021724855e-19 ???
    # 32 bit machine error 2.220446049250313e-16 ???
    ep = 1e-10 # singularity tolerance
    if Sys.WORD_SIZE == 64
        fpsize=Float64
        ϵ = eps(0.0) # 64-bit machine error
    else
        fpsize=Float32
        ϵ = eps(0.0f0) # 32-bit machine error
    end

    if !@isdefined sqrt2π
        sqrt2π = √(2π)
    end

    # unit normal distribution
    unitnorm = Normal()

    n = size(Σ,1) # covariance matrix n x n square

    ckk = 0.0
    dem = 0.0
    am = 0.0
    bm = 0.0
    ik = 0.0

    if eltype(Σ)<:Signed
        c = copy(float(Σ))
    else
        c = copy(Σ)
    end

    if eltype(a)<:Signed
        ap = copy(float(a))
    else
        ap = copy(a)
    end

    if eltype(b)<:Signed
        bp = copy(float(b))
    else
        bp = copy(b)
    end

    d=sqrt.(diag(c))
    for i in 1:n
        if d[i] > 0.0
            c[:,i] /= d[i]
            c[i,:] /= d[i]
            ap[i]=ap[i]/d[i]     # ap n x 1 vector
            bp[i]=bp[i]/d[i]     # bp n x 1 vector
        end
    end

    y=zeros(fpsize,n) # n x 1 zero vector to start

    for k in 1:n
        ik = k
        ckk = 0.0
        dem = 1.0
        s = 0.0
        #pprinta(c)
        for i in k:n
            if c[i,i] > ϵ  # machine error
                cii = sqrt(max(c[i,i],0))

                if i>1 && k>1
                    s=(c[i,1:(k-1)].*y[1:(k-1)])[1]
                end

                ai=(ap[i]-s)/cii
                bi=(bp[i]-s)/cii
                de = cdf(unitnorm,bi) - cdf(unitnorm,ai)

                if de <= dem
                    ckk = cii
                    dem = de
                    am = ai
                    bm = bi
                    ik = i
                end
            end # if c[i,i]> ϵ
        end # for i=
        i = n

        if ik>k
            ap[ik] , ap[k] = ap[k] , ap[ik]
            bp[ik] , bp[k] = bp[k] , bp[ik]

            c[ik,ik] = c[k,k]

            if k > 1
                c[ik,1:(k-1)] , c[k,1:(k-1)] = c[k,1:(k-1)] , c[ik,1:(k-1)]
            end

            if ik<n
                c[(ik+1):n,ik] , c[(ik+1):n,k] = c[(ik+1):n,k] , c[(ik+1):n,ik]
            end

            if k<=(n-1) && ik<=n
                c[(k+1):(ik-1),k] , c[ik,(k+1):(ik-1)] = transpose(c[ik,(k+1):(ik-1)]) , transpose(c[(k+1):(ik-1),k])
            end
        end # if ik>k

        if ckk > k*ep
            c[k,k]=ckk
            if k < n
                c[k:k,(k+1):n] .= 0.0
            end

            for i in (k+1):n
                c[i,k] /= ckk
                c[i:i,(k+1):i] -= c[i,k]*transpose(c[(k+1):i,k])
            end

            if abs(dem)>ep
                y[k] = (exp(-am^2/2)-exp(-bm^2/2))/(sqrt2π*dem)
            else
                if am<-10
                    y[k] = bm
                elseif bm>10
                    y[k]=am
                else
                    y[k]=(am+bm)/2
                end
            end # if abs
        else
            c[k:n,k] .== 0.0
            y[k] = 0.0
        end # if ckk>ep*k
    end # for k=

    return (c, ap, bp)
end # function _chlrdr

# ╔═╡ 50109416-6bae-4b6f-80f8-d2a8a54810f5
"""
`qsimvnv(Σ,a,b;m=iterations)`

Computes the Multivariate Normal probability integral using a quasi-random rule
with m points for positive definite covariance matrix Σ, mean [0,…], with lower
integration limit vector a and upper integration limit vector b.

\$\\Phi_k(\\mathbf{a},\\mathbf{b},\\mathbf{\\Sigma} ) = \\frac{1}{\\sqrt{\\left | \\mathbf{\\Sigma}  \\right |{(2\\pi )^k}}}\\int_{a_1}^{b_1}\\int_{a_2}^{b_2}\\begin{align*}
 &...\\end{align*} \\int_{a_k}^{b_k}e^{^{-\\frac{1}{2}}\\mathbf{x}^t{\\mathbf{\\Sigma }}^{-1}\\boldsymbol{\\mathbf{x}}}dx_k...dx_1\$

Probability `p` is output with error estimate `e`.

# Arguments
- `Σ::AbstractArray`: positive-definite covariance matrix of MVN distribution
- `a::Vector`: lower integration limit column vector
- `b::Vector`: upper integration limit column vector
- `m::Int64`: number of integration points (default 1000*dimension)

# Example
```julia
julia> r = [4 3 2 1;3 5 -1 1;2 -1 4 2;1 1 2 5]
julia> a = [-Inf; -Inf; -Inf; -Inf]
julia> b = [1; 2; 3; 4 ]
julia> m = 5000
julia> (p,e) = qsimvnv(r,a,b;m=m)
(0.605219554009911, 0.0015718064928452481)
```
Results will vary slightly from run-to-run due to the quasi-Monte Carlo
algorithm.

Non-central MVN distributions (with non-zero mean) can use this function by adjusting
the integration limits. Subtract the mean vector, μ, from each
integration vector.

# Example
```julia
julia> #non-central MVN
julia> Σ=[4 2;2 3]
julia> μ = [1;2]
julia> a=[-Inf; -Inf]
julia> b=[2; 2]
julia> (p,e) = qsimvnv(Σ,a-μ,b-μ)
(0.4306346895870772, 0.00015776288569406053)
```
"""
function qsimvnv(Σ,a,b;m=nothing)
    #= rev 1.13

    This function uses an algorithm given in the paper
    "Numerical Computation of Multivariate Normal Probabilities", in
     J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
    Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
    Email : alangenz@wsu.edu
    The primary references for the numerical integration are
    "On a Number-Theoretical Integration Method"
    H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11, and
    "Randomization of Number Theoretic Methods for Multiple Integration"
    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.

    Re-coded in Julia from the MATLAB function qsimvnv(m,r,a,b)

    Alan Genz is the author the MATLAB qsimvnv() function.
    Alan Genz software website: http://archive.is/jdeRh
    Source code to MATLAB qsimvnv() function: http://archive.is/h5L37
    % QSIMVNV(m,r,a,b) and _chlrdr(r,a,b)
    %
    % Copyright (C) 2013, Alan Genz,  All rights reserved.
    %
    % Redistribution and use in source and binary forms, with or without
    % modification, are permitted provided the following conditions are met:
    %   1. Redistributions of source code must retain the above copyright
    %      notice, this list of conditions and the following disclaimer.
    %   2. Redistributions in binary form must reproduce the above copyright
    %      notice, this list of conditions and the following disclaimer in
    %      the documentation and/or other materials provided with the
    %      distribution.
    %   3. The contributor name(s) may not be used to endorse or promote
    %      products derived from this software without specific prior
    %      written permission.
    % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    % "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    % LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    % FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    % COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    % INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    % BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    % OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    % ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    % TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
    % OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    %

    Julia dependencies
    Distributions
    PDMats
    Primes
    Random
    LinearAlgebra

    =#

    if isnothing(m)
        m = 1000*size(Σ,1)  # default is 1000 * dimension
    end

    # check for proper dimensions
    n=size(Σ,1)
    nc=size(Σ,2)    # assume square Cov matrix nxn
    # check dimension > 1
    n >= 2   || throw(ErrorException("dimension of Σ must be 2 or greater. Σ dimension: $(size(Σ))"))
    n == nc  || throw(DimensionMismatch("Σ matrix must be square. Σ dimension: $(size(Σ))"))

    # check dimensions of lower vector, upper vector, and cov matrix match
    (n == size(a,1) == size(b,1)) || throw(DimensionMismatch("iconsistent argument dimensions. Sizes: Σ $(size(Σ))  a $(size(a))  b $(size(b))"))

    # check that a and b are column vectors; if row vectors, fix it
    if size(a,1) < size(a,2)
        a = transpose(a)
    end
    if size(b,1) < size(b,2)
        b = transpose(b)
    end

    # check that lower integration limit a < upper integration limit b for all elements
    all(a .<= b) || throw(ArgumentError("lower integration limit a must be <= upper integration limit b"))

    # check that Σ is positive definate; if not, print warning
    isposdef(Σ) || @warn "covariance matrix Σ fails positive definite check"

    # check if Σ, a, or b contains NaNs
    if any(isnan.(Σ)) || any(isnan.(a)) || any(isnan.(b))
        p = NaN
        e = NaN
        return (p,e)
    end

    # check if a==b
    if a == b
        p = 0.0
        e = 0.0
        return (p,e)
    end

    # check if a = -Inf & b = +Inf
    if all(a .== -Inf) && all(b .== Inf)
        p = 1.0
        e = 0.0
        return (p,e)
    end

    # check input Σ, a, b are floats; otherwise, convert them
    if eltype(Σ)<:Signed
        Σ = float(Σ)
    end

    if eltype(a)<:Signed
        a = float(a)
    end

    if eltype(b)<:Signed
        b = float(b)
    end

    ##################################################################
    #
    # Special cases: positive Orthant probabilities for 2- and
    # 3-dimesional Σ have exact solutions. Integration range [0,∞]
    #
    ##################################################################

    if all(a .== zero(eltype(a))) && all(b .== Inf) && n <= 3
        Σstd = sqrt.(diag(Σ))
        Rcorr = cov2cor(Σ,Σstd)

        if n == 2
            p = 1/4 + asin(Rcorr[1,2])/(2π)
            e = eps()
        elseif n == 3
            p = 1/8 + (asin(Rcorr[1,2]) + asin(Rcorr[2,3]) + asin(Rcorr[1,3]))/(4π)
            e = eps()
        end

        return (p,e)

    end

    ##################################################################
    #
    # get lower cholesky matrix and (potentially) re-ordered integration vectors
    #
    ##################################################################

    (ch,as,bs)=_chlrdr(Σ,a,b) # ch =lower cholesky; as=lower vec; bs=upper vec

    ##################################################################
    #
    # quasi-Monte Carlo integration of MVN integral
    #
    ##################################################################

    ### setup initial values
    ai=as[1]
    bi=bs[1]
    ct=ch[1,1]

    unitnorm = Normal() # unit normal distribution
    rng=RandomDevice()

    # if ai is -infinity, explicity set c=0
    # implicitly, the algorith classifies anythign > 9 std. deviations as infinity
    if ai > -9*ct
        if ai < 9*ct
            c1 = cdf.(unitnorm,ai/ct)
        else
            c1 = 1.0
        end
    else
        c1 = 0.0
    end

    # if bi is +infinity, explicity set d=0
    if bi > -9*ct
        if bi < 9*ct
            d1 = cdf(unitnorm,bi/ct)
        else
            d1 = 1.0
        end
    else
        d1 = 0.0
    end

    #n=size(Σ,1)    # assume square Cov matrix nxn
    cxi=c1          # initial cxi; genz uses ci but it conflicts with Lin. Alg. ci variable
    dci=d1-cxi      # initial dcxi
    p=0.0           # probablity = 0
    e=0.0           # error = 0

    # Richtmyer generators
    ps=sqrt.(primes(Int(floor(5*n*log(n+1)/4)))) # Richtmyer generators
    q=ps[1:n-1,1]
    ns=12
    nv=Int(max(floor(m/ns),1))

    Jnv    = ones(1,nv)
    cfill  = transpose(fill(cxi,nv))    # evaulate at nv quasirandom points row vec
    dpfill = transpose(fill(dci,nv))
    y      = zeros(n-1,nv)              # n-1 rows, nv columns, preset to zero

    #=Randomization loop for ns samples
     j is the number of samples to integrate over,
         but each with a vector nv in length
     i is the number of dimensions, or integrals to comptue =#

    for j in 1:ns                   # loop for ns samples
        c  = copy(cfill)
        dc = copy(dpfill)
        pv = copy(dpfill)
            for i in 2:n
                x=transpose(abs.(2.0 .* mod.((1:nv) .* q[i-1] .+ rand(rng),1) .- 1))     # periodizing transformation
                # note: the rand() is not broadcast -- it's a single random uniform value added to all elements
                y[i-1,:] = quantile.(unitnorm,c .+ x.*dc)
                s = transpose(ch[i,1:i-1]) * y[1:i-1,:]
                ct=ch[i,i]                                      # ch is cholesky matrix
                ai=as[i] .- s
                bi=bs[i] .- s
                c=copy(Jnv)                                     # preset to 1 (>9 sd, +∞)
                d=copy(Jnv)                                     # preset to 1 (>9 sd, +∞)

                c[findall(x -> isless(x,-9*ct),ai)] .= 0.0      # If < -9 sd (-∞), set to zero
                d[findall(x -> isless(x,-9*ct),bi)] .= 0.0      # if < -9 sd (-∞), set to zero
                tstl = findall(x -> isless(abs(x),9*ct),ai)     # find cols inbetween -9 and +9 sd (-∞ to +∞)
                c[tstl] .= cdf.(unitnorm,ai[tstl]/ct)           # for those, compute Normal CDF
                tstl = (findall(x -> isless(abs(x),9*ct),bi))   # find cols inbetween -9 and +9 sd (-∞ to +∞)
                d[tstl] .= cdf.(unitnorm,bi[tstl]/ct)
                @. dc=d-c
                @. pv=pv * dc
            end # for i=
        d=(mean(pv)-p)/j
        p += d
        e=(j-2)*e/j+d^2
    end # for j=

    e=3*sqrt(e)     # error estimate is 3 times standard error with ns samples

    return (p,e)    # return probability value and error estimate
end # function qsimvnv

# ╔═╡ a5f0e937-0b5a-4985-b4a3-6cc9c1dd424b
function Distributions.cdf(N::MvNormal, 𝐱::Vector;
		     a=fill(0, length(N.μ)), b=fill(Inf, length(N.μ)), m=nothing)
	return 1 - first(qsimvnv(N.Σ, a-N.μ+𝐱, b-N.μ+𝐱; m=m))
end	

# ╔═╡ e1a91265-512b-4376-95c6-e29dfee5ad87
1 - cdf(p, γ) # truth: hard to compute probability (risk event estimation)

# ╔═╡ b58eb4ed-510d-419b-b4e1-19f271d88fa7
1 - cdf(𝐩, 𝛄) # truth: hard to compute probability (risk event estimation)

# ╔═╡ ff145501-5313-4ab6-865b-c5d24dcbfa70
function testmvn(;m=nothing)
    #Typical Usage/Example Code
    #Example multivariate Normal CDF for various vectors
    println()

    # from MATLAB documentation
    r = [4 3 2 1;3 5 -1 1;2 -1 4 2;1 1 2 5]
    a = [-Inf; -Inf; -Inf; -Inf] # -inf for each
    b = [1; 2; 3; 4 ]
    m = 5000
    #m=4000 # rule of thumb: 1000*(number of variables)
    (myp,mye)=qsimvnv(r,a,b;m=m)
    println("Answer should about 0.6044 to 0.6062")
    println(myp)
    println("The Error should be <= 0.001 - 0.0014");
    println(mye)

    r=[1 0.6 0.333333;
       0.6 1 0.733333;
       0.333333 0.733333 1]
    r  = [  1   3/5   1/3;
          3/5    1    11/15;
          1/3  11/15    1]
    a=[-Inf;-Inf;-Inf]
    b=[1;4;2]
    #m=3000;
    (myp,mye)=qsimvnv(r,a,b;m=4000)
    println()
    println("Answer shoudl be about 0.82798")
    # answer from Genz paper 0.827984897456834
    println(myp)
    println("The Error should be <= 2.5-05")
    println(mye)

    r=[1 0.25 0.2; 0.25 1 0.333333333; 0.2 0.333333333 1]
    a=[-1;-4;-2]
    b=[1;4;2]
    #m=3000;
    (myp,mye)=qsimvnv(r,a,b;m=4000);
    println()
    println("Answer should be about 0.6537")
    println(myp)
    println("The Error should be <= 2.5-05")
    println(mye)

    # Genz problem 1.5 p. 4-5  & p. 63
    # correct answer F(a,b) = 0.82798
    r = [1/3 3/5 1/3;
         3/5 1.0 11/15;
         1/3 11/15 1.0]
    a = [-Inf; -Inf; -Inf]
    b = [1; 4; 2]
    (myp,mye)=qsimvnv(r,a,b;m=4000)
    println()
    #println("Answer shoudl be about 0.82798")
    println("Answer should be 0.9432")
    println(myp)
    println("The error should be < 6e-05")
    println(mye)

    # Genz page 63 uses a different r Matrix
    r = [1 0 0;
        3/5 1 0;
        1/3 11/15 1]
	a = [-Inf; -Inf; -Inf]
	b = [1; 4; 2]
    (myp,mye)=qsimvnv(r,a,b;m=4000)
    println()
    println("Answer shoudl be about 0.82798")
    println(myp)
    println("The error should be < 6e-05")
    println(mye)
    # mystery solved - he used the wrong sigma Matrix
    # when computing the results on page 6


    # singular cov Example
    r = [1 1 1; 1 1 1; 1 1 1]
    a = [-Inf, -Inf, -Inf]
    b = [1, 1, 1]
    (myp,mye)=qsimvnv(r,a,b)
    println()
    println("Answer should be 0.841344746068543")
    println(myp)
    println("The error should be 0.0")
    println(mye)
    println("Cov matrix is singular")
    println("Problem reduces to a univariate problem with")
    println("p = cdf.(Normal(),1) = ",cdf.(Normal(),1))

    # 5 dimensional Example
    #c = LinearAlgebra.tri(5)
    r = [1 1 1 1 1;
         1 2 2 2 2;
         1 2 3 3 3;
         1 2 3 4 4;
         1 2 3 4 5]
    a = [-1,-2,-3,-4,-5]
    b = [2,3,4,5,6]
    (myp,mye)=qsimvnv(r,a,b)
    println()
    println("Answer should be ~ 0.7613")
    # genz gives 0.4741284  p. 5 of book
    # Julia, MATLAB, and R all give ~0.7613 !
    println(myp)
    println("The error should be < 0.001")
    println(mye)

    # genz reversed the integration limits when computing
    # see p. 63
    a = sort!(a)
    b = 1 .- a
    (myp,mye)=qsimvnv(r,a,b)
    println()
    println("Answer should be ~ 0.4741284")
    # genz gives 0.4741284  p. 5 of book
    # now matches with reversed integration limits
    println(myp)
    println("The error should be < 0.001")
    println(mye)

    # positive orthant of above
    a = [0,0,0,0,0]
    (myp,mye)=qsimvnv(r,a,b)
    println()
    println("Answer should be ~  0.11353418")
    # genz gives 0.11353418   p. 6 of book
    println(myp)
    println("The error should be < 0.001")
    println(mye)

    # now with a = -inf
    a = [-Inf,-Inf,-Inf,-Inf,-Inf]
    (myp,mye)=qsimvnv(r,a,b)
    println()
    println("Answer should be ~ 0.81031455")
    # genz gives 0.81031455  p. 6 of book
    println(myp)
    println("The error should be < 0.001")
    println(mye)

    # eight dimensional Example
    r = [1 1 1 1 1 1 1 1;
         1 2 2 2 2 2 2 2;
         1 2 3 3 3 3 3 3;
         1 2 3 4 4 4 4 4;
         1 2 3 4 5 5 5 5;
         1 2 3 4 5 6 6 6;
         1 2 3 4 5 6 7 7;
         1 2 3 4 5 6 7 8]
    a = -1*[1,2,3,4,5,6,7,8]
    b = [2,3,4,5,6,7,8,9]
    (myp,mye)=qsimvnv(r,a,b)
    println()
    println("Answer should be ~ 0.7594")
    # genz gives 0.32395   p. 6 of book
    # MATLAB gives 0.7594362
    println(myp)
    println("The error should be < 0.001")
    println(mye)


    # eight dim orthant
    a=[0,0,0,0,0,0,0,0]
    b=[Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]
    (myp,mye)=qsimvnv(r,a,b)
    println()
    println("Answer should be ~ 0.196396")
    # genz gives 0.076586  p. 6 of book
    # MATLB gives 0.196383
    println(myp)
    println("The error should be < 0.001")
    println(mye)

    # eight dim with a = -inf
    a=-Inf*[1,1,1,1,1,1,1,1]
    b = [2,3,4,5,6,7,8,9]
    #b = [0,0,0,0,0,0,0,0]
    (myp,mye)=qsimvnv(r,a,b)
    println()
    println("Answer should be ~ 0.9587")
    # genz gives 0.69675    p. 6 of book
    # MATLAB gives 0.9587
    println(myp)
    println("The error should be < 0.001")
    println(mye)
end # testmvn

# ╔═╡ 0d7c5de8-bfda-4938-8a70-9955505e52b9
testmvn()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Primes = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
ColorSchemes = "~3.17.1"
Distributions = "~0.25.48"
PDMats = "~0.11.5"
Plots = "~1.25.9"
PlutoUI = "~0.7.34"
Primes = "~0.5.1"
StatsBase = "~0.33.15"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f9982ef575e19b0e5c7a98c6e75ee496c0f73a93"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "38012bf3553d01255e83928eec9c998e19adfddf"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.48"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "deed294cde3de20ae0b2e0355a6c4e1c6a5ceffc"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.8"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "4a740db447aae0fbeb3ee730de1afbb14ac798a1"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.63.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa22e1ee9e722f1da183eb33370df4c1aeb6c2cd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.63.1+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "13468f237353112a01b2d6b32f3d0f80219944aa"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "1d0a11654dbde41dc437d6733b68ce4b28fbe866"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.9"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8979e9802b4ac3d58c503a20f2824ad67f9074dd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.34"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.Primes]]
git-tree-sha1 = "984a3ee07d47d401e0b823b7d30546792439070a"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "37c1631cb3cc36a535105e6d5557864c82cd8c2b"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "8d0c8e3d0ff211d9ff4a0c2307d876c99d10bdf1"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "a635a9333989a094bddc9f940c04c549cd66afcf"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "118e8411d506d583fbbcf4f3a0e3c5a9e83370b8"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.15"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f35e1879a71cca95f4826a14cdbf0b9e253ed918"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.15"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─c85db890-8bcb-11ec-3a45-2b0d44ba0abc
# ╠═8f20ec09-2a13-4130-989d-6edd87deb89a
# ╠═4ec759e2-6a6e-493e-9e7d-4b9eeba0ebdd
# ╠═a3f021c4-03be-49d9-b27e-f050caedfcd3
# ╠═42be2255-9bb8-4740-958f-9bfe9752f27b
# ╠═3fa3ba32-ee4b-4c55-a0d5-2ea9a848d76f
# ╟─ef39b5ca-715b-4d22-aa89-513d16c31ff2
# ╠═b9d672c1-f71e-4813-9214-84ed490f4914
# ╠═46bcc8a0-2117-4b85-8a44-fcaa0bba6868
# ╠═65668688-a98b-4afa-9d06-68c03ece67b0
# ╠═4212859a-8859-4b0a-9223-ccb1afc8cfcd
# ╠═873ef3ba-c803-46a6-b864-8b88903de7ec
# ╠═70a2a4bb-f4e6-44cb-845e-a8b33999b514
# ╟─6cfada66-a72c-4eb4-9aa9-408a6d16c0ca
# ╠═e1a91265-512b-4376-95c6-e29dfee5ad87
# ╠═34ffdb0b-eb92-4de2-a0e1-2a38c2b15373
# ╠═fbaba7a8-ca9f-445f-baf5-2c02b5029603
# ╠═47ae008a-bd0a-45d5-b09d-2b18621994a2
# ╠═fe2a153b-e90f-4826-8058-cfb9a6c2b41b
# ╟─10f24b89-0c77-4e6c-b10c-14879582c325
# ╠═6fc930f6-7dd0-4aa0-8ce7-30a5582012e0
# ╠═32bbfa28-1a47-4b1b-b2c5-bcc0d14d2f6e
# ╟─14dd7ac0-4367-4bfc-9717-ccdb10bda171
# ╠═45504a50-9ce2-4457-9493-26127f9f537e
# ╠═363154fb-f4e7-4a4a-a878-5e458b738cc0
# ╟─ccdf4496-bee6-4701-940b-e4560149fabe
# ╠═2c2ad947-298a-431b-b9e5-c74a25b0439c
# ╠═e9e0ad28-8d67-4a1d-9798-808b0632e87c
# ╠═e2d1d150-40f5-41df-9bcd-131f8fb14537
# ╟─54c65259-7e7d-4bd2-9983-82a2e957d808
# ╟─f796718c-bcdc-4d5a-be3f-fe882c2679f2
# ╠═2fb742a3-6d96-4332-a9aa-62db643f0811
# ╠═add76ee3-2e70-4a55-84c1-2cb20bc5e30c
# ╠═8c112c36-00c8-42d3-bb10-5966fbef81d9
# ╟─a4876a1a-bbf1-4b7a-83e0-58c4af899c36
# ╠═7667881d-5546-452d-a630-1c972fdec485
# ╠═fff3e2e1-bee5-4a2e-9570-85f8dfac82ca
# ╠═b26c338c-4274-42d5-b1af-b5558a584a58
# ╠═9604d9b2-9aa2-4131-8f42-74ceb74cf0e9
# ╠═7cf56c73-5022-45c3-b995-974d122eea16
# ╟─5c4ab4a0-caab-4670-9edd-c96bb361ffa4
# ╠═fc283c41-11ee-4d7f-b2d6-311dff8777bd
# ╠═f5ee8b07-0edf-4fd3-8584-cbce68e8a06a
# ╟─8df63154-bfdb-46ab-8d71-9a67dbf0e453
# ╠═841ec98d-7e80-4cfc-a92e-8cd700235390
# ╠═96349c9f-1421-4d4c-ae28-3f076da83dc6
# ╟─d8af9650-927e-4069-a8ce-7d1428aa8653
# ╠═b895f340-a407-4637-9a4a-0be5f481ce36
# ╠═44bf340a-8f11-4a1d-ac90-9692e4adec4d
# ╠═5c7352f8-a0d5-4b58-bda9-7c3932601489
# ╠═1d332efa-30c3-4523-9c00-d7c9d9e32f63
# ╠═aeaff93a-84bf-4b47-90f7-fa16b9ed708c
# ╠═9facd395-91cd-46f9-86a8-7c7375a9c4a7
# ╟─7161fab4-834e-45ec-96c3-cafe2742f96d
# ╠═b58eb4ed-510d-419b-b4e1-19f271d88fa7
# ╠═736d48ec-e251-48e5-9b69-59e596caa0ed
# ╟─bb34647f-f3c9-44d7-bf91-d32f7cb0d43a
# ╠═4eac94f9-71f1-469a-8fc4-29b5dd39dc59
# ╠═1314c2a6-903c-4441-bd2f-1aac3878a614
# ╟─75ef8e35-a80d-43b1-8363-109417107a61
# ╠═c3f44bf2-0ac2-42b3-ade4-c9a71b10c3b3
# ╠═46c66d24-4196-4826-9832-10f1ce385867
# ╠═5f5789a4-8dfb-4538-b7ed-4f963b5a41f1
# ╟─c8157008-c6d9-4bb5-a440-dc6c02844dfe
# ╠═ca0345ed-734a-4b67-b3c4-885c0679b908
# ╠═3fe212a8-f689-4df8-bf2c-de46a68ebcaf
# ╠═af9572e5-360f-4096-9891-41d3ce3b6019
# ╟─0e6e4c0b-44ee-45f0-b5e6-f467285ef9b2
# ╠═4147c74d-10bc-48aa-a725-fea135d2e304
# ╠═a5f0e937-0b5a-4985-b4a3-6cc9c1dd424b
# ╟─50109416-6bae-4b6f-80f8-d2a8a54810f5
# ╟─82451f0d-5cee-474d-ae48-adc1bbb1e08b
# ╟─ff145501-5313-4ab6-865b-c5d24dcbfa70
# ╠═0d7c5de8-bfda-4938-8a70-9955505e52b9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
