### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 0b7610be-fea4-11ea-0a69-eff0a019b6e6
using LinearAlgebra, TikzPictures

# ╔═╡ 32b06dc0-fef4-11ea-25f2-cdaeafd52ab4
using TikzNeuralNetworks

# ╔═╡ 3945e080-ffac-11ea-23a8-b968fe84d903
using Statistics # LinearAlgebra

# ╔═╡ 3641cda0-ffbf-11ea-27a3-5fe417918cc0
using AddPackage

# ╔═╡ 991f6770-ffbf-11ea-048c-1113d8640637
@add using PyPlot, ScikitLearn, Random, Parameters

# ╔═╡ 0414b960-ffd7-11ea-313c-3fbce53d7a70
using Zygote

# ╔═╡ dbd0ccc0-fea3-11ea-359e-876ee15179be
md"# Neural networks"

# ╔═╡ 047d3eb2-fea4-11ea-3ff7-0b15c54073e2
md"## Activation functions"

# ╔═╡ 08c79290-fea4-11ea-0a77-e5f5e6ab401d
ReLU(z) = max(z, 0)

# ╔═╡ c8f6b2c0-ffd1-11ea-3972-25233fa64f94
ReLU′(z) = z < 0 ? 0 : 1

# ╔═╡ 1384e6b0-fea4-11ea-2e03-cdad79f986d5
σ(z) = 1/(1 + exp(-z))

# ╔═╡ ae23a200-ffd1-11ea-3cfe-bb895cf90087
σ′(z) = σ(z)*(1 - σ(z))

# ╔═╡ be738170-ffd1-11ea-23ec-45adb917e9c6
tanh′(z) = 1 - tanh(z)^2

# ╔═╡ 233ec7b0-fea4-11ea-3a47-a3125ec028be
md"## Two-layer neural network"

# ╔═╡ a13c0ba0-fea4-11ea-0fa7-cde0a3840c99
TikzPicture(L"""
\node (phi2) [nnnode, label=left:{$\phi(x)_2$}] {};
\node (phi1) [nnnode, above=2mm of phi2, label=left:{$\phi(x)_1$}] {};
\node (phi3) [nnnode, below=2mm of phi2, label=left:{$\phi(x)_3$}] {};

\node (hidden1) [nnnode, above right=0mm and 15mm of phi2, label=above:{$h_1$}] {$\darkblue{\sigma}$};
\node (hidden2) [nnnode, below right=0mm and 15mm of phi2, label=below:{$h_2$}] {$\darkblue{\sigma}$};

\node (score) [nnnode, right=30mm of phi2, label=above:{score}] {};

\draw[->] (phi1) -- (hidden1) node [midway, yshift=7pt] {$\darkred{\mathbf{V}}$};
\draw[->] (hidden1) -- (score) node [midway, yshift=7pt] {$\darkred{\mathbf{w}}$};
\draw[->] (phi1) -- (hidden2);
\draw[->] (phi2) -- (hidden1);
\draw[->] (phi2) -- (hidden2);
\draw[->] (phi3) -- (hidden1);
\draw[->] (phi3) -- (hidden2);
\draw[->] (hidden1) -- (score);
\draw[->] (hidden2) -- (score);	
""",
preamble="""
\\definecolor{stanfordred}{RGB}{140,21,21}
\\definecolor{darkblue}{RGB}{21,21,140}
\\newcommand{\\darkblue}[1]{{\\color{darkblue} #1}}
\\newcommand{\\darkred}[1]{{\\color{stanfordred} #1}}
\\usetikzlibrary{positioning}
\\usetikzlibrary{arrows}
\\tikzset{nnnode/.style = {circle, draw=black, fill=white, minimum size=16pt,},}
\\tikzset{every picture/.style={semithick, >=stealth'}}
""",
width="12cm"
)

# ╔═╡ 11ab44b0-fea4-11ea-042f-a3547f3e2a8b
function neural_network(x, 𝐕, 𝐰, φ, g=ReLU)
    𝐡 = map(𝐯ⱼ -> g(𝐯ⱼ ⋅ φ(x)), 𝐕)
    𝐰 ⋅ 𝐡
end

# ╔═╡ 12cfa070-fea4-11ea-0963-9548b2c07c44
md"## Multi-layer neural network
Also called a *multi-layer perceptron*."

# ╔═╡ 3c132240-fea4-11ea-0b96-8fac4ceb5841
function multi_layer_neural_network(x, 𝐖, φ, 𝐠)
    𝐡ᵢ = φ(x)
    for (i,g) in enumerate(𝐠)
        𝐡ᵢ = map(𝐰ⱼ -> g(𝐰ⱼ ⋅ 𝐡ᵢ), 𝐖[i])
    end
    𝐡ᵢ ⋅ last(𝐖)
end

# ╔═╡ a43062c0-0020-11eb-245e-cf6b2036b878
function plot_nn()
	nn2 = TikzNeuralNetwork(input_size=2,
		input_label=i->"\$x_{$i}\$",
		input_arrows=false,
		hidden_layer_sizes=[4, 1],
		activation_functions=[L"g", L"\sigma"],
		hidden_layer_labels=[L"\tanh", "logistic"],
		output_label=i->L"\hat{y}",
		output_arrows=false,
		output_size=1)
	nn2.tikz.width="12cm"
	return nn2
end

# ╔═╡ b17c5100-0020-11eb-2928-81d80731960e
plot_nn()

# ╔═╡ c43d5c20-ff9f-11ea-15ce-3bef26583d2a
md"""
$$\begin{gather}
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \mathbf{a}^{[0]} = (3\times1) \text{ vector}
\end{gather}$$
"""

# ╔═╡ 492db7ae-ff9e-11ea-38a0-09cead1c5ae6
md"""
For weights $\mathbf{W}$:

$$\begin{gather}
\mathbf{W}^{[1]} = \begin{bmatrix}
& — & \mathbf{w}_1^{[1]\top} & — & \\
& — & \mathbf{w}_2^{[1]\top} & — & \\
& — & \mathbf{w}_3^{[1]\top} & — & \\
& — & \mathbf{w}_4^{[1]\top} & — & \\
\end{bmatrix} = (|\mathbf{a}^{[1]}| \times |\mathbf{x}|) = (4\times3) \text{ matrix}\\
\end{gather}$$
"""

# ╔═╡ e7ffb7c0-ff9f-11ea-3c0e-cbb878af2c6e
md"""
$$\begin{gather}
\mathbf{b}^{[1]} = \begin{bmatrix} b_1^{[1]} \\ b_2^{[1]} \\ b_3^{[1]} \\ b_4^{[1]} \end{bmatrix} = (|\mathbf{a}^{[1]}| \times 1) = (4\times1) \text{ vector}
\end{gather}$$
"""

# ╔═╡ 180dc51e-ff9f-11ea-3ad0-a51012f662b0
md"""
$$\begin{gather}
\mathbf{z}^{[1]} = 
\mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} = 
\begin{bmatrix}
\mathbf{w}_1^{[1]\top} \mathbf{x} + b_1^{[1]}\\
\mathbf{w}_2^{[1]\top} \mathbf{x} + b_2^{[1]}\\
\mathbf{w}_3^{[1]\top} \mathbf{x} + b_3^{[1]}\\
\mathbf{w}_4^{[1]\top} \mathbf{x} + b_4^{[1]}\\
\end{bmatrix} = 
\begin{bmatrix}
z_1^{[1]} \\ z_2^{[1]} \\ z_3^{[1]} \\ z_4^{[1]} \\
\end{bmatrix} = (4\times1) \text{ vector}
\end{gather}$$
"""

# ╔═╡ 07054ea0-ffa0-11ea-1fd1-6f70c3600784
md"""
$$\begin{gather}
\mathbf{a}^{[1]} = \begin{bmatrix} a_1^{[1]} \\ a_2^{[1]} \\ a_3^{[1]} \\ a_4^{[1]} \end{bmatrix} = \sigma(\mathbf{z}^{[1]}) = (4\times1) \text{ vector}
\end{gather}$$
"""

# ╔═╡ d75e9920-ffa1-11ea-1bb0-6d0424b2b5b5
md"""
Then for the next last hidden layer to output layer.

$$\begin{gather}
\mathbf{W}^{[2]} = \mathbf{w}^\top = \begin{bmatrix} w_1^{[2]} &  w_1^{[2]} & w_1^{[2]} & w_1^{[2]} \end{bmatrix} = (1\times4) \text{ matrix}\\
\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]} = (1\times1) \text{ vector}\\
\mathbf{a}^{[2]} = \sigma(\mathbf{z}^{[2]}) = \hat{y}
\end{gather}$$
"""

# ╔═╡ b70c8f9e-ffa2-11ea-1f51-cd5660e3eb37
md"""
## Vectorization

$$\begin{align}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}\\
\mathbf{a}^{[1]} &= \sigma(\mathbf{z}^{[1]})\\
\mathbf{z}^{[2]} &= \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}\\
\mathbf{a}^{[2]} &= \sigma(\mathbf{z}^{[2]})
\end{align}$$

Vectorizing across multiple examples $i$.

$$\begin{align}
\texttt{for i }&\texttt{in 1 to m:}\\
\mathbf{z}^{[1](i)} &= \mathbf{W}^{[1]} \mathbf{x}^{(i)} + \mathbf{b}^{[1]}\\
\mathbf{a}^{[1](i)} &= \sigma(\mathbf{z}^{[1](i)})\\
\mathbf{z}^{[2](i)} &= \mathbf{W}^{[2]} \mathbf{a}^{[1](i)} + \mathbf{b}^{[2]}\\
\mathbf{a}^{[2](i)} &= \sigma(\mathbf{z}^{[2](i)})
\end{align}$$
"""

# ╔═╡ 0e09c0d0-ff9d-11ea-0e5e-37f227129043
md"""
For $m$ training examples, the input data $\mathbf{X}$ is represented as a matrix.

$$\begin{gather}
\mathbf{X} = \begin{bmatrix}
\mid & \mid & & \mid\\
\mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)}\\
\mid & \mid & & \mid\\
\end{bmatrix} = (n_x \times m) = (3 \times m) \text{ matrix}\\
\end{gather}$$
"""

# ╔═╡ e3dd9740-ff9d-11ea-0d04-ad6f8be98423
md"""
$$\begin{gather}
\mathbf{Z}^{[1]} = \begin{bmatrix}
\mid & \mid & & \mid\\
\mathbf{z}^{[1](1)} & \mathbf{z}^{[1](2)} & \cdots & \mathbf{z}^{[1](m)}\\
\mid & \mid & & \mid\\
\end{bmatrix} = (|\mathbf{z}^{[1]}| \times m) = (4\times m) \text{ matrix}\\
\end{gather}$$
"""

# ╔═╡ 385750e0-ffa3-11ea-03d1-a7cc682d4ab2
md"""
$$\begin{gather}
\mathbf{A}^{[1]} = \begin{bmatrix}
\mid & \mid & & \mid\\
\mathbf{a}^{[1](1)} & \mathbf{a}^{[1](2)} & \cdots & \mathbf{a}^{[1](m)}\\
\mid & \mid & & \mid\\
\end{bmatrix} = (|\mathbf{a}^{[1]}| \times m) = (4 \times m) \text{ matrix}\\
\end{gather}$$
"""

# ╔═╡ e2891c10-ffa3-11ea-05a6-f93d0bd8daf8
md"""
Vectorized over all training examples.

$$\begin{align}
\mathbf{Z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{X} + \mathbf{b}^{[1]}\\
\mathbf{A}^{[1]} &= \sigma(\mathbf{Z}^{[1]})\\
\mathbf{Z}^{[2]} &= \mathbf{W}^{[2]} \mathbf{A}^{[1]} + \mathbf{b}^{[2]}\\
\mathbf{A}^{[2]} &= \sigma(\mathbf{Z}^{[2]})
\end{align}$$

Simplified to:

$$\begin{align}
\mathbf{Z}^{[\ell]} &= \mathbf{W}^{[\ell]} \mathbf{A}^{[\ell-1]} + \mathbf{b}^{[\ell]}\\
\mathbf{A}^{[\ell]} &= \sigma(\mathbf{Z}^{[\ell]})
\end{align}$$

Simplified further, with a generalized activation function $g$:

$$\begin{gather}
\mathbf{A}^{[\ell]} = g(\mathbf{W}^{[\ell]} \mathbf{A}^{[\ell-1]} + \mathbf{b}^{[\ell]})
\end{gather}$$
"""

# ╔═╡ 66219680-ffa7-11ea-3528-9184d31f42de
md"""
## Activation functions

Activation functions $a = g(z)$:

For output layer, $\sigma$ works to map between $0-1$:

$$\begin{gather}
\sigma(z) = \frac{1}{1 + e^{-z}}\\
\end{gather}$$

Mean of data closer to zero with $\tanh$, usually works better than $\sigma$ as a hidden layer:

$$\begin{gather}
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\\
\end{gather}$$

The rectified linear unit:

$$\text{ReLU}(z) = \max(0, z)$$

The leaky $\text{ReLU}$:

$$\text{Leaky ReLU}(z) = \max(0.01 z, z)$$

For different activation functions $g^{[1]}$ for first hidden layer, $g^{[2]}$ for second hidden layer, etc.

**Rules to choose activation functions:**
- If output is $\{0, 1\}$, i.e. binary classification, sigmoid is a natural choice for the output layer $\sigma$. For all other layers, $\text{ReLU}$ is a good choice.

**Derivatives**

$$\begin{align}
\sigma^\prime(z) &= \sigma(z)(1 - \sigma(z))\\
\tanh^\prime(z) &= 1 - \tanh(z)^2\\
\text{ReLU}^\prime &= \begin{cases}
0 & \text{if } z < 0\\
1 & \text{if } z > 0\\
\text{undef} & \text{if } z = 0
\end{cases}\\
\text{Leaky ReLU}^\prime &= \begin{cases}
0.01 & \text{if } z < 0\\
1 & \text{if } z > 0\\
\text{undef} & \text{if } z = 0
\end{cases}
\end{align}$$
"""

# ╔═╡ 30cb2962-ffac-11ea-114f-73601457cef0
md"""
## Gradient descent
"""

# ╔═╡ c708c540-ffac-11ea-0632-35fdb0b7a4ef
md"""
$$\begin{gather}
n_x = n^{[0]}, n^{[1]}, n^{[2]} = 1 \tag{in this example}\\
\mathbf{W}^{[1]} = (n^{[1]} \times n^{[0]})\\
\mathbf{b}^{[1]} = (n^{[1]} \times 1)\\
\mathbf{W}^{[1]} = (n^{[2]} \times n^{[1]})\\
\mathbf{b}^{[2]} = (n^{[2]} \times 1)\\
\end{gather}$$

$$J(\theta) = J\left(\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \mathbf{W}^{[2]}, \mathbf{b}^{[2]}\right) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(\hat{y}, y)$$

$$\partial \mathbf{W}^{[1]} = \frac{\partial J}{\partial \mathbf{W}^{[1]}},\quad
\partial \mathbf{b}^{[1]} = \frac{\partial J}{\partial \mathbf{b}^{[1]}}, \quad \ldots$$

Then the gradient update would be:

$$\begin{gather}
\mathbf{W}^{[1]} = \mathbf{W}^{[1]} - \alpha (\partial \mathbf{W}^{[1]})\\
\mathbf{b}^{[1]} = \mathbf{b}^{[1]} - \alpha (\partial \mathbf{b}^{[1]})\\
\ldots
\end{gather}$$
"""

# ╔═╡ 596a9110-ffae-11ea-2c89-7518df5b04c6
md"""
### Backpropogation

[http://neuralnetworksanddeeplearning.com/chap2.html](http://neuralnetworksanddeeplearning.com/chap2.html)

$$\mathbf{Y} = \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix}$$

$$\begin{gather}
\partial \mathbf{Z}^{[2]} = \mathbf{A}^{[2]} - \mathbf{Y}\\
\partial \mathbf{W}^{[2]} = \frac{1}{m} \partial \mathbf{Z}^{[2]} \mathbf{A}^{[1]\top}\\
\partial \mathbf{b}^{[2]} = \frac{1}{m} \sum \partial \mathbf{Z}^{[2]}\\
\partial \mathbf{Z}^{[1]} = \mathbf{W}^{[2]\top} \partial \mathbf{Z}^{[2]} \odot g^{[1]\prime}(\mathbf{Z}^{[1]}) \tag{derivative $g^\prime$}\\
(n^{[1]} \times m) \odot (n^{[1]} \times m) \tag{element-wise product}\\
\partial \mathbf{W}^{[1]} = \frac{1}{m} \partial \mathbf{Z}^{[1]}\mathbf{X}^\top\\
\partial \mathbf{b}^{[1]} = \frac{1}{m} \sum \partial \mathbf{Z}^{[1]} 
\end{gather}$$

`numpy`: `np.sim(dZ[1], axis=1, keepdims=True)`
"""


# ╔═╡ 35d0b6f0-ffac-11ea-27b1-47b519c13c21
function gradient_descent(𝒟train, φ, ∇loss; η=0.1, T=100)
    𝐰 = 0.01randn(length(φ(𝒟train[1][1])))
    for t in 1:T
        𝐰 = 𝐰 .- η*mean(∇loss(x, y, 𝐰, φ) for (x,y) ∈ 𝒟train)
    end
    return 𝐰
end

# ╔═╡ a2bcbdb0-ffa0-11ea-0a41-457fa1b70946
md"""
## Logistic regression
Simple neural network.

$$\begin{gather}
z = \mathbf{w}^\top \mathbf{x} + b\\
a = \sigma(z) = \hat{y}
\end{gather}$$
"""

# ╔═╡ 68ca10f0-ff49-11ea-3147-09eb71687f20
begin
	logreg = TikzNeuralNetwork(input_size=3,
		input_arrows=false,
		input_label=i->"\$\\mathbf{x}_{$i}\$",
		activation_functions=["\$\\sigma(\\theta^\\top x)\$"],
		output_label=i->L"\hat{y}",
		output_arrows=false)
	logreg.tikz.width="10cm"
	logreg
end

# ╔═╡ 24692920-ffb0-11ea-3ab2-47b00719deed
begin
	logreg_spanned = TikzNeuralNetwork(input_size=3,
		input_arrows=false,
		input_label=i->Dict(1=>L"x", 2=>L"w", 3=>L"b")[i],
		activation_functions=[L"z=w^\top x + b", L"a=\sigma(z)"],
		hidden_layer_sizes=[1,1],
		output_label=i->L"\mathcal{L}(a,y)",
		output_arrows=false)
	logreg_spanned.tikz.width="10cm"
	logreg_spanned.tikz.data = replace(logreg_spanned.tikz.data, "fill=lightgray!70"=>"fill=lightgray!70, rectangle")
	logreg_spanned
end

# ╔═╡ 5a430f62-ffb1-11ea-2123-cd71772704ee
md"""
## Coursera data classification (port)
"""

# ╔═╡ 7438c7c0-001a-11eb-1b09-b593715f3e82
PyPlot.svg(true);

# ╔═╡ 55eb5030-ffc0-11ea-0688-c1ecd12f1939
@sk_import datasets: make_moons

# ╔═╡ 98799310-ffd6-11ea-0714-17b135c4c704
@sk_import datasets: make_circles

# ╔═╡ efce9b80-ffc0-11ea-2cb3-4f8209f29cdc
begin
	X, Y = make_moons(n_samples=200, noise=0.2)
 	# X, Y = make_circles(noise=0.2, factor=0.5, random_state=1)
	X, Y = X', reshape(Y, 1, size(Y)[1])
end

# ╔═╡ 10fb310e-ffc1-11ea-0d21-539799c2aebf
size(X)

# ╔═╡ 3d66f4f0-ffc1-11ea-18ae-a5063f76ded5
size(Y)

# ╔═╡ 4016f9c0-ffc1-11ea-0638-8fc7e9f2c52b
clf(); plt.scatter(X[1,:], X[2,:], c=Y, s=40, cmap=plt.cm.Spectral); gcf()

# ╔═╡ 3c930220-ffcc-11ea-0694-e73b06bf7623
@with_kw mutable struct NeuralNetwork
	input_size::Int = 2
	hidden_layer_sizes::Vector{Int} = [5, 1]
	activation_functions::Vector{Function} = [tanh, σ]
	activation_functions′::Vector{Function} = [z->(1 - tanh(z)^2), z->σ(z)*(1-σ(z))]
	output_size::Int = 1
	θ::Dict = Dict()
	forward::Dict = Dict()
	grads::Dict = Dict()
end

# ╔═╡ 6c1a4160-ffcd-11ea-249a-79972ec4eb85
nn = NeuralNetwork()

# ╔═╡ d8083910-ffc1-11ea-31f8-07d7fd4a0b9a
function initialize!(nn, X, Y)
	nn.input_size = size(X)[1]
	nn.output_size = size(Y)[1]
	nₓ = nn.input_size
	for (i,nₕ) in enumerate(vcat(nn.hidden_layer_sizes, nn.output_size))
		Wi = 0.01randn(nₕ, nₓ)
		bi = zeros(nₕ, 1)
		nₓ = nₕ
		nn.θ["W$i"] = Wi
		nn.θ["b$i"] = bi
	end
end

# ╔═╡ 1aaf8430-ffc2-11ea-0520-451f13594ddc
function forwardpass!(nn, X)
	for (i,g) in enumerate(nn.activation_functions)
		Wi = nn.θ["W$i"]
		bi = nn.θ["b$i"]
		Zi = Wi*X .+ bi
		Ai = g.(Zi)
		X = Ai # for next iteration
		nn.forward["Z$i"] = Zi
		nn.forward["A$i"] = Ai
	end
end

# ╔═╡ b6d126c0-ffc2-11ea-2400-9bd611d2d8b7
function cost(nn, Y)
	N = length(nn.activation_functions)
	Aₙ = nn.forward["A$N"]
	return -mean(Y .* log.(Aₙ) .+ (1 .- Y) .* log.(1 .- Aₙ))
end

# ╔═╡ 92237430-ffd7-11ea-3356-d9cd94e2a587
function cost2(Aₙ, Y)
	return -mean(Y .* log.(Aₙ) .+ (1 .- Y) .* log.(1 .- Aₙ))
end

# ╔═╡ a19c1100-ffd8-11ea-37d1-b3f2e0b6dc0f
linear(θ, x) = θ.W*x .+ θ.b

# ╔═╡ ea7a6f60-ffd9-11ea-395c-f769cabcc23f
loss(a, y) = -(y .* log.(a) .+ (1 .- y) .* log(1 .- a))

# ╔═╡ ba7f4af0-0020-11eb-2112-e13c9ebaec6c
plot_nn()

# ╔═╡ 835e1c70-ffc3-11ea-31a5-214731bbc366
function backpropagation!(nn, X, Y)
	m = size(X)[2]

	W1 = nn.θ["W1"]
	W2 = nn.θ["W2"]
	
	𝐀1 = nn.forward["A1"]
	𝐀2 = nn.forward["A2"]
	
	# for each activation, get the gradient
# 	θ1 = (W=nn.θ["W1"], b=nn.θ["b1"])
# 	θ2 = (W=nn.θ["W2"], b=nn.θ["b2"])
# 	d2 = first(gradient(a->mean(loss.(σ.(linear(a, 𝐀1)), Y)), θ2))
# 	d1 = first(gradient(a->mean(tanh.(linear(a, X))), θ1))
# 	dW1 = θ2.W'*d2.W * tanh'.(nn.forward["Z1"])
# 	db1 = 1/m * sum(dZ1)

	# TODO: Generalize.
	dZ2 = 𝐀2 .- Y # assumes binary classification with sigmoid
	dW2 = 1/m * dZ2*𝐀1'
	db2 = 1/m * sum(dZ2)
	dZ1 = W2'*dZ2 .* (1 .- 𝐀1.^2)
	dW1 = 1/m * dZ1*X'
	db1 = 1/m * sum(dZ1)
	
	nn.grads["dW1"] = dW1
	nn.grads["db1"] = db1
	nn.grads["dW2"] = dW2
	nn.grads["db2"] = db2

# 	nn.grads["dW1"] = d1.W
# 	nn.grads["db1"] = d1.b

# 	nn.grads["dW1"] = dW1
# 	nn.grads["db1"] = db1
# 	nn.grads["dW2"] = d2.W
# 	nn.grads["db2"] = d2.b
end

# ╔═╡ 24fd6bb0-0025-11eb-0489-5f1193474b18
md"""
$$\begin{align}
\theta &= \langle \mathbf{W}^{[j]}, \mathbf{b}^{[j]} \rangle\\
\mathbf{Z}^{[j]} &= \mathbf{W}^{[j]} \mathbf{A}^{[j-1]} + \mathbf{b}^{[j]}\\
\mathbf{A}^{[j]} &= \sigma(\mathbf{Z}^{[j]})
\end{align}$$
"""

# ╔═╡ 410ff860-ffc4-11ea-1d71-6141309e0112
function update!(nn, α=1.2)
	for i in 1:length(nn.activation_functions)	
		Wi = nn.θ["W$i"]
		bi = nn.θ["b$i"]

		dWi = nn.grads["dW$i"]
		dbi = nn.grads["db$i"]

		nn.θ["W$i"] = Wi .- α*dWi
		nn.θ["b$i"] = bi .- α*dbi
	end
end

# ╔═╡ bd826122-ffc5-11ea-05de-9995d65c1b86
function train!(nn, X, Y, niters=10000)	
	initialize!(nn, X, Y)
	
	for i in 1:niters
		forwardpass!(nn, X)
		ℒ = cost(nn, Y)
		backpropagation!(nn, X, Y)
		update!(nn)
	end

	return nn
end

# ╔═╡ 0dee25e0-ffc6-11ea-0b20-3f4781d6ff83
function predict(nn, X)
	forwardpass!(nn, X)
	N = length(nn.activation_functions)
	Aₙ = nn.forward["A$N"]
	predictions = Aₙ .> 0.5
	return predictions
end

# ╔═╡ 2504f1ee-ffc6-11ea-3b9a-d17dcf5d9b6e
mean(predict(nn, X))

# ╔═╡ 7af70d00-ffc6-11ea-1bd9-ddb90f5b04cd
begin
	trigger = nothing
	train!(nn, X, Y)
end

# ╔═╡ 1870d340-ffcc-11ea-12c1-c7765b8a2957
md"### Accuracy"

# ╔═╡ 03a3e210-ffca-11ea-26e3-132e96817f3c
begin
	trigger
	predictions = predict(nn, X)
	sum(predictions .== Y) / length(Y) * 100
end

# ╔═╡ 1f400f30-ffca-11ea-3ef6-1b1722d88043
# (Y * predictions' + (1 .- Y) * (1 .- predictions')) / length(Y)

# ╔═╡ 859b8c80-ffc7-11ea-1590-2503626327ab
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

# ╔═╡ 89572a60-ffc6-11ea-2480-af31c91aac6c
function plot_decision_boundary(model, X, y)
    # Set min and max values and give it some padding
    x_min, x_max = minimum(X[1,:]) - 1, maximum(X[1,:]) + 1
    y_min, y_max = minimum(X[2,:]) - 1, maximum(X[2,:]) + 1
    h = 0.01

	# Generate a grid of points with distance h between them
    xx, yy = meshgrid(x_min:h:x_max, y_min:h:x_max)

	# Predict the function value for the whole grid
    Z = map((x,y)->model([x, y])[1], xx, yy)

	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[1,:], X[2,:], c=y, cmap=plt.cm.Spectral)
end

# ╔═╡ 96a71ad0-ffc7-11ea-069d-1f4e1916d2ab
plot_decision_boundary(x->predict(nn, x), X, Y); gcf()

# ╔═╡ 41cda7ee-fea4-11ea-32d8-677bd3f7bc0b
md"## Examples"

# ╔═╡ 4533b650-fea4-11ea-329b-9363cdc0a214
begin
    function test_neural_network(g=σ)
        x = 2
        φ = x -> [x, x^2, sqrt(abs(x))]
        𝐕 = [[2,-1,3], [3,0,1]]
        𝐰 = [+1, -1]
        neural_network(x, 𝐕, 𝐰, φ, g)
    end

    @info test_neural_network(σ) ≈ -0.013563772681566943
    @info test_neural_network(ReLU) ≈ -3.1715728752538093

    @info σ(0) == 0.5
    @info ReLU(1) == 1
    @info ReLU(-1) == 0
end

# ╔═╡ 5d285f40-fea4-11ea-3295-b7fa7064eaae
begin
    function test_two_layer_neural_network(𝐠=[σ])
        x = 2
        φ = x -> [x, x^2, sqrt(abs(x))]
        𝐕 = [[2,-1,3], [3,0,1]]
        𝐰 = [+1, -1]
        𝐖 = [𝐕, 𝐰]
        multi_layer_neural_network(x, 𝐖, φ, 𝐠)
    end

    function test_five_layer_neural_network(𝐠)
        x = 2
        φ = x -> [x, x^2, sqrt(abs(x))]
        𝐕₁ = [[2,-1,3], [3,0,1], [7,5,3]]
        𝐕₂ = [[6,5,9], [3,3,3]]
        𝐕₃ = [[6,5], [3,3], [3,3], [3,3], [3,3]]
        𝐕₄ = [[1,2,3,4,5], [6,7,8,9,0]]
        𝐰 = [+1, -1]
        𝐖 = [𝐕₁, 𝐕₂, 𝐕₃, 𝐕₄, 𝐰]
        multi_layer_neural_network(x, 𝐖, φ, 𝐠)
    end

    @info test_two_layer_neural_network([σ]) ≈ -0.013563772681566943
    @info test_two_layer_neural_network([ReLU]) ≈ -3.1715728752538093
    @info test_five_layer_neural_network([σ,σ,σ,σ]) ≈ -3.1668639943749355e-7
end

# ╔═╡ Cell order:
# ╟─dbd0ccc0-fea3-11ea-359e-876ee15179be
# ╠═0b7610be-fea4-11ea-0a69-eff0a019b6e6
# ╟─047d3eb2-fea4-11ea-3ff7-0b15c54073e2
# ╠═08c79290-fea4-11ea-0a77-e5f5e6ab401d
# ╠═c8f6b2c0-ffd1-11ea-3972-25233fa64f94
# ╠═1384e6b0-fea4-11ea-2e03-cdad79f986d5
# ╠═ae23a200-ffd1-11ea-3cfe-bb895cf90087
# ╠═be738170-ffd1-11ea-23ec-45adb917e9c6
# ╟─233ec7b0-fea4-11ea-3a47-a3125ec028be
# ╟─a13c0ba0-fea4-11ea-0fa7-cde0a3840c99
# ╠═11ab44b0-fea4-11ea-042f-a3547f3e2a8b
# ╟─12cfa070-fea4-11ea-0963-9548b2c07c44
# ╠═3c132240-fea4-11ea-0b96-8fac4ceb5841
# ╠═32b06dc0-fef4-11ea-25f2-cdaeafd52ab4
# ╟─a43062c0-0020-11eb-245e-cf6b2036b878
# ╠═b17c5100-0020-11eb-2928-81d80731960e
# ╟─c43d5c20-ff9f-11ea-15ce-3bef26583d2a
# ╟─492db7ae-ff9e-11ea-38a0-09cead1c5ae6
# ╟─e7ffb7c0-ff9f-11ea-3c0e-cbb878af2c6e
# ╟─180dc51e-ff9f-11ea-3ad0-a51012f662b0
# ╟─07054ea0-ffa0-11ea-1fd1-6f70c3600784
# ╟─d75e9920-ffa1-11ea-1bb0-6d0424b2b5b5
# ╟─b70c8f9e-ffa2-11ea-1f51-cd5660e3eb37
# ╟─0e09c0d0-ff9d-11ea-0e5e-37f227129043
# ╟─e3dd9740-ff9d-11ea-0d04-ad6f8be98423
# ╟─385750e0-ffa3-11ea-03d1-a7cc682d4ab2
# ╟─e2891c10-ffa3-11ea-05a6-f93d0bd8daf8
# ╟─66219680-ffa7-11ea-3528-9184d31f42de
# ╟─30cb2962-ffac-11ea-114f-73601457cef0
# ╟─c708c540-ffac-11ea-0632-35fdb0b7a4ef
# ╟─596a9110-ffae-11ea-2c89-7518df5b04c6
# ╠═3945e080-ffac-11ea-23a8-b968fe84d903
# ╠═35d0b6f0-ffac-11ea-27b1-47b519c13c21
# ╟─a2bcbdb0-ffa0-11ea-0a41-457fa1b70946
# ╟─68ca10f0-ff49-11ea-3147-09eb71687f20
# ╟─24692920-ffb0-11ea-3ab2-47b00719deed
# ╟─5a430f62-ffb1-11ea-2123-cd71772704ee
# ╠═3641cda0-ffbf-11ea-27a3-5fe417918cc0
# ╠═991f6770-ffbf-11ea-048c-1113d8640637
# ╠═7438c7c0-001a-11eb-1b09-b593715f3e82
# ╠═55eb5030-ffc0-11ea-0688-c1ecd12f1939
# ╠═98799310-ffd6-11ea-0714-17b135c4c704
# ╠═efce9b80-ffc0-11ea-2cb3-4f8209f29cdc
# ╠═10fb310e-ffc1-11ea-0d21-539799c2aebf
# ╠═3d66f4f0-ffc1-11ea-18ae-a5063f76ded5
# ╠═4016f9c0-ffc1-11ea-0638-8fc7e9f2c52b
# ╠═3c930220-ffcc-11ea-0694-e73b06bf7623
# ╠═6c1a4160-ffcd-11ea-249a-79972ec4eb85
# ╠═d8083910-ffc1-11ea-31f8-07d7fd4a0b9a
# ╠═1aaf8430-ffc2-11ea-0520-451f13594ddc
# ╠═b6d126c0-ffc2-11ea-2400-9bd611d2d8b7
# ╠═92237430-ffd7-11ea-3356-d9cd94e2a587
# ╠═a19c1100-ffd8-11ea-37d1-b3f2e0b6dc0f
# ╠═ea7a6f60-ffd9-11ea-395c-f769cabcc23f
# ╠═ba7f4af0-0020-11eb-2112-e13c9ebaec6c
# ╠═835e1c70-ffc3-11ea-31a5-214731bbc366
# ╠═0414b960-ffd7-11ea-313c-3fbce53d7a70
# ╟─24fd6bb0-0025-11eb-0489-5f1193474b18
# ╠═410ff860-ffc4-11ea-1d71-6141309e0112
# ╠═bd826122-ffc5-11ea-05de-9995d65c1b86
# ╠═0dee25e0-ffc6-11ea-0b20-3f4781d6ff83
# ╠═2504f1ee-ffc6-11ea-3b9a-d17dcf5d9b6e
# ╠═7af70d00-ffc6-11ea-1bd9-ddb90f5b04cd
# ╟─1870d340-ffcc-11ea-12c1-c7765b8a2957
# ╠═03a3e210-ffca-11ea-26e3-132e96817f3c
# ╠═1f400f30-ffca-11ea-3ef6-1b1722d88043
# ╟─859b8c80-ffc7-11ea-1590-2503626327ab
# ╟─89572a60-ffc6-11ea-2480-af31c91aac6c
# ╠═96a71ad0-ffc7-11ea-069d-1f4e1916d2ab
# ╟─41cda7ee-fea4-11ea-32d8-677bd3f7bc0b
# ╠═4533b650-fea4-11ea-329b-9363cdc0a214
# ╠═5d285f40-fea4-11ea-3295-b7fa7064eaae
