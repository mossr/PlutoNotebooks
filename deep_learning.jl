### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 69d289a0-0412-11eb-1279-37ad0d9e37f1
using Pkg

# ╔═╡ 6e1ae3f0-00ec-11eb-37d5-09ecad61da57
using LinearAlgebra, Statistics, Parameters, Random

# ╔═╡ 458fb9c0-01a9-11eb-35ab-1378e97278a6
using ScikitLearn

# ╔═╡ 80ba3550-0223-11eb-35f8-19eddfb38827
using TikzNeuralNetworks

# ╔═╡ 94f3e1d0-0212-11eb-0dc5-b162efd7d92a
using PyPlot; PyPlot.svg(true);

# ╔═╡ 59b99af2-00ec-11eb-0483-d79aacb643b3
md"""
# Deep learning
"""

# ╔═╡ 3f670290-0412-11eb-2004-39da4e0bf148
begin
    pkg"add Parameters"
    pkg"add ScikitLearn"
    pkg"add PyPlot"
    pkg"add https://github.com/mossr/TikzNeuralNetworks.jl"
end

# ╔═╡ 87728350-0220-11eb-2813-33d3a34d2cc4
md"## Element-wise multiplication"

# ╔═╡ 66636610-014a-11eb-1558-6f141275a045
X ⊙ Y = X .* Y # element-wise multiplication

# ╔═╡ 00a3e060-01aa-11eb-2572-731449361108
md"## Dataset"

# ╔═╡ ff4ded50-01a9-11eb-010b-314042bbe356
@sk_import datasets: make_moons;

# ╔═╡ 88b054d0-0852-11eb-01b4-b1c449fb0a2b
Random.seed!(0);

# ╔═╡ 08feda80-01aa-11eb-2134-936bdef78373
begin
	X, Y = make_moons(n_samples=200, noise=0.2)
	X, Y = X', reshape(Y, 1, size(Y)[1])
end;

# ╔═╡ 37842f90-01aa-11eb-2f89-2d1d748a4cf1
md"## Neural network structure"

# ╔═╡ 2ba1e370-01aa-11eb-1b83-433d7fadf826
md"""
## Activation functions

Activation functions $a = g(z)$.

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

# ╔═╡ 53b5180e-01a9-11eb-26c4-678fe66ad72e
begin
	ReLU(z) = max(z, 0)
	ReLU′(z) = z < 0 ? 0 : 1
	σ(z) = 1/(1 + exp(-z))
	σ′(z) = σ(z)*(1 - σ(z))
	tanh′(z) = 1 - tanh(z)^2
	gaussian(z) = exp(-z^2)
	gaussian′(z) = -2z*gaussian(z)
	sinc(z) = z == 0 ? 1 : sin(z)/z
	sinc′(z) = z == 0 ? 0 : cos(z)/z - sin(z)/z^2
end

# ╔═╡ 48077080-01a9-11eb-1746-37a91cdeda9b
@with_kw mutable struct NeuralNetwork
    input_size::Int = 2                         # input layer dimension
    hidden_layer_sizes::Vector{Int} = [5, 4, 1] # hidden layer dimensions
    output_size::Int = 1                        # output layer dimension
    L::Int = length(hidden_layer_sizes)         # number of hidden layers
    𝐠::Vector{Function} = [tanh, tanh, σ]       # activation functions
    𝐠′::Vector{Function} = [tanh′, tanh′, σ′]   # activation function derivatives
	λ::Real = 0                                 # L₂ regularization parameter

	# cached parameters, forward values, and gradients
	𝐖 = Vector(undef, L)    # weights
    𝐛 = Vector(undef, L)    # biases
    𝐙 = Vector(undef, L)    # linear transformations
    𝐀 = Vector(undef, L)    # activations
    d𝐖 = Vector(undef, L)   # weight gradients
    d𝐛 = Vector(undef, L)   # bias gradients
    d𝐙 = Vector(undef, L)   # linear transformation gradients
    d𝐀 = Vector(undef, L+1) # activation gradients
end

# ╔═╡ 646c4e60-01b5-11eb-2e73-bf1da0ecb35a
function plot_nn(nn::NeuralNetwork)
	nn2 = TikzNeuralNetwork(input_size=nn.input_size,
		input_label=i->"\$x_{$i}\$",
		input_arrows=false,
		hidden_layer_sizes=nn.hidden_layer_sizes,
		activation_functions=[L"g", L"g", L"\sigma"], # TODO
		hidden_layer_labels=[L"\tanh", L"\tanh", "logistic"], # TODO
		output_label=i->L"\hat{y}",
		output_arrows=false,
		output_size=nn.output_size)
	try
		nn2.tikz.width="12cm" # Requires newest TikzPictures
	catch
		# Install newest TikzPicture
	end
	return nn2
end

# ╔═╡ e54aafae-01a9-11eb-3561-918c493bce9c
nn = NeuralNetwork();

# ╔═╡ 6848ec50-01b5-11eb-0525-d1e80de80fc9
plot_nn(nn)

# ╔═╡ 42747a90-01aa-11eb-3d29-f38841bea8e0
md"""
## Parameter initialization
Initialize weights $\mathbf{W}$ to small, non-zero values, and biases $\mathbf{b}$ to zero.

$$\mathbf{W}^{[\ell]} = (n^{[\ell]}, n^{[\ell-1]}) = (\text{dim. of next layer, dim. of previous layer})$$
"""

# ╔═╡ ef316cc0-0213-11eb-2b93-83b03d655b7b
function initialize!(nn, X, Y)
	(nn.input_size, nn.output_size) = (size(X)[1], size(Y)[1])
	nn.L = length(nn.hidden_layer_sizes)
	nn.𝐖 = Vector(undef, nn.L)
	nn.𝐛 = Vector(undef, nn.L)
	nn.𝐙 = Vector(undef, nn.L)
	nn.𝐀 = Vector(undef, nn.L)
	nn.d𝐖 = Vector(undef, nn.L)
	nn.d𝐛 = Vector(undef, nn.L)
	nn.d𝐙 = Vector(undef, nn.L)
	nn.d𝐀 = Vector(undef, nn.L+1)
	nₕ₋₁ = nn.input_size
	for (ℓ,nₕ) in enumerate(nn.hidden_layer_sizes)
		nn.𝐖[ℓ] = 0.01randn(nₕ, nₕ₋₁)
		nn.𝐛[ℓ] = zeros(nₕ, 1)
		nₕ₋₁ = nₕ
	end
end

# ╔═╡ 706883c0-01b8-11eb-3166-0b7cd9be5cdd
md"""
## Forward propagation
Pass the inputs $\mathbf{X}=\mathbf{A}^{[0]}$ throught the neural networks linear transform and activation functions.

$$\begin{gather}
\mathbf{Z}^{[\ell]} = \mathbf{W}^{[\ell]} \mathbf{A}^{[\ell-1]} + \mathbf{b}^{[\ell]}\\
\mathbf{A}^{[\ell]} = g^{[\ell]}(\mathbf{Z}^{[\ell]})
\end{gather}$$
"""

# ╔═╡ 4d8daa50-01aa-11eb-1a85-fd7e9e9a292e
function forwardpass!(nn, 𝐗)
	(𝐖, 𝐛, 𝐙, 𝐠) = (nn.𝐖, nn.𝐛, nn.𝐙, nn.𝐠)
	𝐀 = 𝐗
	for (ℓ,g) in enumerate(𝐠)
		𝐙[ℓ] = 𝐖[ℓ]*𝐀 .+ 𝐛[ℓ]
		nn.𝐀[ℓ] = 𝐀
		𝐀 = g.(𝐙[ℓ])
	end	
	return 𝐀
end

# ╔═╡ 0ff4b470-01ac-11eb-2503-5d903ea6af7d
md"""
## Cost function
Here we use the cross-entropy loss, or log-loss $\mathcal{L}$.
"""

# ╔═╡ 881bad50-01ac-11eb-0a28-21a990a1f233
md"$$\mathcal{L}(\hat{y},y) = -\bigl(y\log(\hat{y}) + (1-y)\log(1-\hat{y})\bigr)$$"

# ╔═╡ 853d0d20-0159-11eb-0947-c54b0846fbc2
loss(a, y) = -(y * log(a) + (1 - y) * log(1 - a))

# ╔═╡ c3d7baf2-01ac-11eb-002a-ad7aba3a0e82
md"$$\nabla\mathcal{L}(\hat{y},y)=-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$"

# ╔═╡ 0514c480-01a9-11eb-3ea8-915a020f5ca3
∇loss(a, y) = -(y / a) + (1 - y)/(1 - a)

# ╔═╡ 8c3b9730-06cd-11eb-19f0-dfc5552fcecf
md"""
#### $L_2$ Regularization

$$\frac{\lambda}{2}\sum_{\ell=1}^{L} \lVert \mathbf{w}^{[\ell]} \rVert^2_F$$ 

Frobenius norm formula:

$$\lVert \mathbf w^{[\ell]} \rVert^2_F = \sum_{i=1}^{n^{[\ell]}} \sum_{j=1}^{n^{[\ell-1]}}\left(w_{i,j}^{[\ell]}\right)^2$$
"""

# ╔═╡ 4b18c92e-06cd-11eb-3e9f-f1b1cd2c7948
L₂(𝐖, λ) = λ/2 .* sum(𝐖 .^ 2)

# ╔═╡ efe1afc0-01ac-11eb-2dd9-33ab1132c6ca
md"""
The cost is defined as the mean loss given the prediction $\hat{y}$ and the target $y$.

$$\mathcal J(\mathbf{\hat Y}, \mathbf Y) = \underbrace{\frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})}_{\text{cost}} + \underbrace{\frac 1 m \sum_{i=1}^m \left( \frac{\lambda}{2}\sum_{\ell=1}^{L} \lVert \mathbf{w}^{[\ell]} \rVert^2_F \right)}_{L_2 \text{ regularization cost}}$$
"""

# ╔═╡ 3f2082b0-01ac-11eb-39cb-d70372d42349
cost(𝐀, 𝐘, 𝐖, λ) = mean(loss.(𝐀, 𝐘)) .+ mean(L₂.(𝐖, λ))

# ╔═╡ 799bd2f0-01ac-11eb-36bc-a91ee03ac43e
md"""
## Backpropagation

Backpropagate the loss gradient $\nabla\mathcal{L}$ through the network.

$$\begin{align}
d\mathbf{A}^{[L+1]} &= \nabla\mathcal{L}(\mathbf{A}^{[L+1]}, \mathbf{Y})\\
d\mathbf{Z}^{[L]} &= d\mathbf{A}^{[L+1]} \odot g^{\prime[L]}(\mathbf{Z}^{[L]})\\
d\mathbf{W}^{[L]} &= \frac{1}{m} d\mathbf{Z}^{[L]}\mathbf{A}^{[L]\top}\\
d\mathbf{b}^{[L]} &= \frac{1}{m} \sum d\mathbf{Z}^{[L]}\\
d\mathbf{A}^{[L]} &= \mathbf{W}^{[L]\top} d\mathbf{Z}^{[L]}\\
&\vdots
\end{align}$$
"""

# ╔═╡ ac546770-021a-11eb-1af1-15b10ebdd9e6
function backpropagation!(nn, 𝐀ₙ, 𝐘)
	(𝐖, 𝐙, 𝐀, 𝐠′, L) = (nn.𝐖, nn.𝐙, nn.𝐀, nn.𝐠′, nn.L)
	(d𝐖, d𝐛, d𝐀) = (nn.d𝐖, nn.d𝐛, nn.d𝐀)

	d𝐀[L+1] = ∇loss.(𝐀ₙ, 𝐘)

	for ℓ in reverse(1:L)
		m = size(𝐀[ℓ])[2]
		d𝐙 = d𝐀[ℓ+1] ⊙ 𝐠′[ℓ].(𝐙[ℓ])
		d𝐖[ℓ] = 1/m * d𝐙*𝐀[ℓ]'
		d𝐛[ℓ] = 1/m * sum(d𝐙; dims=2)
		d𝐀[ℓ] = 𝐖[ℓ]'d𝐙
	end
end

# ╔═╡ 74037e72-01b0-11eb-335d-4d0ca0f87525
md"""
## Gradient descent parameter update

Update the weights $\mathbf{W}$ and biases $\mathbf{b}$ using gradient descent with a learning rate $\alpha$.

$$\begin{gather}
\mathbf{W}^{[\ell]} \leftarrow \mathbf{W}^{[\ell]} - \alpha (d\mathbf{W}^{[\ell]})\\
\mathbf{b}^{[\ell]} \leftarrow \mathbf{b}^{[\ell]} - \alpha (d\mathbf{b}^{[\ell]})
\end{gather}$$
"""

# ╔═╡ c976fc60-06ce-11eb-3df3-bbc798190682
md"""
Include $L_2$ regularization term in gradient update.

$$\mathbf W^{[\ell]} \leftarrow \mathbf W^{[\ell]} - \alpha \left[(d\mathbf W^{[\ell]}) + \frac \lambda m \mathbf W^{[\ell]} \right]$$
"""

# ╔═╡ 7a92b940-01b0-11eb-2813-35fb1d825be1
function update!(nn, m, α=1.2)
	(𝐖, 𝐛, d𝐖, d𝐛, λ, L) = (nn.𝐖, nn.𝐛, nn.d𝐖, nn.d𝐛, nn.λ, nn.L)
	for ℓ in 1:L
		𝐖[ℓ] = 𝐖[ℓ] .- α*(d𝐖[ℓ] + λ/m .* 𝐖[ℓ])
		𝐛[ℓ] = 𝐛[ℓ] .- α*d𝐛[ℓ]
	end
end

# ╔═╡ dfc566a0-01b0-11eb-3ac4-1fda0111c068
md"""
## Training the model
Initialize weights $\mathbf{W}$ and biases $\mathbf{b}$.
- Forward pass given inputs $\mathbf{X}$
- Backpropagation using outputs $\mathbf{A}_N = \hat{\mathbf{Y}}$
- Gradient descent update of parameters $\mathbf{W}$ and $\mathbf{b}$
- Repeat.
"""

# ╔═╡ e338e280-01b0-11eb-01a9-2745d7e41122
function train!(nn, 𝐗, 𝐘; α=1.2, niters=10000)
	initialize!(nn, 𝐗, 𝐘)
	m = size(X)[2]

	for i in 0:niters
		𝐀ₙ = forwardpass!(nn, 𝐗)
		backpropagation!(nn, 𝐀ₙ, 𝐘)
		update!(nn, m, α)
		(i % 1000 == 0) && @show i, cost(𝐀ₙ, 𝐘, nn.𝐖, nn.λ)
	end

	return nn
end

# ╔═╡ f13757e0-01b0-11eb-0707-6f7238bab586
begin
	global trigger = nothing
	train!(nn, X, Y)
end;

# ╔═╡ a2eee850-01b0-11eb-1416-e1271f103f51
md"""
## Binary classification predictions
$$\text{prediction} = \begin{cases}
1 & \text{if }\hat{y} > 0.5\\
0 & \text{otherwise}
\end{cases}$$
"""

# ╔═╡ b307d940-01b0-11eb-0db1-5f2906ce95f2
function predict(nn, 𝐗)
	Aₙ = forwardpass!(nn, 𝐗)
	return Aₙ .> 0.5
end

# ╔═╡ c148fcf0-01b0-11eb-3f93-739577af860a
md"""
## Accuracy
$$\frac{\text{number of correct predictions}}{\text{total number of examples}}$$ 
"""

# ╔═╡ fd0962d0-021d-11eb-17be-959af87c36b2
function accuracy(nn, X, Y)
	predictions = predict(nn, X)
	return sum(predictions .== Y) / length(Y) * 100
end

# ╔═╡ c40eaca0-01b0-11eb-14e4-fd23dbec7861
begin
	trigger
	md"""
	##### Model accuracy = $(accuracy(nn, X, Y))%
	"""
end

# ╔═╡ 713882b0-0220-11eb-34c5-a39efd8da5d5
md"## Decision boundary"

# ╔═╡ d50fdab0-01b0-11eb-34a4-296308516a67
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

# ╔═╡ cf75fc60-01b0-11eb-19a4-f926e08ee377
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
	clf()
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[1,:], X[2,:], c=y, cmap=plt.cm.Spectral)
	gcf()
end

# ╔═╡ d7903770-01c5-11eb-3bac-91b08b469965
model = x->predict(nn, x)

# ╔═╡ 99a64d50-01c5-11eb-2230-b708a23c89ac
plot_decision_boundary(model, X, Y)

# ╔═╡ 5691b5a0-026e-11eb-09f7-b737c512df06
md"## Circle classification"

# ╔═╡ 4997b7f0-026e-11eb-1b1d-b1af0f65c5e3
@sk_import datasets: make_circles;

# ╔═╡ 55759740-026e-11eb-3cbd-9d458bb360bf
begin
	Xc, Yc = make_circles(n_samples=200, noise=0.2)
	Xc, Yc = Xc', reshape(Yc, 1, size(Yc)[1])
end;

# ╔═╡ b214d560-026e-11eb-0fea-f55b1d2b6299
begin
	Random.seed!(0)
	nnc = NeuralNetwork()
	nnc.hidden_layer_sizes = [60, 40, 20, 1]
	nnc.𝐠 = [ReLU, ReLU, ReLU, σ]
	nnc.𝐠′ = [ReLU′, ReLU′, ReLU′, σ′]
	nnc.λ = 0.005 # L₂ regularization
	trigger2 = nothing

	train!(nnc, Xc, Yc; α=1.2, niters=10_000)
end;

# ╔═╡ a43e2a10-0852-11eb-28e3-3fb4809559ba
begin
	trigger2
	md"""
	##### Model accuracy = $(accuracy(nnc, Xc, Yc))%
	"""
end

# ╔═╡ 9075a7e0-026e-11eb-1c5d-715fbe95032f
plot_decision_boundary(x->predict(nnc, x), Xc, Yc)

# ╔═╡ Cell order:
# ╟─59b99af2-00ec-11eb-0483-d79aacb643b3
# ╠═69d289a0-0412-11eb-1279-37ad0d9e37f1
# ╠═3f670290-0412-11eb-2004-39da4e0bf148
# ╠═6e1ae3f0-00ec-11eb-37d5-09ecad61da57
# ╟─87728350-0220-11eb-2813-33d3a34d2cc4
# ╠═66636610-014a-11eb-1558-6f141275a045
# ╟─00a3e060-01aa-11eb-2572-731449361108
# ╠═458fb9c0-01a9-11eb-35ab-1378e97278a6
# ╠═ff4ded50-01a9-11eb-010b-314042bbe356
# ╠═88b054d0-0852-11eb-01b4-b1c449fb0a2b
# ╠═08feda80-01aa-11eb-2134-936bdef78373
# ╟─37842f90-01aa-11eb-2f89-2d1d748a4cf1
# ╠═48077080-01a9-11eb-1746-37a91cdeda9b
# ╠═e54aafae-01a9-11eb-3561-918c493bce9c
# ╠═80ba3550-0223-11eb-35f8-19eddfb38827
# ╟─646c4e60-01b5-11eb-2e73-bf1da0ecb35a
# ╠═6848ec50-01b5-11eb-0525-d1e80de80fc9
# ╟─2ba1e370-01aa-11eb-1b83-433d7fadf826
# ╠═53b5180e-01a9-11eb-26c4-678fe66ad72e
# ╟─42747a90-01aa-11eb-3d29-f38841bea8e0
# ╠═ef316cc0-0213-11eb-2b93-83b03d655b7b
# ╟─706883c0-01b8-11eb-3166-0b7cd9be5cdd
# ╠═4d8daa50-01aa-11eb-1a85-fd7e9e9a292e
# ╟─0ff4b470-01ac-11eb-2503-5d903ea6af7d
# ╟─881bad50-01ac-11eb-0a28-21a990a1f233
# ╠═853d0d20-0159-11eb-0947-c54b0846fbc2
# ╟─c3d7baf2-01ac-11eb-002a-ad7aba3a0e82
# ╠═0514c480-01a9-11eb-3ea8-915a020f5ca3
# ╟─8c3b9730-06cd-11eb-19f0-dfc5552fcecf
# ╠═4b18c92e-06cd-11eb-3e9f-f1b1cd2c7948
# ╟─efe1afc0-01ac-11eb-2dd9-33ab1132c6ca
# ╠═3f2082b0-01ac-11eb-39cb-d70372d42349
# ╟─799bd2f0-01ac-11eb-36bc-a91ee03ac43e
# ╠═ac546770-021a-11eb-1af1-15b10ebdd9e6
# ╟─74037e72-01b0-11eb-335d-4d0ca0f87525
# ╟─c976fc60-06ce-11eb-3df3-bbc798190682
# ╠═7a92b940-01b0-11eb-2813-35fb1d825be1
# ╟─dfc566a0-01b0-11eb-3ac4-1fda0111c068
# ╠═e338e280-01b0-11eb-01a9-2745d7e41122
# ╠═f13757e0-01b0-11eb-0707-6f7238bab586
# ╟─a2eee850-01b0-11eb-1416-e1271f103f51
# ╠═b307d940-01b0-11eb-0db1-5f2906ce95f2
# ╟─c148fcf0-01b0-11eb-3f93-739577af860a
# ╟─c40eaca0-01b0-11eb-14e4-fd23dbec7861
# ╠═fd0962d0-021d-11eb-17be-959af87c36b2
# ╟─713882b0-0220-11eb-34c5-a39efd8da5d5
# ╠═94f3e1d0-0212-11eb-0dc5-b162efd7d92a
# ╟─d50fdab0-01b0-11eb-34a4-296308516a67
# ╟─cf75fc60-01b0-11eb-19a4-f926e08ee377
# ╠═d7903770-01c5-11eb-3bac-91b08b469965
# ╠═99a64d50-01c5-11eb-2230-b708a23c89ac
# ╟─5691b5a0-026e-11eb-09f7-b737c512df06
# ╠═4997b7f0-026e-11eb-1b1d-b1af0f65c5e3
# ╠═55759740-026e-11eb-3cbd-9d458bb360bf
# ╠═b214d560-026e-11eb-0fea-f55b1d2b6299
# ╟─a43e2a10-0852-11eb-28e3-3fb4809559ba
# ╠═9075a7e0-026e-11eb-1c5d-715fbe95032f
