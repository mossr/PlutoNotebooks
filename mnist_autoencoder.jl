### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ cf365570-054c-11eb-2e15-d117ada9248c
try using AddPackage catch; using Pkg; Pkg.add("AddPackage"); using AddPackage end

# ╔═╡ 1ff5f7d0-0553-11eb-13f4-4fa1c8cfa104
begin
	using Base.Iterators: partition
	@add using Colors
	@add using CUDA
	@add using Images
	@add using Parameters
	@add using Flux
	using Flux.Data.MNIST
	using Flux: @epochs, mse, throttle
	using Statistics
end

# ╔═╡ 5771c7e0-054c-11eb-3cb4-075289b865c0
md"# Autoencoder for MNIST"

# ╔═╡ 06ab3f50-0554-11eb-1eb0-d536ab6204cf
md"## Training arguments"

# ╔═╡ 57ede040-054d-11eb-1cf9-0b53979535b0
@with_kw mutable struct Args
	α::Float64 = 1e-3      # learning rate
	epochs::Int = 20       # number of epochs
	N::Int = 64            # size of the encoding (i.e. hidden layer)
	batchsize::Int = 1000  # batch size for training
	num::Int = 20          # number of random digits in the sample image (UNUSED)
	throttle::Int = 1      # throttle timeout (called once every X seconds)
end

# ╔═╡ 0f009420-0554-11eb-25c9-4be26cf068ae
md"## Dataset processing"

# ╔═╡ ce88f610-055a-11eb-1d29-b937c6b45dd6
global X = MNIST.images()

# ╔═╡ 91295d80-054d-11eb-2c53-99b31d4390eb
function get_processed_data(args)
	# load images and convert image of type RBG to Float
	imgs = channelview.(X)
	# partition into batches of size `batchsize`
	batches = partition(imgs, args.batchsize)
	traindata = [float(hcat(vec.(imgs)...)) for imgs in batches]
	return gpu.(traindata)
end

# ╔═╡ 158854e0-0554-11eb-15d3-93d4ab118d01
md"## Model training"

# ╔═╡ cad32cf0-054d-11eb-0da8-4bc1c7f2bffc
function train(; kwargs...)
	args = Args(; kwargs...)

	traindata = get_processed_data(args)

	@info "Constructing model..."

	# You can try to make the encoder/decoder network larger
	# Also, the output of encder is a coding of the given input.
	# In this case, the input dimension is 28^2 and the output dimension of the
	# encoder is 32. This implies that the coding is a compressed representation.
	# We can make lossy compression via this `encoder`.
	encoder = Dense(28^2, args.N, leakyrelu) |> gpu
	decoder = Dense(args.N, 28^2, leakyrelu) |> gpu

	# define main model as a Chain of encoder and decoder models
	m = Chain(encoder, decoder)

	@info "Training model..."
	loss(x) = mse(m(x), x)

	# callback, optimizer, and training
	callback = throttle(() -> @show(mean(loss.(traindata))), args.throttle)
	opt = ADAM(args.α)

	@epochs args.epochs Flux.train!(loss, params(m), zip(traindata), opt, cb=callback)

	return m, args, callback
end

# ╔═╡ 1ae3caa0-0554-11eb-372e-edb775d1dc94
md"## Sample from model"

# ╔═╡ 5e5b45c0-054e-11eb-053a-7718f1043426
img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

# ╔═╡ 6df36df0-054e-11eb-0b56-0d21d0f36305
function sample(m, num=1)
	# convert image of type RGB to Float
	imgs = channelview.(X)
	# number of random digits (truth)
	before = [imgs[i] for i in rand(1:length(imgs), num)]
	# after applying autoencoder to `before` input image
	after = img.(map(x->cpu(m)(float(vec(x))), before))
	# stack `before` and `after` images them all together
	Gray.(hcat(vcat.(before, after)...))
end

# ╔═╡ a9873310-054e-11eb-3d3e-6d39e2f065d2
m, args, cost = train();

# ╔═╡ 664c8f50-0553-11eb-0951-8b2516f026c1
Markdown.parse(string("\$\$\\frac{1}{m}\\sum_{i=1}^m\\operatorname{Loss}(\\mathbf{\\hat{x}}, \\mathbf{x}) = ", cost(), "\$\$"))

# ╔═╡ 5f088040-054f-11eb-0970-2f1c50b98cb5
sample(m, 20)

# ╔═╡ Cell order:
# ╟─5771c7e0-054c-11eb-3cb4-075289b865c0
# ╠═cf365570-054c-11eb-2e15-d117ada9248c
# ╠═1ff5f7d0-0553-11eb-13f4-4fa1c8cfa104
# ╟─06ab3f50-0554-11eb-1eb0-d536ab6204cf
# ╠═57ede040-054d-11eb-1cf9-0b53979535b0
# ╟─0f009420-0554-11eb-25c9-4be26cf068ae
# ╠═ce88f610-055a-11eb-1d29-b937c6b45dd6
# ╠═91295d80-054d-11eb-2c53-99b31d4390eb
# ╟─158854e0-0554-11eb-15d3-93d4ab118d01
# ╠═cad32cf0-054d-11eb-0da8-4bc1c7f2bffc
# ╟─1ae3caa0-0554-11eb-372e-edb775d1dc94
# ╠═5e5b45c0-054e-11eb-053a-7718f1043426
# ╠═6df36df0-054e-11eb-0b56-0d21d0f36305
# ╠═a9873310-054e-11eb-3d3e-6d39e2f065d2
# ╟─664c8f50-0553-11eb-0951-8b2516f026c1
# ╠═5f088040-054f-11eb-0970-2f1c50b98cb5
