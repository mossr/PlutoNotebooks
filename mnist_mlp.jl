### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ c7356460-055b-11eb-0d69-dd42369dc913
try using AddPackage catch; using Pkg; Pkg.add("AddPackage"); using AddPackage end

# ╔═╡ df4bc620-055b-11eb-39c7-e952028567da
begin
	using Base.Iterators: repeated
	@add using Colors
	@add using CUDA
	@add using Images
	@add using MLDatasets
	@add using Parameters
	@add using Flux
	using Flux.Data: DataLoader
	using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, throttle
	using Statistics
end

# ╔═╡ d7a1f38e-055b-11eb-31ef-55de28cda23d
md"# Classifier for MNIST"

# ╔═╡ d8d88ce0-055d-11eb-32e8-31586d97a59c
md"## Training arguments"

# ╔═╡ 07f1cf70-055c-11eb-05ab-e77200ea8ea7
@with_kw mutable struct Args
	α::Float64 = 3e-4      # learning rate
	batchsize::Int = 1024  # batch size
	epochs::Int = 20       # number of epochs
	device::Function = gpu # set as gpu if available
	throttle::Int = 1      # throttle print every X seconds
end

# ╔═╡ f9bad300-055d-11eb-1e4d-f362cf07c6b5
md"## Data processing"

# ╔═╡ 37042240-055c-11eb-3eba-0da0d64627aa
function getdata(args)
	# load dataset
	xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
	xtest, ytest = MLDatasets.MNIST.testdata(Float32)

	# reshape data to flatten each image into a vector
	xtrain = Flux.flatten(xtrain)
	xtest = Flux.flatten(xtest)

	# one-hot encode the labels
	ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

	# split into batches
	traindata = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
	testdata = DataLoader(xtest, ytest, batchsize=args.batchsize)

	return traindata, testdata
end

# ╔═╡ ffa3bbb0-055d-11eb-2652-fd3b56dcf474
md"## Model"

# ╔═╡ 87cbe9b0-055c-11eb-1c34-9d9bedbe4ad5
function buildmodel(; imgsize=(28,28,1), nclasses=10)
	return Chain(
			Dense(prod(imgsize), 32, relu),
			Dense(32, nclasses))
end

# ╔═╡ 054402a2-055e-11eb-273e-1340d52bbed8
md"""
## Cost function
$$\begin{gather}
\mathcal{L}(\hat{y}, y) = -\frac{1}{n}\sum_{i=1}^n y \left(\hat{y} - \log\left(\sum e^{\hat{y}}\right)\right)\\
\mathcal{J}(\mathbf{\hat{y}}, \mathbf{y}) = \frac{1}{m}\sum \mathcal{L}(\hat{y}, y)
\end{gather}$$
"""

# ╔═╡ b391a990-055c-11eb-13b8-b3f9ad78cea7
cost(dataloader, model) = mean(logitcrossentropy(model(x), y) for (x,y) in dataloader)

# ╔═╡ ea7f0630-055e-11eb-0717-bde12a519dfc
md"## Accuracy"

# ╔═╡ db2160e0-055c-11eb-0c66-6b3cc667e18e
function accuracy(dataloader, model)
	acc = 0
	for (x,y) in dataloader
		acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y))) / size(x,2)
	end
	return acc/length(dataloader)
end

# ╔═╡ ef2b0210-055e-11eb-244b-d18b38d83a71
md"## Model training"

# ╔═╡ 01bccd20-055d-11eb-07a0-73fa53c45c24
function train(; kwargs...)
	# initialize model parameters
	args = Args(; kwargs...)

	# load data
	traindata, testdata = getdata(args)

	# construct model
	m = buildmodel()
	traindata = args.device.(traindata)
	testdata = args.device.(testdata)
	m = args.device(m)
	loss(x,y) = logitcrossentropy(m(x), y)

	# callback, optimizer, and training
	callback = throttle(()->@show(cost(traindata, m)), args.throttle)
	opt = ADAM(args.α)

	@epochs args.epochs Flux.train!(loss, params(m), traindata, opt, cb=callback)

	@show accuracy(traindata, m)
	@show accuracy(testdata, m)

	return m, traindata, testdata
end

# ╔═╡ 6092c160-055d-11eb-3364-0df28086629e
m, traindata, testdata = train();

# ╔═╡ f89b15b0-055e-11eb-3a07-091d5e96e340
begin
	Markdown.parse(string("\$\$\\operatorname{Loss}_\\text{train} = ", 100round(accuracy(traindata, m), digits=5), "\\%\$\$", "\$\$\\operatorname{Loss}_\\text{test} = ", 100round(accuracy(testdata, m), digits=5), "\\%\$\$"))
end

# ╔═╡ Cell order:
# ╟─d7a1f38e-055b-11eb-31ef-55de28cda23d
# ╠═c7356460-055b-11eb-0d69-dd42369dc913
# ╠═df4bc620-055b-11eb-39c7-e952028567da
# ╟─d8d88ce0-055d-11eb-32e8-31586d97a59c
# ╠═07f1cf70-055c-11eb-05ab-e77200ea8ea7
# ╟─f9bad300-055d-11eb-1e4d-f362cf07c6b5
# ╠═37042240-055c-11eb-3eba-0da0d64627aa
# ╟─ffa3bbb0-055d-11eb-2652-fd3b56dcf474
# ╠═87cbe9b0-055c-11eb-1c34-9d9bedbe4ad5
# ╟─054402a2-055e-11eb-273e-1340d52bbed8
# ╠═b391a990-055c-11eb-13b8-b3f9ad78cea7
# ╟─ea7f0630-055e-11eb-0717-bde12a519dfc
# ╠═db2160e0-055c-11eb-0c66-6b3cc667e18e
# ╟─ef2b0210-055e-11eb-244b-d18b38d83a71
# ╠═01bccd20-055d-11eb-07a0-73fa53c45c24
# ╠═6092c160-055d-11eb-3364-0df28086629e
# ╟─f89b15b0-055e-11eb-3a07-091d5e96e340
