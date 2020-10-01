using Distributions
using Random
Random.seed!(0)

# θ₁ = Beta(1, 1)
# θ₂ = Beta(1, 1)


"""
Returns a real number value in the range `[0,1]` with
probability defined by a PDF of a Beta with parameters `a` and `b`.
"""
sampleBeta(a, b) = rand(Beta(a,b))


"""
Gives drug `i` to the next patient. Returns `true` if 
the grud was successful in curing the patient of `false`
if it was not. Throws an error if `i ∉ {1,2}`.
"""
function giveDrug(i)
    # return rand([true, false, false])
    return [true, false][i]
end

# Attempt at general Thompson sampling from the TS Tutorial (algorithm 4).
function thompson_sampling(𝒳, 𝒮, P, T, R; t_max=100) #, 𝛃::Vector{Beta}, 𝛉::Vector=ones(length()))
    for t in 1:t_max
        # sample model
        θ = rand(P)

        # select and apply action
        xₜ = argmax([sum(T(s, x, θ)*R(s) for s in 𝒮) for x in 𝒳])
        sₜ = apply(xₜ)

        # update distribution
        P = P(θ)*T(sₜ, xₜ) / sum(P(ν)*T(ν, xₜ))
    end

    return P
end

X = [1, 1, 1, -1, -1, 1, 1, 1]
S = [1, 2, 3,  2,  1, 2, 3, 4]
P = Beta()
thompson_sampling()






### Huffman coding
function tree_print(N, v="")
    S = ""
    if N isa Char
        return v*"-"
    else
        S *= tree_print(N.left, v*"0")
        S *= tree_print(N.right, v*"1")
    end
    return S
end

# Not multiple dispatched.
function encode(x, N, v=0x0)
    if N isa Char
        return N == x ? v : 0x0
    else
        return encode(x, N.left, v<<1) | encode(x, N.right, v<<1+0x1)
    end
end

function encode(x, N, v="")
    if N isa Char
        return N == x ? v : ""
    else
        return encode(x, N.left, v*"") * encode(x, N.right, v*"1")
    end
end


encode(x, N::Char, v) = N == x ? v : 0b0
encode(x, N, v=0b0) = encode(x, N.left, v<<1) | encode(x, N.right, v<<1+0b1)


encode(s::String) = join(map(c->encode(c, N), collect(s)))
encode(c::Char, N, v="") = encode(c, N.left, v*'0') * encode(c, N.right, v*'1')
encode(c::Char, N::Char, v) = N == c ? v : ""

function decode(S::String, N, v="", next=N)
    for s in S
        next = (next isa Char) ? N : next
        next = (s == '0') ? next.left : next.right
        v *= (next isa Char) ? next : ""
    end
    return v
end


# Loss functions

∇trainloss(𝐰, 𝒟train, φ, ∇loss) = mean(∇loss(x, y, 𝐰, φ) for (x,y) ∈ 𝒟train)

#=
binary_classifier(x, 𝐰, φ) = ŷ(x, 𝐰, φ) ≥ 0 ? +1 : -1


loss_01(x, y, 𝐰, φ)       = 𝕀(margin(x, y, 𝐰, φ) ≤ 0)
loss_squared(x, y, 𝐰, φ)  = residual(x, y, 𝐰, φ)^2
loss_absdev(x, y, 𝐰, φ)   = abs(residual(x, y, 𝐰, φ))
loss_hinge(x, y, 𝐰, φ)    = max(1 - margin(x, y, 𝐰, φ), 0)
loss_logistic(x, y, 𝐰, φ) = log(1 + exp(-margin(x, y, 𝐰, φ)))
loss_cross_entropy(x, y, 𝐰, φ; ŷ=prediction(x, 𝐰, φ)) = -(y*log(ŷ) + (1-y)*log(1-ŷ))

∇loss_squared(x, y, 𝐰, φ) = 2residual(x, y, 𝐰, φ)*φ(x)
∇loss_hinge(x, y, 𝐰, φ)   = margin(x, y, 𝐰, φ) < 1 ? -φ(x)*y : 0

function gradient_descent(𝒟train, φ, ∇trainloss; η=0.1, T=100)
    𝐰 = init_weights(length(φ(𝒟train[1][1])))
    for t in 1:T
        𝐰 = 𝐰 - η*∇trainloss(𝐰, 𝒟train, φ)
    end
    return 𝐰
end


function stochastic_gradient_descent(𝒟train, φ, ∇loss; η=0.1, T=100)
    𝐰 = init_weights(length(φ(𝒟train[1][1])))
    for t in 1:T
        for (x,y) ∈ 𝒟train
            𝐰 = 𝐰 - η*∇loss(x, y, 𝐰, φ)
        end
    end
    return 𝐰
end
=#

# WRONG.
# margin(x, y, 𝐰, φ) < 1 ? -φ(x)*y : φ(x)*y




# One-line two-layer neural network
neural_network(x, 𝐕, 𝐰, φ, g) = 𝐰 ⋅ map(𝐯ⱼ -> g(𝐯ⱼ ⋅ φ(x)), 𝐕)


# From NN notebook

TikzPicture("""
\\tikzstyle{every pin edge}=[<-,shorten <=1pt]
\\tikzstyle{neuron}=[circle,fill=black!25,minimum size=17pt,inner sep=0pt]
\\tikzstyle{input neuron}=[neuron, fill=green!50];
\\tikzstyle{output neuron}=[neuron, fill=red!50];
\\tikzstyle{hidden neuron}=[neuron, fill=blue!50];
\\tikzstyle{annot} = [text width=4em, text centered]

% Draw the input layer nodes
\\foreach \\name / \\y in {1,...,3}
    \\node[input neuron, pin=left:Input \\#\\y] (I-\\name) at (0,-\\y) {};

% Draw the hidden layer nodes
\\foreach \\name / \\y in {1,...,5}
    \\path[yshift=0.5cm]
        node[hidden neuron] (H-\\name) at (\\layersep,-\\y cm) {};

% Draw the output layer node
\\node[output neuron,pin={[pin edge={->}]right:Output}, right of=H-3] (O) {};

% Connect every node in the input layer with every node in the hidden layer.
\\foreach \\source in {1,...,3}
    \\foreach \\dest in {1,...,5}
        \\path (I-\\source) edge (H-\\dest);

% Connect every node in the hidden layer with the output layer
\\foreach \\source in {1,...,5}
    \\path (H-\\source) edge (O);

% Annotate the layers
\\node[annot,above of=H-1, node distance=1cm] (hl) {Hidden layer};
\\node[annot,left of=hl] {Input layer};
\\node[annot,right of=hl] {Output layer};
""",
options="shorten >=1pt,->,draw=black!50, node distance=\\layersep",
preamble="\\def\\layersep{2.5cm}")


using TikzGraphs, LightGraphs, MetaGraphs

begin
    g = DiGraph(6)
    add_edge!(g, 1, 4)
    add_edge!(g, 1, 5)
    add_edge!(g, 2, 4)
    add_edge!(g, 2, 5)
    add_edge!(g, 3, 4)
    add_edge!(g, 3, 5)
    add_edge!(g, 4, 6)
    add_edge!(g, 5, 6)
end

graph = TikzGraphs.plot(g, Layouts.Layered(), node_style="circle, draw=black, fill=white, minimum size=16pt", options="grow'=right, level distance=15mm, sibling distance=2mm, semithick, >=stealth'");

graph.width="12cm"; graph

gnn = TikzGraphs.plot(g, Layouts.Layered(), ["", "", "", L"\sigma", L"\sigma", ""], node_style="circle, draw=black, fill=white, minimum size=16pt",
node_styles=Dict(1=>"label=left:{\$\\phi(x)_1\$}",2=>"label=left:{\$\\phi(x)_2\$}", 3=>"label=left:{\$\\phi(x)_3\$}", 4=>"fill=lightgray!70", 5=>"fill=lightgray!70"),
options="grow'=right, level distance=15mm, sibling distance=2mm, semithick, >=stealth'");

gnn.width="12cm"; gnn



        dW2 = 1/m * dZ2*A1'
        db2 = 1/m * sum(dZ2)
        dZ1 = W2'*dZ2 .* (1 .- A1.^2)
        dW1 = 1/m * dZ1*X'
        db1 = 1/m * sum(dZ1)

    nn.grads["dW1"] = dW1
    nn.grads["db1"] = db1
    nn.grads["dW2"] = dW2
    nn.grads["db2"] = db2



function backpropagation!(nn, X, Y)
    m = size(X)[2]

    Aj = nn.forward["A$(length(nn.activation_functions)-1)"]
    Wj = nn.θ["W$(length(nn.activation_functions)-1)"]
    local dZj
    for i in length(nn.activation_functions):-1:1
        Wi = nn.θ["W$i"]
        Ai = nn.forward["A$i"]
        
        if i == length(nn.activation_functions)
            dZi = Ai .- Y # assumes binary classification with sigmoid
        else
            dZi = Wj' * dZj .* nn.activation_functions′.(Aj)
        end
        dWi = 1/m * dZi*Aj'
        dbi = 1/m * sum(dZi)
        Aj = i == 2 ? X : Ai
        dZj = dZi
    
        nn.grads["dW$i"] = dWi
        nn.grads["db$i"] = dbi
    end
end


# manual backpropagation
    dZ2 = A2 .- Y # assumes binary classification with sigmoid
    dW2 = 1/m * dZ2*A1'
    db2 = 1/m * sum(dZ2)
    dZ1 = W2'*dZ2 .* (1 .- A1.^2)
    dW1 = 1/m * dZ1*X'
    db1 = 1/m * sum(dZ1)
    
    nn.grads["dW1"] = dW1
    nn.grads["db1"] = db1
    nn.grads["dW2"] = dW2
    nn.grads["db2"] = db2

function backpropagation!(nn, X, Y)
    m = size(X)[2]

    W1 = nn.θ["W1"]
    W2 = nn.θ["W2"]
    
    A1 = nn.forward["A1"]
    A2 = nn.forward["A2"]
    
    # for each activation, get the gradient
    # dW2 = first(gradient(a->loss(a, Y), W2))

    # TODO: Generalize.
    dZ2 = A2 .- Y # assumes binary classification with sigmoid
    dW2 = 1/m * dZ2*A1'
    db2 = 1/m * sum(dZ2)
    dZ1 = W2'*dZ2 .* (1 .- A1.^2)
    dW1 = 1/m * dZ1*X'
    db1 = 1/m * sum(dZ1)
    
    nn.grads["dW1"] = dW1
    nn.grads["db1"] = db1
    nn.grads["dW2"] = dW2
    nn.grads["db2"] = db2
end


function backpropagation!(nn, AL, Y)
    L = length(nn.hidden_layer_sizes)
    
    dA = ∇loss.(AL, Y)
    
    for l in reverse(0:L-1)
        A, Z = nn.forward["A$(l+1)"], nn.forward["Z$(l+1)"]
        W, b = nn.θ["W$(l+1)"], nn.θ["b$(l+1)"]
        m = size(A)[2]
        g′ = nn.activation_functions′[l+1]
        dZ = dA ⊙ g′.(Z)
        dA = W' * dZ
        dW = 1/m * dZ*A'
        db = 1/m * sum(dZ)
        nn.grads["dA$l"] = dA
        nn.grads["dW$l"] = dW
        nn.grads["db$l"] = db
    end
end

function backpropagation!(nn, AL, Y)
    L = length(nn.hidden_layer_sizes)
    
    A = nn.forward["A$L"]
    Z = nn.forward["Z$L"]
    m = size(A)[2]
    dA = ∇loss.(A, Y) # TODO: AL
    dZ = nn.activation_functions′[L].(dA)
    dW = 1/m * dZ*A'
    db = 1/m * sum(dZ)
    
    for l in reverse(1:L-1)
        m = size(A)[2]
        dW = 1/m * dZ*A'
        db = 1/m * sum(dZ)
        g′ = nn.activation_functions′[l]
        dZ = dA ⊙ g′.(Z)
        nn.grads["dA$l"] = dA
        nn.grads["dW$(l+1)"] = dW
        nn.grads["db$(l+1)"] = db
        A, Z = nn.forward["A$l"], nn.forward["Z$l"]
        W = nn.θ["W$(l+1)"]
        dA = W' * dZ
    end
end


## WORKS----with Coursera-style breakup of functions
function backpropagation!(nn, AL, Y)
    L = length(nn.hidden_layer_sizes)
    
    dAL = ∇loss.(AL, Y)
    A_prev, W, b = nn.forward["A$L"], nn.θ["W$L"], nn.θ["b$L"]
    Z = nn.forward["Z$L"]
    nn.grads["dA$L"], nn.grads["dW$L"], nn.grads["db$L"] = 
        linear_activation_backward(dAL, Z, A_prev, W, b, nn.activation_functions′[L])
    
    for l in reverse(0:L-2)     
        Z = nn.forward["Z$(l+1)"]
        A_prev, W, b = nn.forward["A$(l+1)"], nn.θ["W$(l+1)"], nn.θ["b$(l+1)"]
        dA_prev_tmp, dW_tmp, db_tmp = 
            linear_activation_backward(nn.grads["dA$(l+2)"], Z, A_prev, W, b,
                                       nn.activation_functions′[l+1])
        nn.grads["dA$(l+1)"] = dA_prev_tmp
        nn.grads["dW$(l+1)"] = dW_tmp
        nn.grads["db$(l+1)"] = db_tmp
        # m = size(A)[2]
        # dW = 1/m * dZ*A'
        # db = 1/m * sum(dZ)
        # g′ = nn.activation_functions′[l]
        # dZ = dA ⊙ g′.(Z)
        # nn.grads["dA$l"] = dA
        # nn.grads["dW$(l+1)"] = dW
        # nn.grads["db$(l+1)"] = db
        # A, Z = nn.forward["A$l"], nn.forward["Z$l"]
        # W = nn.θ["W$(l+1)"]
        # dA = W' * dZ
    end
end

function linear_backward(dZ, A_prev, W, b)
    m = size(A_prev)[2]
    dW = 1/m * dZ*A_prev'
    db = 1/m * sum(dZ; dims=2)
    dA_prev = W' * dZ
    
    return dA_prev, dW, db
end

function linear_activation_backward(dA, Z, A_prev, W, b, g′)
    dZ = dA ⊙ g′.(Z)
    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db
end


function backpropagation!(nn, 𝐀L, 𝐘)
    L = length(nn.hidden_layer_sizes)

    nn.d𝐀[L+1] = ∇loss.(𝐀L, 𝐘)

    for l in reverse(1:L)
        (𝐖, 𝐛) = nn.𝐖[l], nn.𝐛[l]
        (𝐀, 𝐙, d𝐀) = nn.𝐀[l], nn.𝐙[l], nn.d𝐀[l+1]
        g′ = nn.𝐠′[l]

        m = size(𝐀)[2]
        d𝐙 = d𝐀 ⊙ g′.(𝐙)
        d𝐖 = 1/m * d𝐙*𝐀'
        d𝐛 = 1/m * sum(d𝐙; dims=2)
        d𝐀 = 𝐖'd𝐙

        nn.d𝐀[l], nn.d𝐖[l], nn.d𝐛[l] = d𝐀, d𝐖, d𝐛
    end
end

function backpropagation!(nn, 𝐀, 𝐘)
    L = length(nn.hidden_layer_sizes)

    nn.d𝐀[L+1] = ∇loss.(𝐀, 𝐘)

    for l in reverse(1:L)
        m = size(𝐀)[2]
        d𝐙 = nn.d𝐀[l+1] ⊙ nn.𝐠′[l].(nn.𝐙[l])
        d𝐖 = 1/m * d𝐙*nn.𝐀[l]'
        d𝐛 = 1/m * sum(d𝐙; dims=2)
        d𝐀 = nn.𝐖[l]'d𝐙

        nn.d𝐀[l], nn.d𝐖[l], nn.d𝐛[l] = d𝐀, d𝐖, d𝐛
    end
end


function forwardpass!(nn, 𝐗)
    (𝐖, 𝐛, 𝐙, 𝐀, 𝐠) = (nn.𝐖, nn.𝐛, nn.𝐙, nn.𝐀, nn.𝐠)
    𝐀[1] = 𝐗
    for (l,g) in enumerate(𝐠)
        𝐙[l] = 𝐖[l]*𝐀[l] .+ 𝐛[l] # linear forward
        𝐀[l+1] = g.(𝐙[l]) # apply activation
    end 
    return 𝐀[end-1]
end

function cost(nn::NeuralNetwork, 𝐘)
    N = length(nn.𝐠)
    𝐀ₙ = nn.𝐀[N]
    return cost(𝐀ₙ, 𝐘)
end


md"""
## Backpropagation

$$\begin{align}
d\mathbf{Z}^{[L]} &= \mathbf{A}^{[L]} - \mathbf{Y}\\
d\mathbf{W}^{[L]} &= \frac{1}{m} d\mathbf{Z}^{[L]}\mathbf{A}^{[L-1]\top}\\
d\mathbf{b}^{[L]} &= \frac{1}{m} \sum d\mathbf{Z}^{[L]}\\
d\mathbf{Z}^{[L-1]} &= \mathbf{W}^{[L]\top} d\mathbf{Z}^{[L]} \odot g^{\prime[L-1]}(\mathbf{Z}^{[L-1]})\\
&\vdots
\end{align}$$
"""