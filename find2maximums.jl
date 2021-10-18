### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ b372e570-2fcf-11ec-0bc1-73431718551d
using Random, PlutoUI, Statistics

# ╔═╡ d58b2cfe-be5f-4dd9-bf57-10b8927936ae
md"""
# Find the second maximum
"""

# ╔═╡ 36190c75-ddea-4ba1-b0a3-0da601f6862f
md"""
## Divide-and-Concour
Divide-and-concour, baby! Runs in $O(n\log n)$ time.
"""

# ╔═╡ 6d901be0-6e00-411a-b585-c7260dd74246
function find2maximums(x)
	n = length(x)
	if n == 0
		return (-Inf, -Inf)
	elseif n == 1
		return (x[1], -Inf)
	else
		# divide-and-concour
		m1, m12 = find2maximums(x[1:n÷2])
		m2, m22 = find2maximums(x[n÷2+1:end])
		
		if m1 > m2
			return m2 > m12 ? (m1, m2) : (m1, m12)
		else
			return m1 > m22 ? (m2, m1) : (m2, m22)
		end
	end
end

# ╔═╡ b2882ede-0bdb-4841-b801-65800643953e
with_terminal() do 
	@time for _ in 1:100
		x = rand(10_001)
		n = length(x)

		_m1 = maximum(x)
		_m2 = sort(x)[end-1]

		m1, m2 = find2maximums(x) # faster than `select`
		@assert m1 == _m1
		@assert m2 == _m2
	
		@assert select(x, n) == _m1
		@assert select(x, n-1) == _m2
	end
end

# ╔═╡ 5eabe62f-80a5-4d33-a27d-7a56b91ca060
md"""
## QuickSelect
"""

# ╔═╡ 8e9f512a-eb5c-46ac-829d-50cd789c5f0a
median_of_medians(x) = partialsort(x, (length(x)+1)÷2)

# ╔═╡ 2ebc9063-07f0-456c-afc7-7e9ff18e5dd1
function partition(x, pivot)
	L, R = [], []
	for i in 1:length(x) # O(n)
		if x[i] == pivot
			continue
		elseif x[i] < pivot
			push!(L, x[i])
		else
			push!(R, x[i])
		end
	end
	return L, R
end

# ╔═╡ 074a179c-6e63-4bce-a0ef-845d8351c99b
function select(x, k)
	n = length(x)
	if n == 1
		return x[1]
	else
		pivot = median_of_medians(x)
		L, R = partition(x, pivot)
		if length(L) == k-1
			return pivot
		elseif length(L) > k-1
			return select(L, k)
		elseif length(L) < k-1
			return select(R, k-length(L)-1)
		end
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PlutoUI = "~0.7.16"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "f6532909bf3d40b308a0f360b6a0e626c0e263a8"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.1"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "98f59ff3639b3d9485a03a72f3ab35bab9465720"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.6"

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "4c8a7d080daca18545c56f1cac28710c362478f3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.16"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╟─d58b2cfe-be5f-4dd9-bf57-10b8927936ae
# ╠═b372e570-2fcf-11ec-0bc1-73431718551d
# ╟─36190c75-ddea-4ba1-b0a3-0da601f6862f
# ╠═6d901be0-6e00-411a-b585-c7260dd74246
# ╠═b2882ede-0bdb-4841-b801-65800643953e
# ╟─5eabe62f-80a5-4d33-a27d-7a56b91ca060
# ╠═074a179c-6e63-4bce-a0ef-845d8351c99b
# ╠═8e9f512a-eb5c-46ac-829d-50cd789c5f0a
# ╠═2ebc9063-07f0-456c-afc7-7e9ff18e5dd1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
