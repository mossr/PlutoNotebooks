### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 3c32c070-2787-11ec-2c66-27b0c646bc9e
begin
	using PlutoUI

	md"""
	# Subscript and Linear Indexing
	Here we'll show examples of using `sub2ind` defined in [_Algorithms for Decision Making_](https://algorithmsbook.com/files/dm.pdf) (algorithm 4.1).

	We'll also cover the analogous functions in Julia, Python, and MATLAB.
	"""
end

# ╔═╡ c38d21a1-673d-4b50-8f3c-ae068bf57bb2
using LinearAlgebra # for `dot`

# ╔═╡ 478e9671-312a-41f0-8899-fb971147c996
using PyCall

# ╔═╡ 4c50fd12-c92c-4066-8f50-00938b95b1ff
TableOfContents(aside=false)

# ╔═╡ 8a67d915-ff69-425a-a4e4-d31eeb61ddb0
md"""
From the MATLAB `sub2ind` documentation:
"""

# ╔═╡ 29ca6f4b-0b23-4679-b2e3-ca63e3efa1fd
LocalResource("img/ConvertSubscriptIndicesToLinearIndicesForMatricesExample_01.png")

# ╔═╡ 2eb8fcb3-a94f-473d-8d4c-cd4e5cff82cc
md"""
From the MATLAB `ind2sub` documentation:
"""

# ╔═╡ 6c370fdd-e513-4c3d-8d9b-c0014c18913e
LocalResource("img/ConvertLinearIndicesToSubscriptsForMatricesExample_01.png")

# ╔═╡ cd1f62f6-490a-4cbc-9669-d59fef04cbf8
md"""
## Subscript to linear index: `sub2ind`
"""

# ╔═╡ 2300a351-0eeb-4bd6-bba8-4c87653be842
md"""
The function `sub2ind` (algorithm 4.1) returns a linear index into an array with dimensions specified by `siz` given the coordinates `x`.

In other words, we want to go from some coordinates like $(2,3)$ to a single value (_linear_) index like $8$.
"""

# ╔═╡ dd5680dd-70c3-40c6-936c-37508c06d34b
function sub2ind(siz, x)
	k = vcat(1, cumprod(siz[1:end-1]))
	return dot(k, x .- 1) + 1
end

# ╔═╡ a19c4e91-7443-438c-8783-5ead94a3957e
md"""
## Example of `sub2ind`
"""

# ╔═╡ 93ca7b82-8923-4580-bc68-3b26e8bc9629
md"""
We'll create an example `matrix` of size $3\times3$.
"""

# ╔═╡ 87e26ad3-43a3-4a5c-808b-373de0736f11
matrix = [1.1  1.2  1.3;
	      2.1  2.2  2.3;
	      3.1  3.2  3.3]

# ╔═╡ e608f519-ab47-4e14-ad26-20836aad8de8
md"""
Then we get the `size` of the matrix.
"""

# ╔═╡ c68ffa89-c980-4826-ae76-fa1702664e54
siz = size(matrix)

# ╔═╡ e355dcc8-f189-4cc3-bcc8-4f2480e69b3a
md"""
Now we specify the **subscript** style coordinates we want to pull from the matrix. Here we use [row, col] to get the **linear index** of the 2nd row, 3rd column of the matrix.
"""

# ╔═╡ 269b36c1-7fd9-4395-b251-4f6283d83939
coordinate = (2,3) # we want to element in row 2, column 3

# ╔═╡ ad6653b8-48ce-4094-ac68-fb67ae8525f8
md"""
We use `sub2ind` to get the **linear index** given the size of our matrix and the **subscript** coordinate.
"""

# ╔═╡ 6d0a9223-fc70-4410-a9e5-d62f23387eb7
index = sub2ind(siz, coordinate)

# ╔═╡ 4f94f4ca-7cb2-41ad-b0fc-9e1a7bb0c32b
md"""
#### Linear Indexing (i.e., a single value as an index)
"""

# ╔═╡ ac3c91db-c87e-42ee-bbf4-07589910885b
matrix[index]

# ╔═╡ fda83ccb-9750-4903-b1d8-7f3fd363f3b7
md"""
#### Subscript Indexing (i.e., multiple values as indices)
"""

# ╔═╡ 151b4b69-2bbc-432d-886c-0cc33621d115
matrix[coordinate...] # same as matrix[2,3]

# ╔═╡ d082fee2-0ec5-434c-8cdc-80345b3c5556
matrix[2,3]

# ╔═╡ f8f7f5a1-b035-4209-94da-a91e977fe942
md"""
Note here that `[coordinate...]` is called [splatting](https://docs.julialang.org/en/v1/manual/faq/#...-splits-one-argument-into-many-different-arguments-in-function-calls) that "splats" (or splits out) the arguments in place.

So instead of `[coordinate]` returning `[(2,3)]` we use `[coordinate...]` to get `[2,3]`.
"""

# ╔═╡ ce89f273-c758-4996-ad86-d2f2524e18f2
md"""
# Julia
Similar built-in functions can be used instead of the `sub2ind` provided above.

- `LinearIndices` [(docs)](https://docs.julialang.org/en/v1/base/arrays/#Base.LinearIndices)
- `CartesianIndices` [(docs)](https://docs.julialang.org/en/v1/base/arrays/#Base.IteratorsMD.CartesianIndices)


## `LinearIndices`
Here we want to go from a subscript index (multi-valued) to a linear index (single-valued).
```julia
LinearIndicies(A::AbstractArray) -> Array of Linear Indices
```
"""

# ╔═╡ ec4b8c56-8561-427b-9b22-d8c126c15e01
LocalResource("img/ConvertSubscriptIndicesToLinearIndicesForMatricesExample_01.png")

# ╔═╡ ab331378-b903-497c-bc68-8d27f795235c
md"""
The returned variable `linear` is a `Matrix` of the same size as the input matrix.
"""

# ╔═╡ 93f1073e-4144-4bfe-9a4f-16c20fbea7b0
linear = LinearIndices(matrix)

# ╔═╡ 12610f72-dd1b-4335-b96e-616a6433474c
md"""
Then you can use **subscript** coordinate-style indexing to get the corresponding **linear** index.
"""

# ╔═╡ 1fc619bb-09af-4e19-b06c-9044fcda93eb
index_jl = linear[coordinate...]

# ╔═╡ 3534511c-3366-4641-b21c-64ebf94eca57
md"""
Just as before, we can use this single-valued index to pull out the correct value from the matrix.
"""

# ╔═╡ 7d2820c9-7e72-4dc0-9b5c-13152f19b864
matrix[index_jl]

# ╔═╡ c8a5c87e-6166-4d84-9d63-204f005310d1
md"""
## `CartesianIndices`
Now if we want to go the _other direction_ (from a linear index to a subscript index), we can use `CartesianIndices` (here we use "Cartesian" and "subscript" interchangeably).

```julia
CartesianIndices(array_size) -> Array of Subscript/Cartesian Indices
```
"""

# ╔═╡ 0512fe41-bd46-4efa-a5ae-86b656b7b671
LocalResource("img/ConvertLinearIndicesToSubscriptsForMatricesExample_01.png")

# ╔═╡ e3dcbb1a-9d89-476b-b866-2195d1f54fb1
md"""
The returned variable `subscript` is a `Matrix` of the same size as the input matrix.
"""

# ╔═╡ 107000fb-0f6f-4f55-b2d2-2ca66e98a575
subscript = CartesianIndices(siz)

# ╔═╡ ea0651ad-54be-44e9-bbe7-0b53877d060a
md"""
Then if we know the _linear_ index we want (say $8$) we can set its corresponding _subscript_ coordinate.
"""

# ╔═╡ 460c68bc-71ae-4a0b-b514-32371874c9e0
coordinate_jl = subscript[8]

# ╔═╡ 50b8f2f2-b3b8-4b98-baad-bcfcbf4edf5b
md"""
And retrieve the element at index $8$ using the subscript-style of $(2,3)$. 
"""

# ╔═╡ e4e56238-dde5-4f5b-9e36-8728ab4f4d92
matrix[coordinate_jl]

# ╔═╡ d864a106-7649-43e8-b2fe-d17991456898
md"""
# Python

Using NumPy, we get similar functionality from the `numpy.ravel_multi_index` (i.e., subscript to linear) and `numpy.unravel_index` (i.e., linear to subscript).

- `numpy.ravel_multi_index` [(docs)](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html)
- `numpy.unravel_index` [(docs)](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html)
"""

# ╔═╡ 81b41745-2d23-4b4d-80f2-d2092f0e9742
md"""
> Before proceeding, note that Julia uses **column major** ordering while Python (namely, NumPy) uses **row major** ordering (yet another confusing piece, I know...).
>
> See [https://en.wikipedia.org/wiki/Row-_and_column-major_order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
"""

# ╔═╡ eea62400-97a0-424f-b11f-34323b61256e
LocalResource("img/row_and_column_major_order.png")

# ╔═╡ f947e045-a546-4670-80dc-82b58d50451a
md"""
```python
# row major (0-based)
[0 1 2
 3 4 5
 6 7 8]

# column major (0-based)
[0 3 6
 1 4 7
 2 5 8]
```
"""

# ╔═╡ fa338faf-f66e-400e-959f-9ac6051adc85
md"""
> We can use the `order='F'` to operate in column major ordering, but here we'll stick to row major (deafult) to illustrate the differences.
"""

# ╔═╡ 928c2ee3-6b3e-4cde-88f3-256f5dc6878c
md"""
## `numpy.ravel_multi_index`
This is Python's version of the `sub2ind`/`LinearIndices` functionality.
"""

# ╔═╡ 53ef67c3-5ddd-4c3d-96a9-e2bc3c9f28ee
md"""
**NOTE**: Unhide the cell below to edit the Python code and the cell above to see Python code execution.
"""

# ╔═╡ 9668bf6d-e6f9-4b55-ac29-68182fa7a17d
python_code_ravel = """
import numpy as np

# Create the same 3×3 matrix
matrix = np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]])

# Get matrix size and subscript coordinate of interest
siz = matrix.shape
coordinate = (1,2) # Note: 0-based indexing of (row 2, col 3)

# Get linear index using subscript coordinates
index = np.ravel_multi_index(coordinate, siz)

# Ravel (or flatten) matrix, then use linear index to get the element
element = matrix.ravel()[index]

print("Index: ", index, " (note row-major indexing)")
print("Element: ", element)
"""

# ╔═╡ c50fd1ed-db54-4f64-b641-db33f3ba7c25
Markdown.parse("""
```python
$python_code_ravel
```
""")

# ╔═╡ d1bd7f37-b1e8-4cd4-b35b-72c3023a537c
with_terminal() do
	python_code_ravel # to trigger Pluto
	
	# Execute Python code in py environment ($$ will interpolate the string into py)
	py"""
	$$python_code_ravel
	"""
end

# ╔═╡ a61db8f8-20ee-4987-b49a-f4f21af7ecd8
md"""
## `numpy.unravel_index`
This is Python's version of the `ind2sub`/`CartesianIndices` functionality.
"""

# ╔═╡ 545db4c4-5dd1-4a53-865b-f9dba07ee591
md"""
**NOTE**: Unhide the cell below to edit the Python code and the cell above to see Python code execution.
"""

# ╔═╡ b6914b28-fa80-4a6f-ba5c-a46d8e73c09c
python_code_unravel = """
import numpy as np

# Create the same 3×3 matrix
matrix = np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]])

# Get matrix size and subscript coordinate of interest
siz = matrix.shape
index = 5 # Note: 0-based and row-major indexing of (row 2, col 3)

# Get subscript coordinates using linear index
coordinate = np.unravel_index(index, siz)

# Use subscript coordinate to get the element (note, no need to "ravel/flatten")
element = matrix[coordinate]

print("Coordinate: ", coordinate, " (note 0-based indexing)")
print("Element: ", element)
"""

# ╔═╡ 55f6750b-b1ff-4d18-abab-ae579e2b5c32
Markdown.parse("""
```python
$python_code_unravel
```
""")

# ╔═╡ dce16bab-4aa6-4126-babd-539352289977
with_terminal() do
	python_code_unravel # to trigger Pluto
	
	# Execute Python code in py environment ($$ will interpolate the string into py)
	py"""
	$$python_code_unravel
	"""
end

# ╔═╡ c4ba0fef-22b4-4c0d-b460-1432ed5a386f
md"""
# MATLAB
For the `sub2ind` and `ind2sub` functions, see MATLAB's excellent documentation:
- `sub2ind` [(docs)](https://www.mathworks.com/help/matlab/ref/sub2ind.html): Subscript coordinates to linear index.
- `ind2sub` [(docs)](https://www.mathworks.com/help/matlab/ref/ind2sub.html): Linear index to subscript coordinates.

For a matrix with size `siz` ($3\times3$):
```julia
matrix = [1.1 1.2 1.3;
          2.1 2.2 2.3;
          3.1 3.2 3.3];
siz = size(matrix)
```

## `sub2ind`
To go from subscript coordinates $(2,3)$ to linear index:
```julia
index = sub2ind(siz, 2, 3)
# index = 8
```

## `ind2sub`
To go from linear index $8$ to subscript coordinates:
```julia
[row,col] = ind2sub(siz, 8)
# row = 2
# col = 3
```
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"

[compat]
PlutoUI = "~0.7.15"
PyCall = "~1.92.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

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

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "a8709b968a1ea6abc2dc1967cb1db6ac9a00dfb6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.5"

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "633f8a37c47982bff23461db0076a33787b17ecd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.15"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

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

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"
"""

# ╔═╡ Cell order:
# ╟─3c32c070-2787-11ec-2c66-27b0c646bc9e
# ╟─8a67d915-ff69-425a-a4e4-d31eeb61ddb0
# ╟─29ca6f4b-0b23-4679-b2e3-ca63e3efa1fd
# ╟─2eb8fcb3-a94f-473d-8d4c-cd4e5cff82cc
# ╟─6c370fdd-e513-4c3d-8d9b-c0014c18913e
# ╟─4c50fd12-c92c-4066-8f50-00938b95b1ff
# ╟─cd1f62f6-490a-4cbc-9669-d59fef04cbf8
# ╠═c38d21a1-673d-4b50-8f3c-ae068bf57bb2
# ╟─2300a351-0eeb-4bd6-bba8-4c87653be842
# ╠═dd5680dd-70c3-40c6-936c-37508c06d34b
# ╟─a19c4e91-7443-438c-8783-5ead94a3957e
# ╟─93ca7b82-8923-4580-bc68-3b26e8bc9629
# ╠═87e26ad3-43a3-4a5c-808b-373de0736f11
# ╟─e608f519-ab47-4e14-ad26-20836aad8de8
# ╠═c68ffa89-c980-4826-ae76-fa1702664e54
# ╟─e355dcc8-f189-4cc3-bcc8-4f2480e69b3a
# ╠═269b36c1-7fd9-4395-b251-4f6283d83939
# ╟─ad6653b8-48ce-4094-ac68-fb67ae8525f8
# ╠═6d0a9223-fc70-4410-a9e5-d62f23387eb7
# ╟─4f94f4ca-7cb2-41ad-b0fc-9e1a7bb0c32b
# ╠═ac3c91db-c87e-42ee-bbf4-07589910885b
# ╟─fda83ccb-9750-4903-b1d8-7f3fd363f3b7
# ╠═151b4b69-2bbc-432d-886c-0cc33621d115
# ╠═d082fee2-0ec5-434c-8cdc-80345b3c5556
# ╟─f8f7f5a1-b035-4209-94da-a91e977fe942
# ╟─ce89f273-c758-4996-ad86-d2f2524e18f2
# ╟─ec4b8c56-8561-427b-9b22-d8c126c15e01
# ╟─ab331378-b903-497c-bc68-8d27f795235c
# ╠═93f1073e-4144-4bfe-9a4f-16c20fbea7b0
# ╟─12610f72-dd1b-4335-b96e-616a6433474c
# ╠═1fc619bb-09af-4e19-b06c-9044fcda93eb
# ╟─3534511c-3366-4641-b21c-64ebf94eca57
# ╠═7d2820c9-7e72-4dc0-9b5c-13152f19b864
# ╟─c8a5c87e-6166-4d84-9d63-204f005310d1
# ╟─0512fe41-bd46-4efa-a5ae-86b656b7b671
# ╟─e3dcbb1a-9d89-476b-b866-2195d1f54fb1
# ╠═107000fb-0f6f-4f55-b2d2-2ca66e98a575
# ╟─ea0651ad-54be-44e9-bbe7-0b53877d060a
# ╠═460c68bc-71ae-4a0b-b514-32371874c9e0
# ╟─50b8f2f2-b3b8-4b98-baad-bcfcbf4edf5b
# ╠═e4e56238-dde5-4f5b-9e36-8728ab4f4d92
# ╟─d864a106-7649-43e8-b2fe-d17991456898
# ╠═478e9671-312a-41f0-8899-fb971147c996
# ╠═81b41745-2d23-4b4d-80f2-d2092f0e9742
# ╟─eea62400-97a0-424f-b11f-34323b61256e
# ╟─f947e045-a546-4670-80dc-82b58d50451a
# ╟─fa338faf-f66e-400e-959f-9ac6051adc85
# ╟─928c2ee3-6b3e-4cde-88f3-256f5dc6878c
# ╟─c50fd1ed-db54-4f64-b641-db33f3ba7c25
# ╟─d1bd7f37-b1e8-4cd4-b35b-72c3023a537c
# ╟─53ef67c3-5ddd-4c3d-96a9-e2bc3c9f28ee
# ╟─9668bf6d-e6f9-4b55-ac29-68182fa7a17d
# ╟─a61db8f8-20ee-4987-b49a-f4f21af7ecd8
# ╟─55f6750b-b1ff-4d18-abab-ae579e2b5c32
# ╟─dce16bab-4aa6-4126-babd-539352289977
# ╟─545db4c4-5dd1-4a53-865b-f9dba07ee591
# ╟─b6914b28-fa80-4a6f-ba5c-a46d8e73c09c
# ╟─c4ba0fef-22b4-4c0d-b460-1432ed5a386f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
