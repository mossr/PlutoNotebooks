### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ c38d21a1-673d-4b50-8f3c-ae068bf57bb2
using LinearAlgebra # for `dot`

# ╔═╡ 478e9671-312a-41f0-8899-fb971147c996
using PyCall

# ╔═╡ 3c32c070-2787-11ec-2c66-27b0c646bc9e
begin
	using PlutoUI

	md"""
	# Subscript and Linear Indexing
	Here we'll show examples of using `sub2ind` defined in [_Algorithms for Decision Making_](https://algorithmsbook.com/files/dm.pdf) (algorithm 4.1).

	We'll also cover the analogous functions in Julia, Python, and MATLAB.
	"""
end

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

# ╔═╡ 4c50fd12-c92c-4066-8f50-00938b95b1ff
TableOfContents(aside=false)

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
PlutoUI = "~0.7.71"
PyCall = "~1.96.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "dd594eeff41332fa113890893f1ce11a15f08dce"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
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
# ╟─81b41745-2d23-4b4d-80f2-d2092f0e9742
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
# ╠═b6914b28-fa80-4a6f-ba5c-a46d8e73c09c
# ╟─c4ba0fef-22b4-4c0d-b460-1432ed5a386f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
