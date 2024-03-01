using LinearAlgebra
	function squeeze(A::AbstractArray)
		B=dropdims(A, dims=Tuple(findall(size(A) .== 1)))
		return B
	end
