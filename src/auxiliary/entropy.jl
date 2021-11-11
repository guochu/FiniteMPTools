
function entanglement_spectrum(m::AbstractTensorMap{S, 1, 1}) where S
	u, ss, v, err = stable_tsvd(m, trunc = TensorKit.NoTruncation())
	return LinearAlgebra.diag(convert(Array,ss))
end
