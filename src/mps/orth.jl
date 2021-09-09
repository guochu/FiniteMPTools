# orthogonalize mps to be left-canonical or right-canonical



function TensorKit.leftorth!(psi::FiniteMPS; trunc::TruncationScheme = TensorKit.NoTruncation())
	L = length(psi)
	(L > 1) || error("number of input mps tensors must be larger than 1.")
	errs = Float64[]
	maybe_init_boundary_s!(psi)
	for i in 1:L-1
		u, s, v, err = stable_tsvd(psi[i], (1, 2), (3,), trunc=trunc)
		psi[i] = u
		@tensor v2[-1; -2] := s[-1, 1] * v[1, -2]
		@tensor tmp[-1 -2; -3] := v2[-1, 1] * psi[i+1][1,-2,-3]
		psi[i+1] = tmp
		psi.s[i+1] = s
		push!(errs, err)
	end
	return errs
end

function TensorKit.rightorth!(psi::FiniteMPS; trunc::TruncationScheme = TensorKit.NoTruncation())
	L = length(psi)
	(L > 1) || error("number of input mps tensors must be larger than 1.")
	errs = Float64[]
	maybe_init_boundary_s!(psi)
	for i in L:-1:2
		u, s, v, err = stable_tsvd(psi[i], (1,), (2, 3), trunc=trunc)
		psi[i] = permute(v, (1,2), (3,))
		@tensor u2[-1; -2] := u[-1, 1] * s[1, -2]
		@tensor tmp[-1 -2; -3] := psi[i-1][-1, -2, 1] * u2[1, -3]
		psi[i-1] = tmp
		psi.s[i] = s
		push!(errs, err)
	end
	return errs
end


function right_canonicalize!(psi::FiniteMPS; normalize::Bool=false, trunc::TruncationScheme = TensorKit.NoTruncation())
	err1 = leftorth!(psi, trunc=trunc)
	if normalize
		psi[end] /= norm(psi[end])
	end
	err2 = rightorth!(psi, trunc=trunc)
	return vcat(err1, err2)
end

canonicalize!(psi::FiniteMPS; kwargs...) = right_canonicalize!(psi; kwargs...)
svdcompress!(psi; kwargs...) = canonicalize!(psi; kwargs...)