
abstract type MPOCompression end

function svdcompress!(h::FiniteMPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	L = length(h)
	(L > 1) || error("number of input mpo must be larger than 1.")
	Errs = Float64[]
	# we basicaly do not use a hard truncation D for MPO truncation
	trunc = MPSTruncation(D=10000, ϵ=tol, verbosity=verbosity)
	for i in 1:L-1
		u, s, v, err = stable_tsvd(h[i], (1,2,4), (3,), trunc=trunc)
		h[i] = permute(u * s, (1,2), (4,3))
		h[i+1] = @tensor tmp[-1 -2; -3 -4] := v[-1, 1] * h[i+1][1,-2,-3,-4]
		push!(Errs, err)
	end
	for i in L:-1:2
		u, s, v, err = stable_tsvd(h[i], (1,), (2,3,4), trunc=trunc)
		h[i] = permute(s * v, (1,2), (3,4))
		h[i-1] = @tensor tmp[-1 -2; -3 -4] := h[i-1][-1, -2, 1, -4] * u[1, -3]
		push!(Errs, err)
	end
	(verbosity >= 2) && println("maximum truncation error is $(maximum(Errs)).")
	return h
end


@with_kw struct SVDCompression <: MPOCompression
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity
end

@with_kw struct Deparallelise <: MPOCompression
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity
end

@with_kw struct CombineCompression <: MPOCompression
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity	
end


_compress!(h::FiniteMPO, alg::Deparallelise) = deparallelise!(h, tol=alg.tol, verbosity=alg.verbosity)
_compress!(h::FiniteMPO, alg::SVDCompression) = svdcompress!(h, tol=alg.tol, verbosity=alg.verbosity)
function _compress!(h::FiniteMPO, alg::CombineCompression)
	_compress!(h, Deparallelise(tol=alg.tol, verbosity=alg.verbosity))
	return _compress!(h, SVDCompression(tol=alg.tol, verbosity=alg.verbosity))
end
default_mpo_compression() = CombineCompression()
compress!(h::FiniteMPO, alg::MPOCompression=default_mpo_compression()) = _compress!(h, alg)

