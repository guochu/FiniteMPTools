

function prodmpo(physpaces::Vector{S}, m::QTerm) where {S <: EuclideanSpace}
	(S == spacetype(m)) || throw(SpaceMismatch())
	is_constant(m) || throw(ArgumentError("only constant term allowed."))
	return prodmpo(scalar_type(m), physpaces, positions(m), op(m)) * value(coeff(m))
end

prodmpo(physpaces::Vector{<: EuclideanSpace}, m::AdjointQTerm) = adjoint(prodmpo(physpaces, m.parent))


function FiniteMPO(h::QuantumOperator{S}; alg::MPOCompression=default_mpo_compression()) where {S <: EuclideanSpace}
	physpaces = convert(Vector{S}, space(h))
	local mpo
	compress_threshold = 20
	for m in qterms(h)
		if @isdefined mpo
			mpo += prodmpo(physpaces, m)
		else
			mpo = prodmpo(physpaces, m)
		end
		if bond_dimension(mpo) >= compress_threshold
			mpo = compress!(mpo, alg)
			compress_threshold += 5
		end
	end
	mpo = compress!(mpo, alg)
	return mpo
end

FiniteMPO(h::SuperOperatorBase; kwargs...) = FiniteMPO(h.data; kwargs...)
