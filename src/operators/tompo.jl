

function prodmpo(physpaces::Vector{S}, m::QTerm) where {S <: EuclideanSpace}
	(S == spacetype(m)) || throw(SpaceMismatch())
	is_constant(m) || throw(ArgumentError("only constant term allowed."))
	return prodmpo(scalar_type(m), physpaces, positions(m), op(m)) * value(coeff(m))
end


function FiniteMPO(h::QOperator{S}; tol::Real=DeparalleliseTol) where {S <: EuclideanSpace}
	physpaces = convert(Vector{S}, space(h))
	local mpo
	for m in qterms(h)
		# println("positions*******************$(positions(m))")
		if @isdefined mpo
			mpo += prodmpo(physpaces, m)
		else
			mpo = prodmpo(physpaces, m)
		end
		if bond_dimension(mpo) >= 20
			mpo = deparallelise!(mpo, tol=tol)
		end
	end
	mpo = deparallelise!(mpo, tol=tol)
	return mpo
end

