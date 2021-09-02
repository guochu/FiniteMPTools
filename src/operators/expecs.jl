

function _compute_overlap_env(psiA::FiniteMPS, psiB::FiniteMPS)
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(SpaceMismatch())
	hold = r_RR(psiB)
	L = length(psiA)
	cstorage = Vector{typeof(hold)}(undef, L+1)
	cstorage[L+1] = hold
	for i in L:-1:1
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return cstorage
end


"""
	assume the underlying state is canonical
"""
function _expectation(m::QTerm, psi::FiniteMPS)
	isstrict(m) || throw(ArgumentError("QTerm should conserve quantum number."))
	is_constant(m) || throw(ArgumentError("only constant QTerm allowed."))
	is_zero(m) && return 0.
	L = length(psi)
	pos = positions(m)
	ops = op(m)
	pos_end = pos[end]
	(pos_end <= L) || throw(BoundsError())
	util = get_trivial_leg(psi[1])
	@tensor hold[-1; -2 -3] := conj(psi[pos_end][-1, 1, 2]) * psi[pos_end][-3, 4, 2] * ops[end][-2, 1, 3, 4] * util[3] 
	for j in pos_end-1:-1:pos[1]
		pj = findfirst(x->x==j, pos)
		if isnothing(pj)
			hold = updateright(hold, psi[j], pj, psi[j])
		else
			hold = updateright(hold, psi[j], ops[pj], psi[j])
		end
	end	 
	s = psi.s[pos[1]]
	@tensor hnew[-1; -2] := conj(s[-1, 1]) * hold[1, 2, 3] * conj(util[2]) * s[-2, 3]
	return tr(hnew) * value(coeff(m))
end

function _expectation(psiA::FiniteMPS, m::QTerm, psiB::FiniteMPS, cstorage::Vector)
	(length(psiA) == length(psiB) == length(cstorage)-1) || throw(DimensionMismatch())
	isstrict(m) || throw(ArgumentError("QTerm should conserve quantum number."))
	is_constant(m) || throw(ArgumentError("only constant QTerm allowed."))
	is_zero(m) && return 0.
	L = length(psiA)
	pos = positions(m)
	ops = op(m)
	pos_end = pos[end]
	(pos_end <= L) || throw(BoundsError())
	util = get_trivial_leg(psiA[1])
	@tensor hold[-1; -2 -3] := conj(psiA[pos_end][-1, 1, 2]) * cstorage[pos_end+1][2, 3] * psiB[pos_end][-3, 5, 3] * ops[end][-2, 1, 4, 5] * util[4]  
	for j in pos_end-1:-1:pos[1]
		pj = findfirst(x->x==j, pos)
		if isnothing(pj)
			hold = updateright(hold, psiA[j], pj, psiB[j])
		else
			hold = updateright(hold, psiA[j], ops[pj], psiB[j])
		end
	end
	@tensor hnew[-1; -2] := conj(util[1]) * hold[-1, 1, -2]
	for j in pos[1]-1:-1:1
		hnew = updateright(hnew, psiA[j], psiB[j])
	end
	return scalar(hnew) * value(coeff(m))
end

expectation(psiA::FiniteMPS, m::QTerm, psiB::FiniteMPS) = _expectation(psiA, m, psiB, _compute_overlap_env(psiA, psiB))
expectation(m::QTerm, psi::FiniteMPS; iscanonical::Bool=false) = iscanonical ? _expectation(m, psi) : expectation(psi, m, psi)

function expectation(psiA::FiniteMPS, h::QOperator, psiB::FiniteMPS)
	(length(h) <= length(psiA)) || throw(DimensionMismatch())
	cstorage = _compute_overlap_env(psiA, psiB)
	r = 0.
	for m in qterms(h)
		r += _expectation(psiA, m, psiB, cstorage)
	end
	return r
end
function expectation(h::QOperator, psi::FiniteMPS; iscanonical::Bool=false)
	if iscanonical
		r = 0.
		for m in qterms(h)
			r += _expectation(m, psi)
		end
		return r
	else
		return expectation(psi, h, psi)
	end
end


