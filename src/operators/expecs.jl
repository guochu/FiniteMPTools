
"""
	assume the underlying state is canonical
"""
function expectation_canonical(m::QTerm, psi::FiniteMPS)
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

function expectation(psiA::FiniteMPS, m::QTerm, psiB::FiniteMPS, envs::OverlapCache=environments(psiA, psiB))
	cstorage = envs.cstorage
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

expectation(m::QTerm, psi::FiniteMPS; iscanonical::Bool=false) = iscanonical ? expectation_canonical(m, psi) : expectation(psi, m, psi)

function expectation(psiA::FiniteMPS, h::QuantumOperator, psiB::FiniteMPS)
	(length(h) <= length(psiA)) || throw(DimensionMismatch())
	envs = environments(psiA, psiB)
	r = 0.
	for m in qterms(h)
		r += expectation(psiA, m, psiB, envs)
	end
	return r
end
function expectation(h::QuantumOperator, psi::FiniteMPS; iscanonical::Bool=false)
	if iscanonical
		r = 0.
		for m in qterms(h)
			r += expectation_canonical(m, psi)
		end
		return r
	else
		return expectation(psi, h, psi)
	end
end

# density operator
"""
	expectation(m::QTerm, psi::FiniteDensityOperatorMPS) 
	⟨I|h|ρ⟩
"""
expectation(m::QTerm, psi::FiniteDensityOperatorMPS, envs=environments(psi)) = expectation(psi.I, m, psi.data, envs)

function expectation(m::QuantumOperator, psi::FiniteDensityOperatorMPS)
	envs = environments(psi)
	r = 0.
	for m in qterms(h)
		r += expectation(m, psi, envs)
	end
	return r
end
