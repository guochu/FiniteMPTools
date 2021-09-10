


struct OverlapCache{_A<:FiniteMPS, _B<:FiniteMPS, _C} <: AbstractCache
	A::_A
	B::_B
	cstorage::_C
end


function environments(psiA::FiniteMPS, psiB::FiniteMPS)
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(SpaceMismatch())
	hold = r_RR(psiB)
	L = length(psiA)
	cstorage = Vector{typeof(hold)}(undef, L+1)
	cstorage[L+1] = hold
	for i in L:-1:1
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return OverlapCache(psiA, psiB, cstorage)
end

environments(psi::FiniteDensityOperatorMPS) = environments(psi.I, psi.data)



