

# see vumps paper
struct OptimalExpansion <: SubspaceExpansionScheme end

default_expansion() = OptimalExpansion()


# interface

# used when sweep from right to left, expand the left site
function left_expansion!(m::Union{FiniteEnv, ExcitedFiniteEnv}, site::Int, expan::OptimalExpansion, trunc::TruncationScheme)
	mpo = m.h
	mps = m.mps
	hstorage = m.env

	ACi = mps[site-1]
	@tensor ACAR[-1 -2; -3 -4] := ACi[-1,-2,1] * mps[site][1, -3, -4]

	# println("left *************** $(norm(ACAR))")

	AC2 = ac2_prime(ACAR, mpo[site-1], mpo[site], hstorage[site-1], hstorage[site+1])
	NL = leftnull(ACi)
	NR = rightnull(permute(mps[site], (1,), (2,3)))

	intermediate = NL' * AC2 * NR'
	(U1,S1,V1) = stable_svd(intermediate,trunc=trunc)

	a = NL * U1
	b = TensorMap(zeros, space(a, 3)' ⊗ space(mps[site], 2), domain(mps[site]) )

	a = catdomain(ACi, a)
	b = permute(catcodomain(permute(mps[site], (1,), (2,3)), permute(b, (1,), (2,3))), (1,2), (3,))

	u, s, v = stable_svd!(a, trunc=trunc)
	mps[site-1] = u
	mps[site] = @tensor tmp[-1 -2; -3] := s[-1, 1] * v[1, 2] * b[2,-2,-3]
	updateleft!(m, site-1)
end

# used when sweep from left to right, expand the right site
function right_expansion!(m::Union{FiniteEnv, ExcitedFiniteEnv}, site::Int, expan::OptimalExpansion, trunc::TruncationScheme)
	mpo = m.h
	mps = m.mps
	hstorage = m.env

	ACi = mps[site+1]
	@tensor ACAR[-1 -2; -3 -4] := mps[site][-1, -2, 1] * ACi[1,-3,-4] 

	# println("right *************** $(norm(ACAR))")

	AC2 = ac2_prime(ACAR, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2])
	NL = leftnull(mps[site])
	NR = rightnull(permute(ACi, (1,), (2,3)))

	intermediate = NL' * AC2 * NR'
	(U1,S1,V1) = stable_svd(intermediate,trunc=trunc)

    a = TensorMap(zeros,codomain(mps[site]),space(V1,1))
    b = V1*NR;

	a = catdomain(mps[site], a)
	b = permute(catcodomain(permute(ACi, (1,), (2,3)), permute(b, (1,), (2,3))), (1,2), (3,))

	u, s, v = stable_svd(b, (1,), (2,3), trunc=trunc)
	mps[site+1] = permute(v, (1,2), (3,))
	mps[site] = a * u * s
	updateright!(m, site+1)
end

