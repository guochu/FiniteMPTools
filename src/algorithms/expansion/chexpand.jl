



struct CHExpansion <: SubspaceExpansionScheme
	ϵ::Float64
	momentum::Float64
end

"""
	see PHYSICAL REVIEW B 91, 155115 (2015)	
	does not support MPOHamiltonian yet
"""
function CHExpansion(;ϵ::Real=0.01, momentum::Real=0.9)
	(0. < momentum <= 1.) || throw(ArgumentError("momentum should be in (0, 1]."))
	return CHExpansion(convert(Float64, ϵ), convert(Float64, momentum))
end 


function subspace_expansion_left(m::CHExpansion, hleft::AbstractTensorMap, mpoj, mpsA1, mpsA2)
	alpha = m.ϵ
	(alpha == zero(alpha)) && return mpsA1, mpsA2
	_x = space(mpoj, 3)'
	_y = space(mpsA1, 3)'
	ft = isomorphism(_x ⊗ _y, fuse(_x, _y))
	@tensor a[-1 -2; -3] := alpha * hleft[-1, 1, 2] * mpsA1[2, 3, 4] * mpoj[1,-2,5,3] * ft[5, 4, -3]
	b = TensorMap(zeros, eltype(a), space(a, 3)' ⊗ space(mpsA2, 2) ← space(mpsA2, 3)')
	return catdomain(a, mpsA1), permute(catcodomain(permute(b, (1,), (2,3)), permute(mpsA2, (1,), (2,3))), (1,2), (3,))
end

function subspace_expansion_right(m::CHExpansion, hright::AbstractTensorMap, mpoj, mpsA1, mpsA2)
	alpha = m.ϵ
	(alpha == zero(alpha)) && return mpsA1, mpsA2
	_x = space(mpoj, 1)
	_y = space(mpsA2, 1)
	ft = isomorphism(fuse(_x, _y), _x ⊗ _y)
	@tensor b[-1, -2; -3] := alpha * ft[-1, 1, 2] * mpsA2[2,3,4] * mpoj[1, -2, 5, 3] * hright[-3, 5, 4]
	a = TensorMap(zeros, eltype(b), space(mpsA1, 1) ⊗ space(mpsA1, 2) ← space(b, 1) )
	return catdomain(a, mpsA1), permute(catcodomain(permute(b, (1,), (2,3)), permute(mpsA2, (1,), (2,3))), (1,2), (3,))
end


# interface
function left_expansion!(m::Union{FiniteEnv, ExcitedFiniteEnv}, site::Int, expan::CHExpansion, trunc::TruncationScheme)
	mpo = m.h
	mps = m.mps
	hstorage = m.env
	a, b = subspace_expansion_left(expan, hstorage[site-1], mpo[site-1], mps[site-1], mps[site])
	u, s, v = tsvd!(a, trunc=trunc)
	mps[site-1] = u
	mps[site] = @tensor tmp[-1 -2; -3] := s[-1, 1] * v[1, 2] * b[2,-2,-3]
	updateleft!(m, site-1)
end
function right_expansion!(m::Union{FiniteEnv, ExcitedFiniteEnv}, site::Int, expan::CHExpansion, trunc::TruncationScheme)
	mpo = m.h
	mps = m.mps
	hstorage = m.env	
	a, b = subspace_expansion_right(expan, hstorage[site+2], mpo[site+1], mps[site], mps[site+1])
	u, s, v = tsvd(b, (1,), (2,3), trunc=trunc)
	mps[site+1] = permute(v, (1,2), (3,))
	mps[site] = a * u * s
	updateright!(m, site+1)
end


