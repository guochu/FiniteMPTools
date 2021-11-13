

function init_hstorage!(hstorage::Vector, mpo::Union{FiniteMPO, MPOHamiltonian}, mps::FiniteMPS, center::Int)
	# (length(mpo) == length(mps)) || throw(DimensionMismatch())
	# (mod(length(mps), period(mpo))==0) || throw(DimensionMismatch())
	(length(mps)+1 == length(hstorage)) || throw(DimensionMismatch())
	(spacetype(mpo) == spacetype(mps)) || throw(SpaceMismatch())
	isstrict(mpo) || throw(ArgumentError("operator must be strict."))
	right = r_RR(mps, mpo, mps)
	L = length(mps)
	# hstorage = Vector{typeof(right)}(undef, L+1)
	hstorage[L+1] = right
	hstorage[1] = l_LL(mps, mpo, mps)
	for i in L:-1:center+1
		hstorage[i] = updateright(hstorage[i+1], mps[i], mpo[i], mps[i])
	end
	for i in 1:center-1
		hstorage[i+1] = updateleft(hstorage[i], mps[i], mpo[i], mps[i])
	end
	return hstorage
end

function init_hstorage(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::FiniteMPS, center::Int)
	hstorage = Vector{Any}(undef, length(mps)+1)
	init_hstorage!(hstorage, mpo, mps, center)
	return [hstorage...]
end

# function init_hstorage(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::FiniteMPS, center::Int)
# 	# (length(mpo) == length(mps)) || throw(DimensionMismatch())
# 	# (mod(length(mps), period(mpo))==0) || throw(DimensionMismatch())
# 	(spacetype(mpo) == spacetype(mps)) || throw(SpaceMismatch())
# 	isstrict(mpo) || throw(ArgumentError("operator must be strict."))
# 	right = r_RR(mps, mpo, mps)
# 	L = length(mps)
# 	hstorage = Vector{typeof(right)}(undef, L+1)
# 	hstorage[L+1] = right
# 	hstorage[1] = l_LL(mps, mpo, mps)
# 	for i in L:-1:center+1
# 		hstorage[i] = updateright(hstorage[i+1], mps[i], mpo[i], mps[i])
# 	end
# 	for i in 1:center-1
# 		hstorage[i+1] = updateleft(hstorage[i], mps[i], mpo[i], mps[i])
# 	end
# 	return hstorage
# end

init_hstorage_right(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::FiniteMPS) = init_hstorage(mpo, mps, 1)


struct ExpectationCache{M<:Union{FiniteMPO, MPOHamiltonian}, V<:FiniteMPS, H} <: AbstractCache
	mpo::M
	mps::V
	hstorage::H
end

environments(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::FiniteMPS) = ExpectationCache(mpo, mps, init_hstorage_right(mpo, mps))

function Base.getproperty(m::ExpectationCache, s::Symbol)
	if s == :state
		return m.mps
	elseif s == :h
		return m.mpo
	elseif s == :env 
		return m.hstorage
	else
		return getfield(m, s)
	end
end


function updateleft!(env::ExpectationCache, site::Int)
	env.hstorage[site+1] = updateleft(env.hstorage[site], env.mps[site], env.h[site], env.mps[site])
end

function updateright!(env::ExpectationCache, site::Int)
	env.hstorage[site] = updateright(env.hstorage[site+1], env.mps[site], env.h[site], env.mps[site])
end

# function recalculate!(m::ExpectationCache{M, <:FiniteNonSymmetricMPS}, mps::FiniteNonSymmetricMPS, center::Int) where M
# 	if mps !== m.state
# 		init_hstorage!(m.env, m.h, mps, center)
# 	end
# end

# increase_bond!(m::ExpectationCache; D::Int) = nothing
function increase_bond!(m::ExpectationCache; D::Int) 
	if isa(m.state, FiniteNonSymmetricMPS) && (bond_dimension(m.state) < D)
		increase_bond!(m.state, D=D)
		canonicalize!(m.state, normalize=false)
		init_hstorage!(m.env, m.h, m.state, 1)
	end
end


# for excited states
struct ProjectedExpectationCache{M<:Union{FiniteMPO, MPOHamiltonian}, V<:FiniteMPS, H, C} <: AbstractCache
	mpo::M
	mps::V
	projectors::Vector{V}
	hstorage::H
	cstorages::Vector{C}
end

function init_cstorage_right!(cstorage::Vector, psiA::FiniteMPS, psiB::FiniteMPS)
	(length(cstorage) == length(psiA)+1) || throw(DimensionMismatch())
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(SpaceMismatch())
	L = length(psiA)
	hold = r_RR(psiA, psiB)
	# cstorage = Vector{Any}(undef, L+1)
	cstorage[1] = l_LL(psiA)
	cstorage[L+1] = hold
	for i in L:-1:2
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return cstorage
end

function init_cstorage_right(psiA::FiniteMPS, psiB::FiniteMPS)
	cstorage = Vector{Any}(undef, length(psiA)+1)
	init_cstorage_right!(cstorage, psiA, psiB)
	return [cstorage...]
end

environments(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::M, projectors::Vector{M}) where {M <: FiniteMPS} = ProjectedExpectationCache(
	mpo, mps, projectors, init_hstorage_right(mpo, mps), [init_cstorage_right(mps, item) for item in projectors])

function Base.getproperty(m::ProjectedExpectationCache, s::Symbol)
	if s == :state
		return m.mps
	elseif s == :h
		return m.mpo
	elseif s == :env 
		return m.hstorage
	elseif s == :cenvs
		return m.cstorages
	else
		return getfield(m, s)
	end
end


function updateleft!(env::ProjectedExpectationCache, site::Int)
	env.hstorage[site+1] = updateleft(env.hstorage[site], env.mps[site], env.h[site], env.mps[site])
	for l in 1:length(env.cstorages)
	    env.cstorages[l][site+1] = updateleft(env.cstorages[l][site], env.mps[site], env.projectors[l][site])
	end
end


function updateright!(env::ProjectedExpectationCache, site::Int)
	env.hstorage[site] = updateright(env.hstorage[site+1], env.mps[site], env.h[site], env.mps[site])
	for l in 1:length(env.cstorages)
	    env.cstorages[l][site] = updateright(env.cstorages[l][site+1], env.mps[site], env.projectors[l][site])
	end
end

function increase_bond!(m::ProjectedExpectationCache; D::Int) 
	if isa(m.state, FiniteNonSymmetricMPS) && (bond_dimension(m.state) < D)
		increase_bond!(m.state, D=D)
		canonicalize!(m.state, normalize=false)
		init_hstorage!(m.env, m.h, m.state, 1)
		for (cstorage, mps) in zip(m.cenvs, m.projectors)
			init_cstorage_right!(cstorage, m.state, mps)
		end
	end
end






