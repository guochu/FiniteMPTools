

function init_hstorage(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::FiniteMPS, center::Int)
	# (length(mpo) == length(mps)) || throw(DimensionMismatch())
	# (mod(length(mps), period(mpo))==0) || throw(DimensionMismatch())
	(spacetype(mpo) == spacetype(mps)) || throw(SpaceMismatch())
	isstrict(mpo) || throw(ArgumentError("operator must be strict."))
	right = r_RR(mps, mpo, mps)
	L = length(mps)
	hstorage = Vector{typeof(right)}(undef, L+1)
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

init_hstorage_right(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::FiniteMPS) = init_hstorage(mpo, mps, 1)


struct FiniteEnv{M<:Union{FiniteMPO, MPOHamiltonian}, V<:FiniteMPS, H} <: AbstractEnv
	mpo::M
	mps::V
	hstorage::H
end

environments(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::FiniteMPS) = FiniteEnv(mpo, mps, init_hstorage_right(mpo, mps))

function Base.getproperty(m::FiniteEnv, s::Symbol)
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


function updateleft!(env::FiniteEnv, site::Int)
	env.hstorage[site+1] = updateleft(env.hstorage[site], env.mps[site], env.h[site], env.mps[site])
end

function updateright!(env::FiniteEnv, site::Int)
	env.hstorage[site] = updateright(env.hstorage[site+1], env.mps[site], env.h[site], env.mps[site])
end



# for excited states
struct ExcitedFiniteEnv{M<:Union{FiniteMPO, MPOHamiltonian}, V<:FiniteMPS, H, C} <: AbstractEnv
	mpo::M
	mps::V
	projectors::Vector{V}
	hstorage::H
	cstorages::Vector{C}
end


function init_cstorage_right(psiA::FiniteMPS, psiB::FiniteMPS)
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(SpaceMismatch())
	L = length(psiA)
	hold = r_RR(psiA)
	cstorage = Vector{typeof(hold)}(undef, L+1)
	cstorage[1] = l_LL(psiA)
	cstorage[L+1] = hold
	for i in L:-1:2
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return cstorage
end

environments(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::M, projectors::Vector{M}) where {M <: FiniteMPS} = ExcitedFiniteEnv(
	mpo, mps, projectors, init_hstorage_right(mpo, mps), [init_cstorage_right(mps, item) for item in projectors])

function Base.getproperty(m::ExcitedFiniteEnv, s::Symbol)
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


function updateleft!(env::ExcitedFiniteEnv, site::Int)
	env.hstorage[site+1] = updateleft(env.hstorage[site], env.mps[site], env.h[site], env.mps[site])
	for l in 1:length(env.cstorages)
	    env.cstorages[l][site+1] = updateleft(env.cstorages[l][site], env.mps[site], env.projectors[l][site])
	end
end


function updateright!(env::ExcitedFiniteEnv, site::Int)
	env.hstorage[site] = updateright(env.hstorage[site+1], env.mps[site], env.h[site], env.mps[site])
	for l in 1:length(env.cstorages)
	    env.cstorages[l][site] = updateright(env.cstorages[l][site+1], env.mps[site], env.projectors[l][site])
	end
end






