

struct ExactFiniteMPS{M<:MPSTensor} <: AbstractMPS
	data::Vector{M}
	center::Int
end
raw_data(psi::ExactFiniteMPS) = psi.data

Base.eltype(psi::ExactFiniteMPS) = eltype(typeof(psi))
Base.length(psi::ExactFiniteMPS) = length(raw_data(psi))
Base.isempty(psi::ExactFiniteMPS) = isempty(raw_data(psi))
Base.size(psi::ExactFiniteMPS, i...) = size(raw_data(psi), i...)
Base.getindex(psi::ExactFiniteMPS, i::Int) = getindex(raw_data(psi), i)
Base.lastindex(psi::ExactFiniteMPS) = lastindex(raw_data(psi))
Base.firstindex(psi::ExactFiniteMPS) = firstindex(raw_data(psi))

TensorKit.spacetype(::Type{ExactFiniteMPS{A}}) where {A} = spacetype(A)
TensorKit.spacetype(psi::ExactFiniteMPS) = spacetype(typeof(psi))


scalar_type(psi::ExactFiniteMPS) = scalar_type(typeof(psi))
function Base.setindex!(psi::ExactFiniteMPS, v, i::Int)
	_check_mps_tensor_dir(v) || throw(SpaceMismatch())
	(i == psi.center) || throw(ArgumentError("only center can be set."))
	space(v) == space(raw_data(psi)[i]) || throw(SpaceMismatch())
	return setindex!(raw_data(psi), v, i)
end 

Base.eltype(::Type{ExactFiniteMPS{A}}) where {A <: MPSTensor} = A
scalar_type(::Type{ExactFiniteMPS{A}}) where {A <: MPSTensor} = eltype(A)
Base.copy(psi::ExactFiniteMPS) = ExactFiniteMPS(copy(raw_data(psi)), psi.center)

space_l(state::ExactFiniteMPS) = space(state[1], 1)
space_r(state::ExactFiniteMPS) = space(state[end], 3)

"""
	r_RR, right boundary 2-tensor
	i-1
	o-2
"""
r_RR(state::ExactFiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)}, space_r(state), space_r(state))
"""
	l_LL, left boundary 2-tensor
	o-1
	i-2
"""
l_LL(state::ExactFiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)}, space_l(state), space_l(state))


FiniteMPS(psi::ExactFiniteMPS) = FiniteMPS(raw_data(psi))

function _find_center(Ds::Vector{Int})
	pos = argmax(Ds)
	if (pos != 1) && (Ds[pos-1] >= Ds[pos+1])
		pos = pos - 1
	end
	return pos
end

function _exactmps_side_tensors(::Type{T}, physpaces::Vector{S}, sector::Sector= first(sectors(oneunit(S)))) where {T <:Number, S <: EuclideanSpace}
	(sectortype(S) == typeof(sector)) || throw(SpaceMismatch())
	left = oneunit(S)
	L = length(physpaces)
	middle_site = _find_center(max_bond_dimensions(physpaces, sector))
	mpstensors = Vector{Any}(undef, L)
	for i in 1:middle_site-1
		physpace = physpaces[i]
		mpstensors[i] = isomorphism(Matrix{T}, left ⊗ physpace, fuse(left, physpace))
		left = space(mpstensors[i], 3)'
	end
	right = S(sector=>1)
	for i in L:-1:middle_site+1
		physpace = physpaces[i]
		tmp = isomorphism(Matrix{T}, fuse(physpace', right), physpace' ⊗ right)
		mpstensors[i] = permute(tmp, (1, 2), (3,))
		right = space(mpstensors[i], 1)
	end
	return mpstensors, left, right, middle_site
end

function ExactFiniteMPS(f, ::Type{T}, physpaces::Vector{S}; sector::Sector= first(sectors(oneunit(S)))) where {T <:Number, S <: EuclideanSpace}
	mpstensors, left, right, middle_site = _exactmps_side_tensors(T, physpaces, sector)
	mpstensors[middle_site] = TensorMap(f, T, left ⊗ physpaces[middle_site], right)
	(norm(mpstensors[middle_site]) == 0.) && throw(ArgumentError("invalid sector."))
	return ExactFiniteMPS([mpstensors...], middle_site)
end

function ExactFiniteMPS(psi::FiniteMPS)
	isstrict(psi) || throw(ArgumentError("only strict mps allowed."))
	sr = space_r(psi)'
	sector = first(sectors(sr))
	target_psi, left, right, middle_site = _exactmps_side_tensors(scalar_type(psi), [space(item, 2) for item in raw_data(psi)], sector)

	L = length(psi)

	cleft = l_LL(psi)
	cright = r_RR(psi)
	for i in 1:middle_site-1
		cleft = updateleft(cleft, target_psi[i], psi[i])
	end
	for i in L:-1:middle_site+1
		cright = updateright(cright, target_psi[i], psi[i])
	end
	target_psi[middle_site] = c_proj(psi[middle_site], cleft, cright)
	return ExactFiniteMPS([target_psi...], middle_site)
end



