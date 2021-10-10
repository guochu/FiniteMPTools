

_check_mpo_tensor_dir(m::MPOTensor) = (!isdual(space(m, 1))) && (!isdual(space(m, 2))) && isdual(space(m, 3)) && isdual(space(m, 4))

function _check_mpo_space(mpotensors::Vector; strict::Bool=true)
	all(_check_mpo_tensor_dir.(mpotensors)) || throw(SpaceMismatch())
	for i in 1:length(mpotensors)-1
		(space(mpotensors[i], 3) == space(mpotensors[i+1], 1)') || throw(SpaceMismatch())
	end
	# boundaries should be dimension 1
	u = oneunit(space(mpotensors[1], 1))
	if strict
		(space(mpotensors[1], 1) == u == space(mpotensors[end], 3)') || throw(SpaceMismatch())
	else
		(space(mpotensors[1], 1) == u) || throw(SpaceMismatch())
	end
	
	return true
end


# another MPO object is needed to support non-number conserving operators, such a

"""
	FiniteMPO{A <: MPOTensor}
Finite Matrix Product Operator which stores a chain of rank-4 site tensors.
"""
struct FiniteMPO{A <: MPOTensor} <: AbstractMPO
	data::Vector{A}

"""
	FiniteMPO{A}(mpotensors::Vector)
Constructor entrance for FiniteMPO, which only supports strictly quantum number conserving operators

site tensor convention:
i mean in arrow, o means out arrow
    o 
    |
    2
o-1   3-i
	4
	|
	i
The left and right boundaries are always vacuum.
The case that the right boundary is not vacuum corresponds to operators which do not conserve quantum number, 
such as a†, this case is implemented with another MPO object.
"""
function FiniteMPO{A}(mpotensors::Vector) where {A<:MPOTensor}
	isempty(mpotensors) && error("no input mpstensors.")
	_check_mpo_space(mpotensors, strict=false)
	return new{A}(mpotensors)
end

end


"""
	The raw mpo data as a list of 4-dimension tensors
	This is not supposed to be directly used by users
"""
raw_data(h::FiniteMPO) = h.data


Base.eltype(h::FiniteMPO) = eltype(typeof(h))
Base.length(h::FiniteMPO) = length(raw_data(h))
Base.isempty(h::FiniteMPO) = isempty(raw_data(h))
Base.size(h::FiniteMPO, i...) = size(raw_data(h), i...)
Base.getindex(h::FiniteMPO, i::Int) = getindex(raw_data(h), i)
Base.lastindex(h::FiniteMPO) = lastindex(raw_data(h))
Base.firstindex(h::FiniteMPO) = firstindex(raw_data(h))

TensorKit.spacetype(::Type{FiniteMPO{A}}) where A = spacetype(A)
TensorKit.spacetype(h::FiniteMPO) = spacetype(typeof(h))

scalar_type(h::FiniteMPO) = scalar_type(typeof(h))

Base.eltype(::Type{FiniteMPO{M}}) where {M <: MPOTensor} = M
function Base.setindex!(h::FiniteMPO, v, i::Int)
	_check_mpo_tensor_dir(v) || throw(SpaceMismatch())
	return setindex!(raw_data(h), v, i)
end 
FiniteMPO(mpotensors::Vector{A}) where {A <: MPOTensor} = FiniteMPO{A}(mpotensors)
Base.copy(h::FiniteMPO) = FiniteMPO(copy(raw_data(h)))

scalar_type(::Type{FiniteMPO{M}}) where {M <: MPOTensor} = eltype(M)

space_l(state::FiniteMPO) = space(state[1], 1)
space_r(state::FiniteMPO) = space(state[end], 3)

"""
	r_RR, right boundary 2-tensor
	i-1
	o-2
"""
r_RR(state::FiniteMPO{T}) where T = isomorphism(Matrix{eltype(T)}, space_r(state), space_r(state))
"""
	l_LL, left boundary 2-tensor
	o-1
	i-2
"""
l_LL(state::FiniteMPO{T}) where T = isomorphism(Matrix{eltype(T)}, space_l(state), space_l(state))

isstrict(h::FiniteMPO) = space_r(h)' == oneunit(space_r(h))

bond_dimension(h::FiniteMPO, bond::Int) = begin
	((bond >= 1) && (bond < length(h))) || throw(BoundsError())
	dim(space(h[bond], 3))
end 
bond_dimensions(h::FiniteMPO) = [bond_dimension(h, i) for i in 1:length(h)-1]
bond_dimension(h::FiniteMPO) = maximum(bond_dimensions(h))

physical_spaces(psi::FiniteMPO) = [space(item, 2) for item in raw_data(psi)]

# function select_sector(h::FiniteMPO; sector::Sector)
# 	s_r = space_r(h)'
# 	hassector(s_r, sector) || throw(ArgumentError("sector does not exist."))
# 	m = isometry(s_r, typeof(s_r)(sector=>dim(s_r, sector))) 
# 	mpotensors = copy(raw_data(h)[1:end-1])
# 	@tensor tmp[-1 -2 ; -3 -4] := h[end][-1, -2, 1, -4] * m[1, -3]
# 	push!(mpotensors, tmp)
# 	return FiniteMPO(mpotensors)
# end

