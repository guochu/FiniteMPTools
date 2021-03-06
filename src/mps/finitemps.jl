


# simply store the 3-dimensional tensors and the singular matrices on the bonds.
# It is the user's resonse to make it canonical 
# We will always use the (approximate) right canonical form, in DMRG the mixed form will be handled internally,
# after each sweep the resulting mps will be right canonical


"""
	struct FiniteMPS{A<:MPSTensor, B<:MPSBondTensor}
Finite Matrix Product States, which stores a chain of rank-3 tensors (MPS site tensors) and another chain of rank-2 tensors (MPS bond tensors)
Multiply all the site tensors will result in the global quantum state. The bond tensors are stored for convience, which could be (but not necessary)
the singular matrices on each bond. The number of bond tensors is equal to the number of site tensors plus 1 (two trivial bond tensors are added on 
the boundaries). Explicit storage of those bond tensors are useful to compute the local observables (only if they correctly correspond to the singular
matrices) and control the truncation error. the function canonocalize! is used to prepare an arbitrary MPS into right-canonical form, with the 
correct bond tensors. 
"""
struct FiniteMPS{A<:MPSTensor, B<:MPSBondTensor} <: AbstractMPS
	data::Vector{A}
	svectors::Vector{B}
end

function Base.getproperty(psi::FiniteMPS, s::Symbol)
	if s == :s
		return MPSBondView(psi)
	else
		return getfield(psi, s)
	end
end

"""
	The raw mps data as a list of 3-dimension tensors
	This is not supposed to be directly used by users
"""
raw_data(psi::FiniteMPS) = psi.data

"""
	The singular vectors are stored anyway even if the mps is not unitary.
	The raw singular vectors may not correspond to the correct Schmidt numbers
"""
raw_singular_matrices(psi::FiniteMPS) = psi.svectors

const FiniteNonSymmetricMPS{A, B} = FiniteMPS{A, B} where {A <: NonSymmetricMPSTensor, B}

Base.eltype(psi::FiniteMPS) = eltype(typeof(psi))
Base.length(psi::FiniteMPS) = length(raw_data(psi))
Base.isempty(psi::FiniteMPS) = isempty(raw_data(psi))
Base.size(psi::FiniteMPS, i...) = size(raw_data(psi), i...)
Base.getindex(psi::FiniteMPS, i::Int) = getindex(raw_data(psi), i)
Base.lastindex(psi::FiniteMPS) = lastindex(raw_data(psi))
Base.firstindex(psi::FiniteMPS) = firstindex(raw_data(psi))

Base.getindex(psi::FiniteMPS,r::AbstractRange{Int64}) = [psi[ri] for ri in r]

TensorKit.spacetype(::Type{FiniteMPS{A, B}}) where {A, B} = spacetype(A)
TensorKit.spacetype(psi::FiniteMPS) = spacetype(typeof(psi))
virtualspace(psi::FiniteMPS, n::Int) = space(psi[n], 3)'

scalar_type(psi::FiniteMPS) = scalar_type(typeof(psi))


Base.eltype(::Type{FiniteMPS{A, B}}) where {A <: MPSTensor, B} = A
scalar_type(::Type{FiniteMPS{A, B}}) where {A <: MPSTensor, B} = eltype(A)
function Base.setindex!(psi::FiniteMPS, v, i::Int)
	_check_mps_tensor_dir(v) || throw(SpaceMismatch())
	return setindex!(raw_data(psi), v, i)
end 
function Base.copy(psi::FiniteMPS)
	if svectors_uninitialized(psi)
		return FiniteMPS(copy(raw_data(psi)))
	else
		return FiniteMPS(copy(raw_data(psi)), copy(raw_singular_matrices(psi)))
	end
end

sector(psi::AbstractMPS) = first(sectors(space_r(psi)'))

function Base.cat(psiA::FiniteMPS, psiB::FiniteMPS)
	(space_r(psiA)' == space_l(psiB)) || throw(SpaceMismatch("can not cat two states with non compatible sectors."))
	return FiniteMPS(vcat(raw_data(psiA), raw_data(psiB)))
end
function Base.complex(psi::FiniteMPS)
	if scalar_type(psi) <: Real
		data = [complex(item) for item in raw_data(psi)]
		if svectors_uninitialized(psi)
			return FiniteMPS(data)
		else
			return FiniteMPS(data, raw_singular_matrices(psi))
		end
	end
	return psi
end

function _svectors_uninitialized(x)
	isempty(x) && return true
	s = raw_singular_matrices(x)
	for i in 1:length(s)
		isassigned(s , i) || return true
	end
	return false
end
svectors_uninitialized(psi::FiniteMPS) = _svectors_uninitialized(psi)
svectors_initialized(psi::FiniteMPS) = !svectors_uninitialized(psi)

_check_mps_tensor_dir(m::MPSTensor) = (!isdual(space(m, 1))) && (!isdual(space(m, 2))) && isdual(space(m, 3))
_check_mps_bond_dir(m::MPSBondTensor) = (!isdual(space(m, 1))) && isdual(space(m, 2))

function _check_mps_space(mpstensors::Vector; strict::Bool=true)
	all(_check_mps_tensor_dir.(mpstensors)) || throw(SpaceMismatch())
	for i in 1:length(mpstensors)-1
		(space(mpstensors[i], 3) == space(mpstensors[i+1], 1)') || throw(SpaceMismatch())
	end

	# just require the left boundary to be a single sector
	(dim(space(mpstensors[1], 1)) == 1) || throw(SpaceMismatch("left boundary should be a single sector."))
	# require the right boundary has dimension 1.
	if strict
		(dim(space(mpstensors[end], 3)) == 1) || throw(SpaceMismatch())
	end
	return true
end

function singular_matrix_type(::Type{A}) where {S<:EuclideanSpace, A <: MPSTensor{S}}
	T = real(eltype(A))
	return tensormaptype(S, 1, 1, T)
end

"""
	FiniteMPS{A, B}(mpstensors::Vector)

Entrance constructor for finite MPS
site tensor convention:
i mean in arrow, o means out arrow
    o 
    |
o-1 2 3-i
The quantum number follows from right to left.	
We only allow mps is be strict, namey the left boundary and the right boundary has a single sector with dimension 1).
The left boundary in nost of the cases is chosen to be the vacuum.
"""
function FiniteMPS{A, B}(mpstensors::Vector) where {A<:MPSTensor, B<:MPSBondTensor}
	isempty(mpstensors) && error("no input mpstensors.")
	_check_mps_space(mpstensors, strict=false)
	S = spacetype(A)
	T = real(eltype(A))
	return FiniteMPS{A, B}(mpstensors, Vector{singular_matrix_type(A)}(undef, length(mpstensors)+1))
end

"""
	FiniteMPS(mpstensors::Vector{A}) where {A <: MPSTensor}
construct finite MPS from a list of site tensors.
"""
FiniteMPS(mpstensors::Vector{A}) where {A <: MPSTensor} = FiniteMPS{A, singular_matrix_type(A)}(mpstensors)

space_l(state::FiniteMPS) = space(state[1], 1)
space_r(state::FiniteMPS) = space(state[end], 3)


r_RR(psiA::FiniteMPS, psiB::FiniteMPS) = isomorphism(Matrix{promote_type(scalar_type(psiA), scalar_type(psiB))}, space_r(psiA), space_r(psiB))
l_LL(psiA::FiniteMPS, psiB::FiniteMPS) = isomorphism(Matrix{promote_type(scalar_type(psiA), scalar_type(psiB))}, space_l(psiA), space_l(psiB))


"""
	r_RR, right boundary 2-tensor
	i-1
	o-2
"""
r_RR(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)}, space_r(state), space_r(state))
"""
	l_LL, left boundary 2-tensor
	o-1
	i-2
"""
l_LL(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)}, space_l(state), space_l(state))

isstrict(psi::FiniteMPS) = dim(space_r(psi)) == 1

function maybe_init_boundary_s!(psi::FiniteMPS)
	isempty(psi) && return
	L = length(psi)
	if !isassigned(raw_singular_matrices(psi), 1)
		# (dim(space(psi[1], 1)) == 1) || throw(SpaceMismatch())
		psi.s[1] = id(space_l(psi))
	end
	if !isassigned(raw_singular_matrices(psi), L+1)
		psi.s[L+1] = id(space_r(psi)')
	end
end

function _is_left_canonical(psij::MPSTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[1,2,-1]) * psij[1,2,-2]
	return isapprox(r, one(r); kwargs...) 
end

function _is_right_canonical(psij::MPSTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[-1,1,2]) * psij[-2,1,2]
	return isapprox(r, one(r); kwargs...) 
end

function is_right_canonical(psi::FiniteMPS; kwargs...)
	all([_is_right_canonical(item; kwargs...) for item in raw_data(psi)]) || return false
	# we also check whether the singular vectors are the correct Schmidt numbers
	svectors_uninitialized(psi) && return false
	hold = l_LL(psi)
	for i in 1:length(psi)-1
		hold = updateleft(hold, psi[i], psi[i])
		@tensor tmp[-1; -2] := conj(psi.s[i+1][1, -1]) * psi.s[i+1][1, -2]
		isapprox(hold, tmp; kwargs...) || return false
	end
	return true
end

"""
	iscanonical(psi::FiniteMPS; kwargs...) = is_right_canonical(psi; kwargs...)
check if the state is right-canonical, the singular vectors are also checked that whether there are the correct Schmidt numbers or not
This form is useful for time evolution for stability issue and also efficient for computing observers of unitary systems
"""
iscanonical(psi::FiniteMPS; kwargs...) = is_right_canonical(psi; kwargs...)

"""
	bond_dimension(psi::FiniteMPS[, bond::Int])
	bond_dimension(h::FiniteMPO[, bond::Int])
return bond dimension at the given bond, or return the largest bond dimension of all bonds.
"""
bond_dimension(psi::FiniteMPS, bond::Int) = begin
	((bond >= 1) && (bond < length(psi))) || throw(BoundsError())
	dim(space(psi[bond], 3))
end 
bond_dimensions(psi::FiniteMPS) = [bond_dimension(psi, i) for i in 1:length(psi)-1]
bond_dimension(psi::FiniteMPS) = maximum(bond_dimensions(psi))

"""
	physical_spaces(psi::FiniteMPS)
	physical_spaces(psi::FiniteMPO) 
Return all the physical spaces of FiniteMPS or FiniteMPO
"""
physical_spaces(psi::FiniteMPS) = [space(item, 2) for item in raw_data(psi)]

function max_virtual_spaces(physpaces::Vector{S}, sector::Sector) where {S <: EuclideanSpace}
	(sectortype(S) == typeof(sector)) || throw(SpaceMismatch())
	L = length(physpaces)
	left = oneunit(S)
	right = S(sector=>1)
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = fuse(virtualpaces[i-1], physpaces[i-1])
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(physpaces[i]', virtualpaces[i+1]))
	end
	return virtualpaces
end

function max_bond_dimensions(physpaces::Vector{S}, sector::Sector) where {S <: EuclideanSpace}
	(sectortype(S) == typeof(sector)) || throw(SpaceMismatch())
	L = length(physpaces)
	left = oneunit(S)
	right = S(sector=>1)
	Ds = [1 for i in 1:L+1]
	for i in 1:L
		left = fuse(left, physpaces[i])
		Ds[i+1] = dim(left)
	end
	Ds[end] = 1
	for i in L:-1:1
		right = fuse(physpaces[i]', right)
		Ds[i] = min(Ds[i], dim(right))
	end
	return Ds
end
# for general MPS this is the only we can do
function max_bond_dimensions(psi::FiniteMPS)
	isstrict(psi) || throw(ArgumentError("only strict mps allowed."))
	physpaces = [space(item, 2) for item in raw_data(psi)]
	sector = first(sectors(space_r(psi)'))
	return max_bond_dimensions(physpaces, sector)
end

function max_bond_dimensions(physpaces::Vector{Int}, D::Int) 
	L = length(physpaces)
	left = 1
	right = 1
	virtualpaces = Vector{Int}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = min(virtualpaces[i-1] * physpaces[i-1], D)
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = min(virtualpaces[i], physpaces[i] * virtualpaces[i+1])
	end
	return virtualpaces
end
max_bond_dimensions(psi::FiniteNonSymmetricMPS, D::Int) = max_bond_dimensions(dim.(physical_spaces(psi)), D)

"""
	an easy way to fill zeros to increase bond in the nonsymmetric case
"""
function increase_bond!(psi::FiniteNonSymmetricMPS; D::Int)
	S = spacetype(psi)
	T = scalar_type(psi)
	if bond_dimension(psi) < D
		virtualpaces = max_bond_dimensions(psi, D)
		for i in 1:length(psi)
			psij = psi[i][]
			sl = max(min(virtualpaces[i], D), size(psij, 1))
			sr = max(min(virtualpaces[i+1], D), size(psij, 3))
			m = zeros(T, sl, size(psij, 2), sr)
			m[1:size(psij, 1), :, 1:size(psij, 3)] .= psij
			psi[i] = TensorMap(m, S(Trivial()=>sl) ??? space(psi[i], 2), S(Trivial()=>sr))
		end
	end
	return psi
end

