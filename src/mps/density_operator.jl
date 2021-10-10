
# density operator is treated as a specialized MPS by fusing the two physical indices
# a fuser is a 3-d tensor from two physical indices into a single index
# fuser is used to go back to an MPO 
# I is ⟨I| which is used to compute expectations
# Thus default constructor should never be directly used by users

"""
	struct FiniteDensityOperatorMPS{A<:MPSTensor, B<:MPSBondTensor}
"""
struct FiniteDensityOperatorMPS{A<:MPSTensor, B<:MPSBondTensor} <: AbstractMPS
	data::FiniteMPS{A, B}
	fusers::Vector{A}
	I::FiniteMPS{A, B}
end

function Base.getproperty(psi::FiniteDensityOperatorMPS, s::Symbol)
	if s == :s
		return psi.data.s
	else
		return getfield(psi, s)
	end
end


Base.eltype(psi::FiniteDensityOperatorMPS) = eltype(typeof(psi))
Base.length(psi::FiniteDensityOperatorMPS) = length(psi.data)
Base.isempty(psi::FiniteDensityOperatorMPS) = isempty(psi.data)
Base.size(psi::FiniteDensityOperatorMPS, i...) = size(psi.data, i...)
Base.getindex(psi::FiniteDensityOperatorMPS, i::Int) = getindex(psi.data, i)
Base.lastindex(psi::FiniteDensityOperatorMPS) = lastindex(psi.data)
Base.firstindex(psi::FiniteDensityOperatorMPS) = firstindex(psi.data)

Base.getindex(psi::FiniteDensityOperatorMPS,r::AbstractRange{Int64}) = [psi[ri] for ri in r]

TensorKit.spacetype(::Type{FiniteDensityOperatorMPS{A, B}}) where {A, B} = spacetype(A)
TensorKit.spacetype(psi::FiniteDensityOperatorMPS) = spacetype(typeof(psi))
virtualspace(psi::FiniteDensityOperatorMPS, n::Int) = space(psi[n], 3)'

scalar_type(psi::FiniteDensityOperatorMPS) = scalar_type(typeof(psi))


Base.eltype(::Type{FiniteDensityOperatorMPS{A, B}}) where {A, B} = A
scalar_type(::Type{FiniteDensityOperatorMPS{A, B}}) where {A, B} = eltype(A)
Base.setindex!(psi::FiniteDensityOperatorMPS, v, i::Int) = setindex!(psi.data, v, i)
Base.copy(psi::FiniteDensityOperatorMPS) = FiniteDensityOperatorMPS(copy(psi.data), psi.fusers, psi.I)

bond_dimension(psi::FiniteDensityOperatorMPS, bond::Int) = bond_dimension(psi.data, bond)
bond_dimensions(psi::FiniteDensityOperatorMPS) = bond_dimensions(psi.data)
bond_dimension(psi::FiniteDensityOperatorMPS) = bond_dimension(psi.data)

physical_spaces(psi::FiniteDensityOperatorMPS) = physical_spaces(psi.data)
space_l(psi::FiniteDensityOperatorMPS) = space_l(psi.data)
space_r(psi::FiniteDensityOperatorMPS) = space_r(psi.data)

sector(psi::FiniteDensityOperatorMPS) = sector(psi.data)

LinearAlgebra.tr(psi::FiniteDensityOperatorMPS) = dot(psi.I, psi.data)

canonicalize!(psi::FiniteDensityOperatorMPS; kwargs...) = canonicalize!(psi.data; kwargs...)
