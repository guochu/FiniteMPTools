


struct AdjointFiniteMPS{A<:MPSTensor, B<:MPSBondTensor} <: AbstractMPS
	parent::FiniteMPS{A, B}
end


# Base.eltype(psi::AdjointFiniteMPS) = eltype(typeof(psi))
Base.length(psi::AdjointFiniteMPS) = length(psi.parent)
Base.isempty(psi::AdjointFiniteMPS) = isempty(psi.parent)

scalar_type(psi::AdjointFiniteMPS) = scalar_type(typeof(psi))


# Base.eltype(::Type{AdjointFiniteMPS{A, B}}) where {A <: MPSTensor, B} = A
scalar_type(::Type{AdjointFiniteMPS{A, B}}) where {A <: MPSTensor, B} = eltype(A)



Base.adjoint(psi::FiniteMPS) = AdjointFiniteMPS(psi)
Base.adjoint(psi::AdjointFiniteMPS) = psi.parent


isstrict(psi::AdjointFiniteMPS) = isstrict(psi.parent)