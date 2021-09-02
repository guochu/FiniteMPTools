


struct AdjointFiniteMPO{A <: MPOTensor} <: AbstractMPO
	parent::FiniteMPO{A}
end



Base.length(h::AdjointFiniteMPO) = length(h.parent)
Base.isempty(h::AdjointFiniteMPO) = isempty(h.parent)

scalar_type(h::AdjointFiniteMPO) = scalar_type(typeof(h))


scalar_type(::Type{AdjointFiniteMPO{A}}) where {A <: MPOTensor, B} = eltype(A)



Base.adjoint(h::FiniteMPO) = AdjointFiniteMPO(h)
Base.adjoint(h::AdjointFiniteMPO) = h.parent


isstrict(h::AdjointFiniteMPO) = isstrict(h.parent)