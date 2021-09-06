


struct AdjointFiniteMPO{A <: MPOTensor} <: AbstractMPO
	parent::FiniteMPO{A}
end



Base.length(h::AdjointFiniteMPO) = length(h.parent)
Base.isempty(h::AdjointFiniteMPO) = isempty(h.parent)

scalar_type(h::AdjointFiniteMPO) = scalar_type(typeof(h))


scalar_type(::Type{AdjointFiniteMPO{A}}) where {A <: MPOTensor} = eltype(A)



Base.adjoint(h::FiniteMPO) = AdjointFiniteMPO(h)
Base.adjoint(h::AdjointFiniteMPO) = h.parent


isstrict(h::AdjointFiniteMPO) = isstrict(h.parent)



struct ConjugateFiniteMPO{A <: MPOTensor} <: AbstractMPO
	parent::FiniteMPO{A}
end


Base.length(h::ConjugateFiniteMPO) = length(h.parent)
Base.isempty(h::ConjugateFiniteMPO) = isempty(h.parent)

scalar_type(h::ConjugateFiniteMPO) = scalar_type(typeof(h))


scalar_type(::Type{ConjugateFiniteMPO{A}}) where {A <: MPOTensor} = eltype(A)



Base.conj(h::FiniteMPO) = ConjugateFiniteMPO(h)
Base.conj(h::ConjugateFiniteMPO) = h.parent


isstrict(h::ConjugateFiniteMPO) = isstrict(h.parent)
