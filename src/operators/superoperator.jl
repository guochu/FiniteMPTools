

# The default constructor is never to be explicited by the user.
# superoperator has essentially the same data structure as a QuantumOperator
struct SuperOperatorBase{S <: EuclideanSpace, M<:MPOTensor, F}
	data::QuantumOperator{S, M}
	fuser::F

function SuperOperatorBase{S, M, F}(data::QuantumOperator, fuser::F) where {S <: EuclideanSpace, M<:MPOTensor, F}
	((fuser === ⊗) || (fuser == ⊠)) || throw(ArgumentError("fuser must be ⊗ or ⊠."))
	new{S, M, F}(data, fuser)
end

end
SuperOperatorBase(data::QuantumOperator{S, M}, fuser::F) where {S <: EuclideanSpace, M<:MPOTensor, F} = SuperOperatorBase{S, M, F}(data, fuser)
SuperOperatorBase{S, M}(fuser::F) where {S <: EuclideanSpace, M<:MPOTensor, F} = SuperOperatorBase{S, M, F}(QuantumOperator{S, M}(), fuser)

# function Base.getproperty(x::SuperOperatorBase, s::Symbol)
# 	if s == :fusers
		
# 	end
# end

scalar_type(::Type{SuperOperatorBase{S, M}}) where {S <: EuclideanSpace, M <: MPOTensor} = scalar_type(M)
scalar_type(x::SuperOperatorBase) = scalar_type(typeof(x))

TensorKit.spacetype(::Type{SuperOperatorBase{S, M}}) where {S, M} = S
TensorKit.spacetype(x::SuperOperatorBase) = spacetype(typeof(x))
TensorKit.space(x::SuperOperatorBase) = space(x.data)
TensorKit.space(x::SuperOperatorBase, i::Int) = space(x.data, i)

Base.copy(x::SuperOperatorBase) = SuperOperatorBase(copy(x.data))

Base.empty!(s::SuperOperatorBase) = empty!(s.data)
Base.isempty(s::SuperOperatorBase) = isempty(s.data)
Base.keys(s::SuperOperatorBase) = keys(s.data)
Base.similar(s::SuperOperatorBase{S, M}) where {S <: EuclideanSpace, M <: MPOTensor} = SuperOperatorBase{S, M}()
Base.length(x::SuperOperatorBase) = length(x.data)

is_constant(x::SuperOperatorBase) = is_constant(x.data)
interaction_range(x::SuperOperatorBase)  = interaction_range(x.data)

(x::SuperOperatorBase)(t::Number) = SuperOperatorBase(x.data(t), x.fuser)
Base.:*(x::SuperOperatorBase, y::Number) = SuperOperatorBase(x.data * y, x.fuser)
Base.:*(y::Number, x::SuperOperatorBase) = x * y
Base.:/(x::SuperOperatorBase, y::Number) = x * (1/y)
Base.:+(x::SuperOperatorBase) = x
Base.:-(x::SuperOperatorBase) = SuperOperatorBase(-x.data, x.fuser)
Base.:+(x::SuperOperatorBase, y::SuperOperatorBase) = begin
	(x.fuser === y.fuser) || throw(ArgumentError("fuser mismatch."))
	return SuperOperatorBase(x.data + y.data, x.fuser)
end 
Base.:-(x::SuperOperatorBase, y::SuperOperatorBase) = x + (-1) * y
shift(x::SuperOperatorBase, n::Int) = SuperOperatorBase(shift(x.data, n), x.fuser)
isstrict(x::SuperOperatorBase) = isstrict(x.data)
qterms(x::SuperOperatorBase, args...) = qterms(x.data, args...)
_expm(x::SuperOperatorBase, dt::Number) = _expm(x.data, dt)
absorb_one_bodies(h::SuperOperatorBase) = SuperOperatorBase(absorb_one_bodies(h.data), h.fuser)


"""
	superoperator(x::Vector, y::Vector; fuser=⊠)
	utility function to return x ⊗ conj(y) |ρ⟩ = x ρ y^†
	basically a copy from ⊗(x, y) or ⊠(x, y), 
"""
function superoperator(x::QTerm, y::QTerm; fuser=⊠) 
	((fuser === ⊠) || (fuser === ⊗)) || throw(ArgumentError("fuser should be ⊗ or ⊠."))
    if fuser === ⊠
        (isstrict(x) && isstrict(y)) || throw(ArgumentError("only strict QTerms allowed."))
    end	
    pos, opx, opy = _coerce_qterms(x, y)
    v = _otimes_n_a(opx, opy, fuser, right=oneunit(spacetype(opy[1]))')
	return QTerm(pos, v, coeff=coeff(x) * conj(coeff(y)))
end

TensorKit.id(x::QTerm) = QTerm(positions(x), [id(Matrix{scalar_type(x)}, oneunit(spacetype(x)) ⊗ space(m, 2)) for m in op(x)], coeff=1.)

# add new terms into SuperOperatorBase

"""
	add_unitary!(m::SuperOperatorBase, x::QTerm)
	add a term -i x ρ + i ρ x', namely -i x ⊗ conj(I) + I ⊗ conj(x)
"""
function add_unitary!(m::SuperOperatorBase, x::QTerm)
	x2 = -im * x
	iden = id(x)
	add!(m.data, superoperator(x2, iden, fuser=m.fuser) )
	add!(m.data, superoperator(iden, x2, fuser=m.fuser) )
end 

function add_unitary!(m::SuperOperatorBase, x::QuantumOperator)
	for t in qterms(x)
		add_unitary!(m, t)
	end
end

function add_dissipation!(m::SuperOperatorBase, x::QTerm)
	add!(m.data, 2*superoperator(x, x, fuser=m.fuser))
	x2 = x' * x
	iden = -id(x)
	add!(m.data, superoperator(x2, iden, fuser=m.fuser) )
	add!(m.data, superoperator(iden, x2, fuser=m.fuser))
end

function add_dissipation!(m::SuperOperatorBase, x::QuantumOperator)
	terms = qterms(x)
	for t1 in terms
		for t2 in terms
			add!(m.data, 2*superoperator(t1, t2, fuser=m.fuser))
			x2 = t2' * t1
			iden = -id(x2)
			add!(m.data, superoperator(x2, iden, fuser=m.fuser))
			add!(m.data, superoperator(iden, x2, fuser=m.fuser))
		end
	end
end

"""
	superoperator(h::QuantumOperator, f)
	-ihρ + iρ conj(h)
"""
function superoperator(h::QuantumOperator; fuser=⊠)
	terms = []
	for t in qterms(h)
		x2 = -im * t
		iden = id(t)
		push!(terms, superoperator(x2, iden, fuser=fuser))
		push!(terms, superoperator(iden, x2, fuser=fuser))
	end
	return SuperOperatorBase(QuantumOperator([terms...]), fuser)
end


