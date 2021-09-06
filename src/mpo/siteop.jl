

# convenient wrapper for single site operators

abstract type AbstractSiteOperator end

TensorKit.spacetype(m::AbstractSiteOperator) = spacetype(typeof(m))
Base.eltype(m::AbstractSiteOperator) = eltype(typeof(m))

raw_data(m::AbstractSiteOperator) = m.data

Base.:(==)(x::M, y::M) where {M<:AbstractSiteOperator} = raw_data(x) == raw_data(y)
Base.isapprox(x::M, y::M; kwargs...) where {M<:AbstractSiteOperator} = isapprox(raw_data(x), raw_data(y); kwargs...)

function Base.getproperty(m::AbstractSiteOperator, s::Symbol)
	if s == :op
		return m.data
	else
		return getfield(m, s)
	end
end

struct ScalarSiteOp{M<:MPSBondTensor} <: AbstractSiteOperator
	data::M
end

TensorKit.spacetype(::Type{ScalarSiteOp{M}}) where M = spacetype(M)
Base.eltype(::Type{ScalarSiteOp{M}}) where M = eltype(M)
physical_space(m::ScalarSiteOp) = space(raw_data(m), 1)


function Base.:+(x::ScalarSiteOp, y::ScalarSiteOp)
	(spacetype(x) == spacetype(y)) || throw(SpaceMismatch())
	return ScalarSiteOp(raw_data(x) + raw_data(y))
end
Base.:*(x::ScalarSiteOp, y::Number) = ScalarSiteOp(raw_data(x) * y)
Base.:*(y::Number, x::ScalarSiteOp) = x * y
Base.:/(x::ScalarSiteOp, y::Number) = SiteOp(raw_data(x) / y)
Base.:-(x::ScalarSiteOp, y::ScalarSiteOp) = x + (-y)


function Base.:*(x::ScalarSiteOp, y::ScalarSiteOp)
	(spacetype(x) == spacetype(y)) || throw(SpaceMismatch())
	return ScalarSiteOp(raw_data(x) * raw_data(y))
end

"""	
	a scalar operator has to be hermitian?
"""
Base.adjoint(x::ScalarSiteOp) = ScalarSiteOp(raw_data(x)')


struct SiteOp{M <: MPOTensor} <: AbstractSiteOperator
	data::M

function SiteOp{M}(m::M) where {M <: MPOTensor}
	_check_mpo_tensor_dir(m) || throw(SpaceMismatch())
	(space(m, 1) == oneunit(space(m, 1))) || throw(ArgumentError("the left auxiliary index should be trivial."))
	new{M}(m)
end

end

function _remove_trivial(m::MPOTensor)
	util = get_trivial_leg(m)
	@tensor out[-1; -2] := conj(util[1]) * m[1, -1, 2, -2] * util[2]
	return out
end

SiteOp(m::MPOTensor) = SiteOp{typeof(m)}(m)


ScalarSiteOp(m::MPOTensor) = ScalarSiteOp(_remove_trivial(m))
"""
	ScalarSiteOp(m::SiteOp)
"""
ScalarSiteOp(m::SiteOp) = ScalarSiteOp(raw_data(m))


# function SiteOp(m::AbstractTensorMap{S, 1, 1}; right=oneunit(S)) where {S <: EuclideanSpace}
# 	bh = id(Matrix{eltype(m)}, right)
# 	@tensor out[-1 -2; -3 -4] := bh[-1, -3] * m[-2, -4]
# 	return SiteOp(out)
# end


TensorKit.spacetype(::Type{SiteOp{M}}) where M = spacetype(M)
TensorKit.space(x::SiteOp) = space(raw_data(x))
TensorKit.space(x::SiteOp, i::Int) = space(raw_data(x), i)
Base.eltype(::Type{SiteOp{M}}) where M = eltype(M)
physical_space(m::SiteOp) = space(raw_data(m), 2)
space_l(m::SiteOp) = space(raw_data(m), 1)
space_r(m::SiteOp) = space(raw_data(m), 3)

isstrict(m::SiteOp) = space_l(m) == oneunit(spacetype(m)) == space_r(m)'

function Base.:+(x::SiteOp, y::SiteOp)
	(spacetype(x) == spacetype(y)) || throw(SpaceMismatch())
	embed_r = right_embedders(promote_type(eltype(x), eltype(y)), space_r(x)', space_r(y)')
	@tensor out[-1 -2; -3 -4] := raw_data(x)[-1, -2, 1, -4] * embed_r[1][1, -3]
	@tensor out[-1, -2, -3, -4] += raw_data(y)[-1, -2, 1, -4] * embed_r[2][1, -3]
	return SiteOp(out)
end
Base.:*(x::SiteOp, y::Number) = SiteOp(raw_data(x) * y)
Base.:*(y::Number, x::SiteOp) = x * y
Base.:/(x::SiteOp, y::Number) = SiteOp(raw_data(x) / y)
Base.:-(x::SiteOp, y::SiteOp) = x + (-y)


function Base.:*(x::SiteOp, y::SiteOp)
	(spacetype(x) == spacetype(y)) || throw(SpaceMismatch())
	T = promote_type(eltype(x), eltype(y))
	embed_l = isomorphism(Matrix{T}, fuse(space_l(x), space_l(y)), space_l(x) ⊗ space_l(y) )
	embed_r = isomorphism(Matrix{T}, space_r(x)' ⊗ space_r(y)', fuse(space_r(x)', space_r(y)') )

	@tensor out[-1 -2; -3 -4] := embed_l[-1, 1, 2] * raw_data(x)[1,-2,3,4] * raw_data(y)[2,4,5,-4] * embed_r[3,5,-3]
	return SiteOp(out)
end


function Base.:*(x::SiteOp, y::ScalarSiteOp)
	(spacetype(x) == spacetype(y)) || throw(SpaceMismatch())
	@tensor out[-1 -2; -3 -4] := raw_data(x)[-1, -2, -3, 1] * raw_data(y)[1, -4]
	return SiteOp(out)
end
function Base.:*(x::ScalarSiteOp, y::SiteOp)
	(spacetype(x) == spacetype(y)) || throw(SpaceMismatch())
	@tensor out[-1 -2; -3 -4] := raw_data(x)[-2, 1] * raw_data(y)[-1, 1, -3, -4]
	return SiteOp(out)
end



struct AdjointSiteOp{M <: MPOTensor} <: AbstractSiteOperator
	parent::SiteOp{M}
end

function Base.getproperty(m::AdjointSiteOp, s::Symbol)
	if s == :op
		return raw_data(m)
	else
		return getfield(m, s)
	end
end

Base.adjoint(m::SiteOp) = AdjointSiteOp(m)
Base.adjoint(m::AdjointSiteOp) = m.parent

TensorKit.spacetype(::Type{AdjointSiteOp{M}}) where M = spacetype(M)
Base.eltype(::Type{AdjointSiteOp{M}}) where M = eltype(M)

raw_data(m::AdjointSiteOp) = adjoint(raw_data(m.parent))

Base.:+(x::AdjointSiteOp, y::AdjointSiteOp) = adjoint(x.parent + y.parent)
Base.:-(x::AdjointSiteOp, y::AdjointSiteOp) = adjoint(x.parent - y.parent)
Base.:*(x::AdjointSiteOp, y::Number) = adjoint(x.parent * conj(y))
Base.:*(y::Number, x::AdjointSiteOp) = x * y
Base.:/(x::AdjointSiteOp, y::Number) = adjoint(x.parent / conj(y))
Base.:*(x::AdjointSiteOp, y::AdjointSiteOp) = adjoint(y.parent * x.parent)

Base.:(==)(x::AdjointSiteOp, y::AdjointSiteOp) = x.parent == y.parent
Base.isapprox(x::AdjointSiteOp, y::AdjointSiteOp; kwargs...) = isapprox(x.parent, y.parent; kwargs...)


function dot_prod(x::AdjointSiteOp, y::SiteOp; left::EuclideanSpace, right::EuclideanSpace)
	xadj = adjoint(raw_data(x.parent))
	fuser_l = isomorphism(left ⊗ space_l(y)', space(xadj, 3))
	fuser_r = isomorphism(right' ⊗ space_r(y)', space(xadj, 1))
	@tensor out[-1 -2; -3 -4] := fuser_l[-1, 1, 2] * xadj[3, -2, 2, 4] * raw_data(y)[1, 4, 5, -4] * fuser_r[-3, 5, 3]
	return SiteOp(out)
end

Base.:*(x::AdjointSiteOp, y::SiteOp) = dot_prod(x, y, left=oneunit(spacetype(x)), right=oneunit(spacetype(y)))

function dot_prod(x::SiteOp, y::AdjointSiteOp; left::EuclideanSpace, right::EuclideanSpace)
	yadj = adjoint(raw_data(y.parent))
	fuser_l = isomorphism(left ⊗ space(yadj, 3)', space_l(x))
	fuser_r = isomorphism(right' ⊗ space(yadj, 1)', space_r(x))
	@tensor out[-1 -2; -3 -4] := fuser_l[-1, 1, 2] * raw_data(x)[2, -2, 3, 4] * yadj[5, 4, 1, -4] * fuser_r[-3, 5, 3]
	return SiteOp(out)
end
Base.:*(x::SiteOp, y::AdjointSiteOp) = dot_prod(x, y, left=oneunit(spacetype(x)), right=oneunit(spacetype(x)))


# for non-symmetric case
_to_tensor_map(m::AbstractMatrix) = TensorMap(m, ℂ^(size(m, 1)) ← ℂ^(size(m, 1)) )
SiteOp(m::AbstractMatrix) = SiteOp(_add_legs(_to_tensor_map(m)))
ScalarSiteOp(m::AbstractMatrix) = ScalarSiteOp(_to_tensor_map(m))

