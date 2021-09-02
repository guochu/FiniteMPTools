
# generic initializers

"""
	prodmpo(::Type{T}, physpaces::Vector{S}, ms::AbstractDict{Int, M}) where {T <: Number, S <: EuclideanSpace, M <: MPOTensor{S}}
	construct product mpo with chain of 4-dimensional tensors, missing points are interpretted as identity.
"""
function prodmpo(::Type{T}, physpaces::Vector{S}, ms::AbstractDict{Int, M}) where {T <: Number, S <: EuclideanSpace, M <: MPOTensor{S}}
	L = length(physpaces)
	for (k, v) in ms
		((k>= 1) && (k <= L)) || throw(BoundsError())
		(physpaces[k] == space(v, 2) == space(v, 4)') || throw(SpaceMismatch("space mismatch on site $k."))
	end
	tensor_type = tensormaptype(S, 2, 2, T)
	mpotensors = Vector{tensor_type}(undef, L)
	left = oneunit(S)
	for i in 1:L
		mpotensors[i] = convert(tensor_type, get(ms, i, id(left ⊗ physpaces[i]))) 
		left = space(mpotensors[i], 3)'
	end
	return FiniteMPO(mpotensors)
end

function _site_ops_to_dict(pos::Vector{Int}, ms::Vector)
	(length(pos) == length(ms)) || throw(DimensionMismatch())
	(length(Set(pos)) == length(pos)) || throw(ArgumentError("duplicate positions not allowed."))
	return Dict(k=>v for (k, v) in zip(pos, ms))
end

prodmpo(::Type{T}, physpaces::Vector{S}, pos::Vector{Int}, ms::Vector{M}) where {T <: Number, S <: EuclideanSpace, M <: MPOTensor{S}} = prodmpo(
	T, physpaces, _site_ops_to_dict(pos, ms))

prodmpo(physpaces::Vector{S}, ms::AbstractDict{Int, M}) where {S <: EuclideanSpace, M <: MPOTensor{S}} = prodmps(eltype(M), physpaces, ms)
prodmpo(physpaces::Vector{S}, pos::Vector{Int}, ms::Vector{M}) where {S <: EuclideanSpace, M <: MPOTensor{S}} = prodmps(eltype(M), physpaces, pos, ms)

function prodmpo(::Type{T}, L::Int, ms::AbstractDict{Int, M}) where {T <: Number, M <: MPOTensor}
	isempty(ms) && error("input should not be empty.")
	s = space(first(ms)[2], 2)
	return prodmpo(T, [s for i in 1:L], ms)
end
function prodmpo(::Type{T}, L::Int, pos::Vector{Int}, ms::Vector{M}) where {T <: Number, M <: MPOTensor}
	isempty(ms) && error("input should not be empty.")
	s = space(ms[1], 2)
	return prodmpo(T, [s for i in 1:L], pos, ms)
end
prodmpo(L::Int, ms::AbstractDict{Int, M}) where {M <: MPOTensor} = prodmpo(eltype(M), L, ms)
prodmpo(L::Int, pos::Vector{Int}, ms::Vector{M}) where {M <: MPOTensor} = prodmpo(eltype(M), L, pos, ms)


# # initializers with no symmetry

# """
# 	prodmpo(::Type{T}, physpaces::Vector{S}, ms::AbstractDict{Int, M}) where {T <: Number, S<:Union{CartesianSpace, ComplexSpace}, M <: AbstractMatrix}
# 	non-symmetric case
# """
# function prodmpo(::Type{T}, physpaces::Vector{S}, ms::AbstractDict{Int, M}) where {T <: Number, S<:Union{CartesianSpace, ComplexSpace}, M <: AbstractMatrix}
# 	tensor_type = tensormaptype(S, 2, 2, T)
# 	right = oneunit(S)
# 	return prodmpo(T, physpaces, Dict(k=>TensorMap(v, right ⊗ physpaces[k] ← right ⊗ physpaces[k]) for (k, v) in ms))
# end

# prodmpo(::Type{T}, ds::Vector{Int}, ms::AbstractDict{Int, M}) where {T <: Number, M <: AbstractMatrix} = prodmpo(T, [ℂ^d for d in ds], ms)
# prodmpo(::Type{T}, ds::Vector{Int}, pos::Vector{Int}, ms::Vector{<:AbstractMatrix}) where {T <: Number} = prodmpo(T, ds, _site_ops_to_dict(pos, ms))

# function _get_elt(ms::AbstractDict)
# 	T = Float64
# 	for (k, v) in ms
# 		T = promote_type(T, eltype(v))
# 	end
# 	return T
# end

# prodmpo(ds::Vector{Int}, ms::AbstractDict{Int, M}) where {M <: AbstractMatrix} = prodmpo(_get_elt(ms), ds, ms)
# prodmpo(ds::Vector{Int}, pos::Vector{Int}, ms::Vector{<:AbstractMatrix}) = prodmpo(ds, _site_ops_to_dict(pos, ms))

# function prodmpo(::Type{T}, L::Int, ms::AbstractDict{Int, M}) where {T <: Number, M <: AbstractMatrix}
# 	isempty(ms) && error("input should not be empty.")
# 	v = first(ms)[2]
# 	return prodmpo(T, [size(v, 1) for i in 1:L], ms)
# end
# prodmpo(::Type{T}, L::Int, pos::Vector{Int}, ms::Vector{<:AbstractMatrix}) where {T <: Number} = prodmpo(T, L, _site_ops_to_dict(pos, ms))

# prodmpo(L::Int, ms::AbstractDict{Int, M}) where {M <: AbstractMatrix} = prodmpo(_get_elt(ms), L, ms)
# prodmpo(L::Int, pos::Vector{Int}, ms::Vector{<:AbstractMatrix}) = prodmpo(L, _site_ops_to_dict(pos, ms))


# # initializers with Abelian symmetry...

