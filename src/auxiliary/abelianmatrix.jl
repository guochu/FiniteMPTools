
function _dict_scalar_type(data::AbstractDict)
	T = Float64
	for (k, v) in data
		T = promote_type(T, eltype(v))
	end
	return T	
end

function _is_abelian_sector(::Type{C}) where {C <: Sector}
	(C <: TensorKit.AbelianIrrep) && return true
	if (C <: TensorKit.ProductSector)
		symms = C.parameters[1].parameters
		for item in symms
			(item <: TensorKit.AbelianIrrep) || return false
		end
		return true
	end
	return false
end

"""
	struct AbelianMatrix{S <: EuclideanSpace, T <: Number, Ic <: Sector} 
Struct to hold matricies which are labeled with U(1) indices, note that the quantum numbers does not have to be conserved, compared to TensorMap.
This is for convenience to construct U(1) MPO tensors
"""
struct AbelianMatrix{S <: EuclideanSpace, T <: Number, Ic <: Sector} 
	physpace::S
	data::Dict{Tuple{Ic, Ic}, Matrix{T}}


function AbelianMatrix{S, T, C}(physpace::S, data::AbstractDict) where {S <: GradedSpace, T <: Number, C <: Sector}
	isdual(physpace) && throw(ArgumentError("dual space not allowed."))
	_is_abelian_sector(C) || throw(ArgumentError("Abelian sector expected."))
	(sectortype(S) == C) || throw(ArgumentError("Abelian sector expected."))
	new_data = Dict{Tuple{C, C}, Matrix{T}}()
	for (k, vo) in data
		k1 = convert(C, k[1])
		k2 = convert(C, k[2])
		v = convert(Matrix{T}, vo)
		(hassector(physpace, k1) && hassector(physpace, k2)) || throw(ArgumentError("sector does not exist."))
		((dim(physpace, k1) == size(v, 1)) && (dim(physpace, k2) == size(v, 1))) || throw(SpaceMismatch())
		if any(x->x != zero(T), v)
			new_data[(k1, k2)] = v
		end
	end
	return new{S, T, C}(physpace, new_data)
end 

end


TensorKit.space(m::AbelianMatrix) = m.physpace
raw_data(m::AbelianMatrix) = m.data

AbelianMatrix(::Type{T}, physpace::S, data::AbstractDict) where {S <: GradedSpace, T <: Number} = AbelianMatrix{S, T, sectortype(S)}(physpace, data)

Base.copy(m::AbelianMatrix) = AbelianMatrix(space(m), copy(raw_data(m)))
Base.similar(m::AbelianMatrix) = AbelianMatrix(space(m), typeof(raw_data(m))() )

scalar_type(m::AbelianMatrix) = _dict_scalar_type(raw_data(m))
AbelianMatrix(physpace::S, data::AbstractDict) where {S <: GradedSpace} = AbelianMatrix(_dict_scalar_type(data), physpace, data)

Base.:*(m::AbelianMatrix, y::Number) = AbelianMatrix(space(m), Dict(k=>v*y for (k, v) in raw_data(m)))
Base.:*(y::Number, m::AbelianMatrix) = m * y
Base.:/(m::AbelianMatrix, y::Number) = m * (1 / y)
Base.:+(m::AbelianMatrix) = m
Base.:-(m::AbelianMatrix) = m * (-1)
function Base.:+(x::AbelianMatrix{S}, y::AbelianMatrix{S}) where {S <: GradedSpace}
	(space(x) == space(y)) || throw(SpaceMismatch())
	T = promote_type(scalar_type(x), scalar_type(y))
	return AbelianMatrix(T, space(x), merge(+, raw_data(x), raw_data(y)))
end
Base.:-(x::AbelianMatrix{S}, y::AbelianMatrix{S}) where {S <: GradedSpace} = x + (-1) * y

function Base.:*(x::AbelianMatrix{S}, y::AbelianMatrix{S}) where {S <: GradedSpace}
	T = promote_type(scalar_type(x), scalar_type(y))
	Ic = sectortype(S)
	r = Dict{Tuple{Ic, Ic}, Matrix{T}}()
	for (kx, vx) in raw_data(x)
		for (ky, vy) in raw_data(y)
			if kx[2] == ky[1]
				kr = (kx[1], ky[2])
				m = get!(r, kr, zeros(T, size(vx, 1), size(vy, 2)))
				m .+= vx * vy
			end
		end
	end
	return AbelianMatrix(T, space(x), r)
end
Base.adjoint(x::AbelianMatrix) = AbelianMatrix(space(x), Dict((k[2], k[1])=>v' for (k, v) in raw_data(x)))

function Base.:(==)(x::AbelianMatrix{S}, y::AbelianMatrix{S}) where {S <: GradedSpace} 
	(space(x) == space(y)) || throw(SpaceMismatch())
	return raw_data(x) == raw_data(y)
end

function LinearAlgebra.tr(x::AbelianMatrix)
	r = zero(scalar_type(x))
	for (k, v) in x
		if k[1] == k[2]
			r += tr(v)
		end
	end
	return r
end

LinearAlgebra.dot(x::AbelianMatrix{S}, y::AbelianMatrix{S}) where {S <: GradedSpace} = tr(x' * y)
LinearAlgebra.norm(x::AbelianMatrix) = sqrt(real(dot(x, x)))


function abelian_matrix_from_dense(m::AbstractMatrix{T}) where {T <: Number}
	(size(m, 1) == size(m, 2)) || throw(ArgumentError("square matrix ecpected."))
	physpace = U1Space(i-1=>1 for i in 1:size(m, 1))
	data = Dict{Tuple{Int, Int}, Matrix{T}}()
	for i in 1:size(m, 1)
		for j in 1:size(m, 2)
			if m[i, j] != zero(T)
				data[(i-1, j-1)] = ones(1, 1) * m[i, j]
			end
		end
	end
	return AbelianMatrix(physpace, data)
end

function Base.kron(x::AbelianMatrix{S}, y::AbelianMatrix{S}) where {S <: GradedSpace}
	physpace = fuse(space(x) ⊠ space(y))
	C = sectortype(physpace)
	T = promote_type(scalar_type(x), scalar_type(y))
	data = Dict{Tuple{C, C}, Matrix{T}}()
	for (kx, vx) in raw_data(x)
		for (ky, vy) in raw_data(y)
			k = (kx[1] ⊠ ky[1], kx[2] ⊠ ky[2])
			data[k] = kron(vx, vy)
		end
	end
	return AbelianMatrix(physpace, data)
end

Base.zero(x::AbelianMatrix) = similar(x)

function Base.one(x::AbelianMatrix)
	physpace = space(x)
	data = typeof(raw_data(x))()
	T = scalar_type(x)
	for s in sectors(physpace)
		d = dim(physpace, s)
		data[(s, s)] = Matrix{T}(LinearAlgebra.I, d, d)
	end
	return AbelianMatrix(physpace, data)
end
