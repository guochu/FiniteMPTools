



struct SchurMPOTensor{S<:EuclideanSpace, M<:MPOTensor, T<:Number} <: AbstractSiteMPOTensor{S}
	Os::Array{Union{M, T}, 2}
	imspaces::Vector{S}
	omspaces::Vector{S}
	pspace::S
end

Base.copy(x::SchurMPOTensor) = SchurMPOTensor(copy(x.Os), copy(x.imspaces), copy(x.omspaces), x.pspace)

# upper triangular form
# the middle diagonal terms may be identity operator or the JW operator,
function SchurMPOTensor{S, M, T}(data::Array{E, 2}) where {S<:EuclideanSpace, M <:MPOTensor, T<:Number, E}
	# (isa(data[1, 1], MPOTensor) && isa(data[end, end], MPOTensor)) || throw(ArgumentError("upper left and lower right corner should be identity tensors."))
	m, n = size(data)
	(m >= 1 && n >= 1 && m==n) || throw(DimensionMismatch())
	new_data = Array{Union{M, T}, 2}(undef, m, n) 
	omspaces = Vector{Union{Missing, S}}(missing, n)
	imspaces = Vector{Union{Missing, S}}(missing, m)
	pspace = nothing
	for i in 1:m
		for j in 1:n
			sj = data[i, j]
			if isa(sj, MPSBondTensor)
				sj = _add_legs(sj)
			end

			# println(typeof(sj))

 			if isa(sj, MPOTensor)
 				sj = convert(M, sj)
 				s_l = space(sj, 1)
 				s_r = space(sj, 3)'
 				s_p = space(sj, 2)
 				(s_p == space(sj, 4)') || throw(SpaceMismatch())
 				if !ismissing(imspaces[i])
 					# (imspaces[i] == s_l) || println(imspaces[i], s_l)
 					(imspaces[i] == s_l) || throw(SpaceMismatch())
 				else
 					imspaces[i] = s_l
 				end
 				if !ismissing(omspaces[j])
 					(omspaces[j] == s_r) || throw(SpaceMismatch())
 				else
 					omspaces[j] = s_r
 				end
 				if isnothing(pspace)
 					pspace = s_p
 				else
 					(pspace == s_p) || throw(SpaceMismatch())
 				end
 				_is_id, scal = isid(sj)
 				if _is_id
 					sj = scal
 				end
 			else
 				isa(sj, Number) || throw(ArgumentError("elt should either be a tensor or a scalar."))
 			end

			if i > j
				(sj == zero(T)) || throw(ArgumentError("Canonical MPO Tensor is upper triangular."))
				# new_data[i, j] = zero(T)
			# elseif i == j
			# 	isa(sj, Number) || throw(ArgumentError("diagonals should either be scalars."))
 			end

 			new_data[i, j] = sj
		end
	end
	for i in 1:m
		ismissing(imspaces[i]) && throw(ArgumentError("imspace $i not assigned."))
	end
	for j in 1:n
		ismissing(omspaces[j]) && throw(ArgumentError("omspace $j not assigned."))
	end
	return SchurMPOTensor{S, M, T}(new_data, [imspaces...], [omspaces...], pspace)
end
SchurMPOTensor(data::Array{Union{M, T}, 2}) where {M <:MPOTensor, T<:Number} = SchurMPOTensor{spacetype(M), M, T}(data)

"""
	SchurMPOTensor(data::Array{Any, 2}) 
"""
function SchurMPOTensor(data::Array{Any, 2}) 
	m, n = size(data)
	S = nothing
	T = Float64
	for sj in data
		if isa(sj, Number)
			T = promote_type(T, typeof(sj))
		elseif isa(sj, AbstractTensorMap)
			T = promote_type(T, eltype(sj))
			if isnothing(S)
				S = spacetype(sj)
			else
				(S == spacetype(sj)) || throw(SpaceMismatch())
			end
		else
			throw(ArgumentError("eltype must be scalar or TensorMap."))
		end		
	end
	isnothing(S) && throw(ArgumentError("there must be at least one TensorMap type."))
	M = tensormaptype(S, 2, 2, T)
	return SchurMPOTensor{S, M, T}(data)
end


Base.contains(m::SchurMPOTensor{S, M, T}, i::Int, j::Int) where {S, M, T} = (i <= j) && !(m.Os[i, j] == zero(T))
isscal(x::SchurMPOTensor{S,M,T}, i::Int, j::Int) where {S,M,T} = x.Os[i, j] isa T && contains(x,i,j)
scalar_type(::Type{SchurMPOTensor{S,M,T}}) where {S,M,T} = T

function Base.setindex!(m::SchurMPOTensor{S, M, T}, v, i::Int, j::Int) where {S, M, T}
	(i > j) && throw(ArgumentError("not allowed to set the low triangular portion."))
	if isa(v, Number)
		m.Os[i, j] = convert(T, v)
	elseif isa(v, MPOTensor)
		((space(v, 1) == m.imspaces[i]) && (space(v, 3)' == m.omspaces[j])) || throw(SpaceMismatch())
		m.Os[i, j] = convert(M, v) 
	elseif isa(v, MPSBondTensor)
		b_iden = isomorphism(Matrix{T}, m.imspaces[i], m.omspaces[j])
		@tensor tmp[-1 -2; -3 -4] := b_iden[-1, -3] * v[-2, -4]
		m.Os[i, j] = tmp
	else
		throw(ArgumentError("input should be scalar, MPOTensor or MPSBondTensor type."))
	end
end
