


struct ExpoMPOTensor{S<:EuclideanSpace, M<:MPOTensor, T<:Number} <: AbstractSiteMPOTensor{S}
	Os::Array{Union{M, T}, 2}
	imspaces::Vector{S}
	omspaces::Vector{S}
	pspace::S
end


Base.copy(x::ExpoMPOTensor) = ExpoMPOTensor(copy(x.Os), copy(x.imspaces), copy(x.omspaces), x.pspace)

function ExpoMPOTensor{S, M, T}(data::Array{E, 2}) where {S<:EuclideanSpace, M <:MPOTensor, T<:Number, E}
	# (isa(data[1, 1], MPOTensor) && isa(data[end, end], MPOTensor)) || throw(ArgumentError("upper left and lower right corner should be identity tensors."))
	m, n = size(data)
	(m >= 1 && n >= 1 && m==n) || throw(DimensionMismatch())
	new_data = Array{Union{M, T}, 2}(undef, m, n) 
	omspaces = Vector{S}(undef, n)
	imspaces = Vector{S}(undef, m)
	pspace = nothing
	for i in 1:m
		for j in 1:n
			sj = data[i, j]
			if isa(sj, MPSBondTensor)
				sj = _add_legs(sj)
			end

 			if isa(sj, MPOTensor)
 				sj = convert(M, sj)
 				s_l = space(sj, 1)
 				s_r = space(sj, 3)'
 				s_p = space(sj, 2)
 				(s_p == space(sj, 4)') || throw(SpaceMismatch())
 				if isassigned(imspaces, i)
 					(imspaces[i] == s_l) || throw(SpaceMismatch())
 				else
 					imspaces[i] = s_l
 				end
 				if isassigned(omspaces, j)
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

 			new_data[i, j] = sj
		end
	end
	for i in 1:m
		isassigned(imspaces, i) || throw(ArgumentError("imspace $i not assigned."))
	end
	for j in 1:n
		isassigned(omspaces, j) || throw(ArgumentError("omspace $j not assigned."))
	end
	return ExpoMPOTensor{S, M, T}(new_data, imspaces, omspaces, pspace)
end
ExpoMPOTensor(data::Array{Union{M, T}, 2}) where {M <:MPOTensor, T<:Number} = ExpoMPOTensor{spacetype(M), M, T}(data)

"""
	ExpoMPOTensor(data::Array{Any, 2}) 
"""
function ExpoMPOTensor(data::Array{Any, 2}) 
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
	return ExpoMPOTensor{S, M, T}(data)
end

Base.contains(m::ExpoMPOTensor{S, M, T}, i::Int, j::Int) where {S, M, T} = (i <= j) && !(m.Os[i, j] == zero(T))
isscal(x::ExpoMPOTensor{S,M,T}, i::Int, j::Int) where {S,M,T} = x.Os[i, j] isa T && contains(x,i,j)
scalar_type(::Type{ExpoMPOTensor{S,M,T}}) where {S,M,T} = T



