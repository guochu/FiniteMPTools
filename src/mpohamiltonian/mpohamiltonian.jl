


struct MPOHamiltonian{S<:EuclideanSpace, M<:MPOTensor, T<:Number}
	data::PeriodicArray{SchurMPOTensor{S, M, T}}
end

raw_data(x::MPOHamiltonian) = x.data
period(h::MPOHamiltonian) = size(raw_data(h), 1)
Base.getindex(x::MPOHamiltonian, i::Int) = getindex(raw_data(x), i)
Base.copy(x::MPOHamiltonian) = MPOHamiltonian(copy(raw_data(x)))

TensorKit.spacetype(::Type{MPOHamiltonian{S, M, T}}) where {S, M, T} = S
TensorKit.spacetype(x::MPOHamiltonian) = spacetype(typeof(x))
scalar_type(::Type{MPOHamiltonian{S, M, T}}) where {S, M, T} = T
scalar_type(x::MPOHamiltonian) = scalar_type(typeof(x))

function MPOHamiltonian(data::AbstractVector{<:SchurMPOTensor})
	isempty(data) && throw(ArgumentError("empty data."))
	(data[1].imspaces == data[end].omspaces) || throw(SpaceMismatch())
	d = odim(data[1])
	for i in 2:length(data)
		(d == odim(data[i])) || throw(DimensionMismatch("odim mismatch."))
	end
	return MPOHamiltonian(PeriodicArray(data))
end 
MPOHamiltonian(data::SchurMPOTensor) = MPOHamiltonian([data])

function Base.getindex(x::MPOHamiltonian{S, M, T}, i::Int, j::Int, k::Int)::M where {S, M, T}
	x[i][j, k]
end 

Base.getindex(x::MPOHamiltonian, i::Colon, j::Int, k::Int) = [getindex(x, i,j,k) for i in 1:period(x)]

# isscal(x::MPOHamiltonian, a::Int, b::Int, c::Int) = isscal(x[a], b, c)

"""
checks if ham[:,i,i] = 1 for every i
"""
function isid(ham::MPOHamiltonian{S,M,T},i::Int) where {S,M,T}
	r = true
	for b in 1:period(ham)
		r = r && isscal(ham[b],i,i) && abs(ham[b].Os[i,i]-one(T))<1e-14
	end
	return r
end 


space_l(x::MPOHamiltonian) = raw_data(x)[1].imspaces[1]
space_r(x::MPOHamiltonian) = (raw_data(x)[end].omspaces[end])'
odim(x::MPOHamiltonian) = odim(x[1])

# """
# 	r_RR, right boundary 2-tensor
# 	i-1
# 	o-2
# """
# r_RR(state::MPOHamiltonian) = isomorphism(scalar_type(state), space_r(state), space_r(state))
# """
# 	l_LL, left boundary 2-tensor
# 	o-1
# 	i-2
# """
# l_LL(state::MPOHamiltonian) = isomorphism(scalar_type(state), space_l(state), space_l(state))

physical_spaces(x::MPOHamiltonian) = PeriodicArray([x[i].pspace for i in 1:period(x)])

function isstrict(x::MPOHamiltonian)
	v = oneunit(spacetype(x))
	return (space_l(x) == v) && (space_r(x)' == v)
end

function right_embedders(::Type{T}, a::S...) where {T <: Number, S <: EuclideanSpace}
    V = ⊕(a...) 
    ts = [TensorMap(zeros, T, aj, V) for aj in a]
    for c in sectors(V)
    	n = 0
    	for i in 1:length(ts)
    		ni = dim(a[i], c)
    		block(ts[i], c)[:, (n+1):(n+ni)] .= LinearAlgebra.Diagonal( ones(ni, 2) )
    		n += ni
    	end
    end
    return ts
end

function left_embedders(::Type{T}, a::S...) where {T <: Number, S <: EuclideanSpace}
    V = ⊕(a...) 
    ts = [TensorMap(zeros, T, V, aj) for aj in a]
    for c in sectors(V)
    	n = 0
    	for i in 1:length(ts)
    		ni = dim(a[i], c)
    		block(ts[i], c)[(n+1):(n+ni), :] .= LinearAlgebra.Diagonal( ones(ni, 2) )
    		n += ni
    	end
    end
    return ts	
end

function FiniteMPO(h::MPOHamiltonian{S, M, T}, L::Int) where {S, M, T}
	(mod(L, period(h)) == 0) || throw(DimensionMismatch())
	mpotensors = Vector{M}(undef, L)
	embedders = PeriodicArray([right_embedders(T, h[i].omspaces...) for i in 1:period(h)])

	tmp = TensorMap(zeros, T, oneunit(S)*h[1].pspace ← space(embedders[1][1], 2)' * h[1].pspace )
	for i in 1:length(embedders[1])
		@tensor tmp[-1, -2, -3, -4] += h[1, 1, i][-1,-2,1,-4] * embedders[1][i][1, -3]
	end
	mpotensors[1] = tmp
	for n in 2:L-1
		tmp = TensorMap(zeros, T, space(mpotensors[n-1], 3)' * h[n].pspace ← space(embedders[n][1], 2)' * h[n].pspace )
		for (i, j) in opkeys(h[n])
			@tensor tmp[-1, -2, -3, -4] += conj(embedders[n-1][i][1, -1]) * h[n, i, j][1,-2,2,-4] * embedders[n][j][2, -3]
		end
		for (i, j) in scalkeys(h[n])
			iden = h[n].Os[i, j] * isomorphism(Matrix{T}, h[n].pspace, h[n].pspace)
			@tensor tmp[-1, -2, -3, -4] += conj(embedders[n-1][i][1, -1]) * embedders[n][j][1, -3] * iden[-2, -4] 
		end
		mpotensors[n] = tmp
	end
	tmp = TensorMap(zeros, T, space(embedders[L][1], 2)' * h[L].pspace, oneunit(S) * h[L].pspace )
	_a = size(h[L], 2)
	for i in 1:length(embedders[L])
		@tensor tmp[-1, -2, -3, -4] += conj(embedders[L-1][i][1, -1]) * h[L, i, _a][1,-2,-3,-4]
	end
	mpotensors[L] = tmp
	return FiniteMPO(mpotensors)
end





