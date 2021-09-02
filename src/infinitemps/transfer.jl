

function updateleft(hold::MPSBondTensor, A::MPSTensor, B::MPSTensor)
	@tensor out[-1; -2] := conj(A[1,2,-1]) * hold[1,3] * B[3,2,-2]
end

function updateright(hold::MPSBondTensor, A::MPSTensor, B::MPSTensor)
	@tensor out[-1; -2] := conj(A[-1,1,2]) * hold[2,3] * B[-2,1,3]
end


function updateleft(v, As::Vector, Bs::Vector)
	@assert length(As) == length(Bs)
	for (A, B) in zip(As, Bs)
		v = updateleft(v, A, B)
	end
	return v
end


function updateright(v, As::Vector, Bs::Vector)
	@assert length(As) == length(Bs)
	for (A, B) in Iterators.reverse(zip(As, Bs))
		v = updateright(v, A, B)
	end
	return v	
end

function updateleft(v::MPSBondTensor,A::AbstractArray,Ab::AbstractArray, rvec, lvec)
	v = updateleft(v, A, Ab)
	@tensor v[-1;-2]-=rvec[1,2]*v[2,1]*lvec[-1,-2]
	return v
end

function updateleft(v::AbstractTensorMap{S, 1, 2},A::AbstractArray,Ab::AbstractArray, rvec, lvec) where {S}
	v = updateleft(v, A, Ab)
	@tensor v[-1, -2-3]-=rvec[1,2]*v[2,-2,1]*lvec[-1,-3]
	return v
end

function updateleft(v, A::AbstractArray, h::AbstractArray, Ab::AbstractArray)
	for (hj, aj, abj) in zip(h, A, Ab)
		v = updateleft(v, aj, hj, abj)
	end
	return v
end

function updateright(v, A::AbstractArray, h::AbstractArray, Ab::AbstractArray)
	for (hj, aj, abj) in zip(h, A, Ab)
		v = updateright(v, aj, hj, abj)
	end
	return v
end

function updateright(v::MPSBondTensor,A::AbstractArray,Ab::AbstractArray, rvec,lvec)
	v = updateright(v)
	@tensor v[-1; -2]-=lvec[1,2]*v[2,1]*rvec[-1,-2]
	return v
end


function updateright(v::AbstractTensorMap{S, 1, 2},A::AbstractArray,Ab::AbstractArray, rvec,lvec) where S
	v = updateright(v)
	@tensor v[-1; -2 -3]-=lvec[1,2]*v[2,-2,1]*rvec[-1,-3]
    return v
end
