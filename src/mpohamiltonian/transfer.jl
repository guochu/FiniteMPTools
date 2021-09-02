

function r_RR(psiA::M, h::MPOHamiltonian, psiB::M) where {M <: Union{FiniteMPS, ExactFiniteMPS}}
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	L = length(psiA)
	(mod(L, period(h))==0) || throw(DimensionMismatch())
	T = promote_type(scalar_type(psiA), scalar_type(h), scalar_type(psiB))
	rrr = r_RR(psiA)

	i = size(h[L], 2)
	util_right = Tensor(ones,T,h[L].omspaces[i])
	@tensor ctr[-1; -2 -3] := rrr[-1,-3]*util_right[-2]

	right_starter = Vector{typeof(ctr)}(undef, size(h[L], 2))
	right_starter[end] = ctr
	for i in 1:size(h[L], 2)-1
		util_right = Tensor(zeros,T,h[L].omspaces[i])

		@tensor ctr[-1; -2 -3]:= rrr[-1,-3]*util_right[-2]
		right_starter[i] = ctr
	end
	return right_starter
end 


function l_LL(psiA::M, h::MPOHamiltonian, psiB::M) where {M <: Union{FiniteMPS, ExactFiniteMPS}}
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	L = length(psiA)
	(mod(L, period(h))==0) || throw(DimensionMismatch())
	T = promote_type(scalar_type(psiA), scalar_type(h), scalar_type(psiB))
	lll = l_LL(psiA)

	i = 1
	util_left = Tensor(ones,T,h[1].imspaces[i]')
	@tensor ctl[-1; -2 -3]:= lll[-1,-3]*util_left[-2]
	left_starter = [ctl]

	for i in 2:size(h[1], 1)
		util_left = Tensor(zeros,T,h[1].imspaces[i]')
		@tensor ctl[-1; -2 -3]:= lll[-1,-3]*util_left[-2]
		push!(left_starter, ctl)
	end
	return left_starter
end 



function updateright(hold::Vector, psiAj::MPSTensor{S}, hj::AbstractSiteMPOTensor{S}, psiBj::MPSTensor) where {S <: EuclideanSpace}
	@assert length(hold) == size(hj, 2)
	T = promote_type(eltype(psiAj), scalar_type(hj), eltype(psiBj))
	hnew = [TensorMap(zeros, T, space(psiAj, 1)' , hj.imspaces[i]' ⊗ space(psiBj, 1)' ) for i in 1:size(hj, 1) ]
	for (i, j) in keys(hj)
		if isscal(hj, i, j)
			hnew[i] += hj.Os[i, j] * updateright(hold[j], psiAj, nothing, psiBj)
		else
			hnew[i] += updateright(hold[j], psiAj, hj.Os[i, j], psiBj)
		end
	end
	return hnew
end

function updateleft(hold::Vector, psiAj::MPSTensor{S}, hj::AbstractSiteMPOTensor{S}, psiBj::MPSTensor) where {S <: EuclideanSpace}
	@assert length(hold) == size(hj, 1)
	T = promote_type(eltype(psiAj), scalar_type(hj), eltype(psiBj))
	hnew = [TensorMap(zeros, T, space(psiAj, 3)' , hj.omspaces[j] ⊗ space(psiBj, 3)' ) for j in 1:size(hj, 2)]

	for (i, j) in keys(hj)
		if isscal(hj, i, j)
			hnew[j] += hj.Os[i, j] * updateleft(hold[i], psiAj, nothing, psiBj)
		else
			hnew[j] += updateleft(hold[i], psiAj, hj.Os[i, j], psiBj)
		end
	end
	return hnew
end


