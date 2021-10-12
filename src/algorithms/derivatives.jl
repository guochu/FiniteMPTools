
function ac_prime(x::MPSTensor{S}, m::MPOTensor{S}, hleft::AbstractTensorMap{S, 1, 2}, hright::AbstractTensorMap{S, 1, 2}) where {S <: EuclideanSpace}
	@tensor tmp[-1 -2; -3] := ((hleft[-1, 1, 2] * x[2,3,4]) * m[1,-2,5,3]) * hright[-3,5,4]
end


function ac_prime(x::MPSTensor{S}, ham::AbstractSiteMPOTensor{S}, hleft::Vector, hright::Vector) where {S <: EuclideanSpace}
	tmp = zero(x)
    for (i,j) in opkeys(ham)
        @tensor tmp[-1,-2,-3]+=hleft[i][-1,5,4]*x[4,2,1]*ham[i,j][5,-2,3,2]*hright[j][-3,3,1]
    end
    for (i,j) in scalkeys(ham)
        scal = ham.Os[i,j]
        @tensor tmp[-1,-2,-3]+=hleft[i][-1,5,4]*(scal*x)[4,-2,1]*hright[j][-3,5,1]
    end
    return tmp
end


function ac2_prime(x::MPOTensor{S}, h1::MPOTensor{S}, h2::MPOTensor{S}, hleft::AbstractTensorMap{S, 1, 2}, hright::AbstractTensorMap{S, 1, 2}) where {S <: EuclideanSpace}
	@tensor tmp[-1 -2; -3 -4] := (((hleft[-1, 1, 2] * x[2, 3, 4, 5]) * h1[1, -2, 6, 3]) * h2[6, -3, 7, 4]) * hright[-4, 7, 5]
end

function ac2_prime(x::MPOTensor{S},h1::AbstractSiteMPOTensor{S},h2::AbstractSiteMPOTensor{S},hleft::Vector, hright::Vector) where {S <: EuclideanSpace}
	@assert size(h1, 2) == size(h2, 1)
    tmp=zero(x)

    for (i,j) in keys(h1)
        for k in 1:size(h2, 2)
            contains(h2,j,k) || continue

            if isscal(h1,i,j) && isscal(h2,j,k)
                scal = h1.Os[i,j]*h2.Os[j,k]
                @tensor tmp[-1,-2,-3,-4] += (scal*hleft[i])[-1,7,6]*x[6,-2,-3,1]*hright[k][-4,7,1]
            elseif isscal(h1,i,j)
                scal = h1.Os[i,j]
                @tensor tmp[-1,-2,-3,-4]+=(scal*hleft[i])[-1,7,6]*x[6,-2,3,1]*h2[j,k][7,-3,2,3]*hright[k][-4,2,1]
            elseif isscal(h2,j,k)
                scal = h2.Os[j,k]
                @tensor tmp[-1,-2,-3,-4]+=(scal*hleft[i])[-1,7,6]*x[6,5,-3,1]*h1[i,j][7,-2,2,5]*hright[k][-4,2,1]
            else
                @tensor tmp[-1,-2,-3,-4]+=hleft[i][-1,7,6]*x[6,5,3,1]*h1[i,j][7,-2,4,5]*h2[j,k][4,-3,2,3]*hright[k][-4,2,1]
            end
        end

    end

    return tmp
end


function c_prime(x::MPSBondTensor{S}, hleft::AbstractTensorMap{S, 1, 2}, hright::AbstractTensorMap{S, 1, 2}) where {S <: EuclideanSpace}
    @tensor tmp[-1; -2] := (hleft[-1, 1, 2] * x[2, 3]) * hright[-2, 1, 3]
end

function c_prime(x::MPSBondTensor{S}, hleft::Vector, hright::Vector) where {S <: EuclideanSpace}
    @assert length(hleft) == length(hright)
    tmp = zero(x)
    for (hl, hr) in zip(hleft, hright)
        @tensor tmp[-1, -2] += (hl[-1, 1, 2] * x[2, 3]) * hr[-2,1,3]
    end
    return tmp
end

function c_proj(x::MPSTensor, cleft::AbstractTensorMap{S, 1, 1}, cright::AbstractTensorMap{S, 1, 1}) where {S <: EuclideanSpace}
	@tensor tmp[-1 -2; -3] := cleft[-1, 1] * x[1, -2, 2] * cright[-3, 2]
end


# function c2_proj(x::MPOTensor, cleft::AbstractTensorMap{S, 1, 1}, cright::AbstractTensorMap{S, 1, 1}) where {S <: EuclideanSpace}
# 	@tensor tmp[-1 -2; -3 -4] := cleft[-1, 1] * x[1, -2, -3, 2] * cright[-4, 2]
# end
