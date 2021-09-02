


function LinearAlgebra.dot(psiA::M, psiB::M) where {M <: FiniteMPS}
	(length(psiA) == length(psiB)) || throw(ArgumentError("dimension mismatch."))
    hold = l_LL(psiA)
    for i in 1:length(psiA)
        hold = updateleft(hold, psiA[i], psiB[i])
    end
    return tr(hold)
end


function LinearAlgebra.norm(psi::FiniteMPS; iscanonical::Bool=false) 
	iscanonical ? norm(psi[1]) : sqrt(real(dot(psi, psi)))
end
LinearAlgebra.norm(psi::AdjointFiniteMPS) = norm(psi.parent)

distance2(a::M, b::M) where {M <: FiniteMPS} = _distance2(a, b)
distance(a::M, b::M) where {M <: FiniteMPS} = _distance(a, b)


function LinearAlgebra.normalize!(psi::FiniteMPS; iscanonical::Bool=false)
    n = norm(psi, iscanonical=iscanonical)
    (n ≈ 0.) && @warn "quantum state has zero norm."
    if n != one(n)
        if iscanonical
            psi[1] = psi[1] / n
        else
            factor = n^(1 / length(psi))
            for i in 1:length(psi)
                psi[i] = psi[i] / factor
            end
        end  
    end
    return psi
end
LinearAlgebra.normalize(psi::FiniteMPS; iscanonical::Bool=false) = normalize!(copy(psi); iscanonical=iscanonical)

function LinearAlgebra.lmul!(f::Number, psi::FiniteMPS)
    if !isempty(psi)
        psi[1] *= f
    end
    return psi
end
LinearAlgebra.rmul!(psi::FiniteMPS, f::Number) = lmul!(f, psi)

Base.:*(psi::FiniteMPS, f::Number) = lmul!(f, copy(psi))
Base.:*(f::Number, psi::FiniteMPS) = psi * f
Base.:/(psi::FiniteMPS, f::Number) = psi * (1/f)


# # the function catdomain has to be generalized
# _cat_mps_site_tensor(a, b) = permute(catcodomain(permute(a, (2,), (1,3)), permute(b, (2,), (1,3))), (2,), (1,3))
# Base.:+(psiA::M, psiB::M) where {M <: FiniteMPS} = FiniteMPS([_cat_mps_site_tensor(a, b) for (a, b) in zip(raw_data(psiA), raw_data(psiB))])


# since generic catdomain is not implemented, this is an easy way around
# function _left_embedders(a::MPSTensor, b::MPSTensor)
#     # println(space(a))
#     # println(space(b))
#     V1, = codomain(a)
#     V2, = codomain(b)
#     V = V1 ⊕ V2
#     T = promote_type(eltype(a), eltype(b))
#     t1 = TensorMap(zeros, T, V, V1)
#     t2 = TensorMap(zeros, T, V, V2)
#     for c in sectors(V)
#         block(t1, c)[1:dim(V1, c), :] .= LinearAlgebra.Diagonal( ones(size(block(a, c), 1)) )
#         block(t2, c)[dim(V1, c) .+ (1:dim(V2, c)), :] .= LinearAlgebra.Diagonal( ones(size(block(b, c), 1)) )
#     end
#     return t1, t2
# end

function _right_embedders(a, b)
    V1, = domain(a)
    V2, = domain(b)
    V = V1 ⊕ V2
    T = promote_type(eltype(a), eltype(b))
    t1 = TensorMap(zeros, T, V1, V)
    t2 = TensorMap(zeros, T, V2, V)
    for c in sectors(V)
        block(t1, c)[:, 1:dim(V1, c)] .= LinearAlgebra.Diagonal( ones(size(block(a, c), 2)) )
        block(t2, c)[:, dim(V1, c) .+ (1:dim(V2, c))] .= LinearAlgebra.Diagonal( ones(size(block(b, c), 2)) )
    end
    return t1, t2
end


"""
    Base.:+(psiA::M, psiB::M) where {M <: FiniteMPS}
    addition of two MPSs
"""
function Base.:+(psiA::M, psiB::M) where {M <: FiniteMPS}
    (length(psiA) == length(psiB)) || throw(DimensionMismatch())
    (isempty(psiA)) && error("input mps is empty.")
    ((space_l(psiA) == space_l(psiB)) && (space_r(psiA)==space_r(psiB))) || throw(SpaceMismatch())
    if length(psiA) == 1
        return FiniteMPS([psiA[1] + psiB[1]])
    end
    embedders = [_right_embedders(aj, bj) for (aj, bj) in zip(raw_data(psiA), raw_data(psiB))]
    r = []
    for i in 1:length(psiA)
        if i == 1
            @tensor m1[-1 -2; -3] := psiA[i][-1,-2,2] * embedders[i][1][2, -3]
            @tensor m2[-1 -2; -3] := psiB[i][-1,-2,2] * embedders[i][2][2, -3]
        elseif i == length(psiA)
            @tensor m1[-1 -2; -3] := (embedders[i-1][1])'[-1, 1] * psiA[i][1,-2,-3] 
            @tensor m2[-1 -2; -3] := (embedders[i-1][2])'[-1, 1] * psiB[i][1,-2,-3] 
        else          
            @tensor m1[-1 -2; -3] := (embedders[i-1][1])'[-1, 1] * psiA[i][1,-2,2] * embedders[i][1][2, -3]
            @tensor m2[-1 -2; -3] := (embedders[i-1][2])'[-1, 1] * psiB[i][1,-2,2] * embedders[i][2][2, -3]
        end
        push!(r, m1 + m2)
    end
    return FiniteMPS([r...])
end
Base.:-(psiA::M, psiB::M) where {M <: FiniteMPS} = psiA + (-1) * psiB
Base.:-(psi::FiniteMPS) = -1 * psi

const MPS_APPROX_EQUAL_ATOL = 1.0e-14
Base.isapprox(psiA::M, psiB::M; atol=MPS_APPROX_EQUAL_ATOL) where {M <: FiniteMPS} = distance2(psiA, psiB) <= atol


