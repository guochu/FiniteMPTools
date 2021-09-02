

function LinearAlgebra.dot(hA::M, hB::M) where {M <: FiniteMPO}
	(length(hA) == length(hB)) || throw(ArgumentError("dimension mismatch."))
    hold = l_LL(hA)
    for i in 1:length(hA)
        hold = updateleft(hold, hA[i], hB[i])
    end
    return tr(hold)
end
LinearAlgebra.norm(h::FiniteMPO) = sqrt(real(dot(h, h)))
LinearAlgebra.norm(h::AdjointFiniteMPO) = norm(h.parent)
distance2(hA::M, hB::M) where {M <: FiniteMPO} = _distance2(hA, hB)
distance(hA::M, hB::M) where {M <: FiniteMPO} = _distance(hA, hB)

function LinearAlgebra.lmul!(f::Number, h::FiniteMPO)
    if !isempty(h)
        h[1] *= f
    end
    return h
end
LinearAlgebra.rmul!(h::FiniteMPO, f::Number) = lmul!(f, h)

Base.:*(h::FiniteMPO, f::Number) = lmul!(f, copy(h))
Base.:*(f::Number, h::FiniteMPO) = h * f
Base.:/(h::FiniteMPO, f::Number) = h * (1/f)



# since generic catdomain is not implemented, this is an easy way around
function _left_embedders(a::MPOTensor, b::MPOTensor)
    # println(space(a))
    # println(space(b))
    a = permute(a, (1,), (2,3,4))
    b = permute(b, (1,), (2,3,4))
    # println(space(a))
    # println(space(b))
    V1 = space(a, 1)
    V2 = space(b, 1)
    V = V1 ⊕ V2
    T = promote_type(eltype(a), eltype(b))
    t1 = TensorMap(zeros, T, V, V1)
    t2 = TensorMap(zeros, T, V, V2)
    for c in sectors(V)
        block(t1, c)[1:dim(V1, c), :] .= LinearAlgebra.Diagonal( ones(size(block(a, c), 1)) )
        block(t2, c)[dim(V1, c) .+ (1:dim(V2, c)), :] .= LinearAlgebra.Diagonal( ones(size(block(b, c), 1)) )
    end
    return t1, t2
end


"""
    Base.:+(hA::M, hB::M) where {M <: FiniteMPO}
    addition of two MPOs
"""
function Base.:+(hA::M, hB::M) where {M <: FiniteMPO}
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    (isempty(hA)) && error("input mpo is empty.")
    ((space_l(hA) == space_l(hB)) && (space_r(hA)==space_r(hB))) || throw(SpaceMismatch())
    embedders = [_left_embedders(aj, bj) for (aj, bj) in zip(raw_data(hA), raw_data(hB))]
    r = []
    for i in 1:length(hA)
        if i == 1
            @tensor m1[-1 -2; -3 -4] := hA[i][-1,-2,1,-4] * (embedders[i+1][1])'[1, -3]
            @tensor m2[-1 -2; -3 -4] := hB[i][-1,-2,1,-4] * (embedders[i+1][2])'[1, -3]
        elseif i == length(hA)
            @tensor m1[-1 -2; -3 -4] := embedders[i][1][-1, 1] * hA[i][1,-2,-3,-4] 
            @tensor m2[-1 -2; -3 -4] := embedders[i][2][-1, 1] * hB[i][1,-2,-3,-4] 
        else          
            @tensor m1[-1 -2; -3 -4] := embedders[i][1][-1, 1] * hA[i][1,-2,2,-4] * (embedders[i+1][1])'[2, -3]
            @tensor m2[-1 -2; -3 -4] := embedders[i][2][-1, 1] * hB[i][1,-2,2,-4] * (embedders[i+1][2])'[2, -3]
        end
        push!(r, m1 + m2)
    end
    return FiniteMPO([r...])
end
Base.:-(hA::M, hB::M) where {M <: FiniteMPO} = hA + (-1) * hB
Base.:-(h::FiniteMPO) = -1 * h

function Base.:*(h::FiniteMPO, psi::FiniteMPS)
    (length(h) == length(psi)) || throw(DimensionMismatch())
    isempty(h) && throw(ArgumentError("input operator is empty."))
    r = [@tensor tmp[-1 -2; -3 -4 -5] := a[-1, -3, -4, 1] * b[-2, 1, -5] for (a, b) in zip(raw_data(h), raw_data(psi))]
    vacuum = oneunit(space_l(psi))
    left = isomorphism(fuse(vacuum, vacuum), vacuum ⊗ vacuum)
    fusion_ts = [isomorphism(space(item, 4)' ⊗ space(item, 5)', fuse(space(item, 4)', space(item, 5)')) for item in r]
    mpstensors = similar(raw_data(psi))
    @tensor tmp[-1 -2; -3] := left[-1, 1, 2] * r[1][1,2,-2,3,4] * fusion_ts[1][3,4,-3]
    mpstensors[1] = tmp
    for i in 2:length(h)
        @tensor tmp[-1 -2; -3] := conj(fusion_ts[i-1][1,2,-1]) * r[i][1,2,-2,3,4] * fusion_ts[i][3,4,-3]
        mpstensors[i] = tmp
    end
    return FiniteMPS(mpstensors)
end
Base.:*(h::AdjointFiniteMPO, psi::AdjointFiniteMPS) = (h.parent * psi.parent)'


"""
    Base.:*(hA::M, hB::M) where {M <: FiniteMPO}
    a * b
"""
function Base.:*(hA::M, hB::M) where {M <: FiniteMPO}
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    isempty(hA) && throw(ArgumentError("input operator is empty."))
    r = [@tensor tmp[-1 -2 -3; -4 -5 -6] := a[-1, -3, -4, 1] * b[-2, 1, -5, -6] for (a, b) in zip(raw_data(hA), raw_data(hB))]
    vacuum = oneunit(space_l(hA))
    left = isomorphism(fuse(vacuum, vacuum), vacuum ⊗ vacuum)
    fusion_ts = [isomorphism(space(item, 4)' ⊗ space(item, 5)', fuse(space(item, 4)', space(item, 5)')) for item in r]
    mpotensors = similar(raw_data(hA))
    @tensor tmp[-1 -2; -3 -4] := left[-1, 1, 2] * r[1][1,2,-2,3,4,-4] * fusion_ts[1][3,4,-3]
    mpotensors[1] = tmp
    for i in 2:length(hA)
        @tensor tmp[-1 -2; -3 -4] := conj(fusion_ts[i-1][1,2,-1]) * r[i][1,2,-2,3,4,-4] * fusion_ts[i][3,4,-3]
        mpotensors[i] = tmp
    end
    return FiniteMPO(mpotensors)
end

"""
    Base.:*(hA::M, hB::M) where {M <: AdjointFiniteMPO} 
    a† * b†
"""
Base.:*(hA::M, hB::M) where {M <: AdjointFiniteMPO} = (hB.parent * hA.parent)'

function mpo_tensor_adjoint(vj::MPOTensor)
    rj = vj'
    sl = space(rj, 3)'
    ml = isomorphism(sl, flip(sl))
    sr = space(rj, 1)
    mr = isomorphism(flip(sr), sr)
    @tensor tmp[-1 -2; -3 -4] := ml[1, -1] * rj[2,-2,1,-4] * mr[-3,2]
    return tmp
end

function FiniteMPO(h::AdjointFiniteMPO)
    isstrict(h) || throw(ArgumentError("not strict operator allowed."))
    return FiniteMPO(mpo_tensor_adjoint.(raw_data(h.parent)))
end  

function dot_prod(hA::AdjointFiniteMPO, hB::FiniteMPO; right::EuclideanSpace)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    isempty(hA) && throw(ArgumentError("input operator is empty."))    
    hAp = adjoint.(raw_data(hA.parent))
    L = length(hAp)
    vacuum = oneunit(right)
    fl = isomorphism(vacuum' ⊗ vacuum, fuse(vacuum, vacuum))
    mpotensors = similar(raw_data(hB))
    for i in 1:L-1
        _a = space(hAp[i], 1)'
        _b = space(hB[i], 3)'
        fp = isomorphism(_a ⊗ _b, fuse(_a, _b) )
        @tensor tmp[-1 -2; -3 -4] := conj(fl[1,2,-1]) * hAp[i][3, -2, 1, 5] * hB[i][2,5,4,-4] * fp[3,4,-3]
        mpotensors[i] = tmp
        fl = fp
    end
    fp = isomorphism(right ⊗ space_r(hB)', space(hAp[L], 1))
    @tensor tmp[-1 -2; -3 -4] := conj(fl[1,2,-1]) * hAp[L][3, -2, 1, 5] * hB[L][2,5,4,-4] * fp[-3,4,3]
    mpotensors[L] = tmp
    return FiniteMPO(mpotensors)
end

"""
    Base.:*(hA::AdjointFiniteMPO, hB::FiniteMPO)
    a† * b
"""
Base.:*(hA::AdjointFiniteMPO, hB::FiniteMPO) = dot_prod(hA, hB; right=oneunit(spacetype(hB))')

function dot_prod(hA::FiniteMPO, hB::AdjointFiniteMPO; right::EuclideanSpace)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    isempty(hA) && throw(ArgumentError("input operator is empty."))    
    hBp = adjoint.(raw_data(hB.parent))    
    L = length(hA)
    vacuum = oneunit(right)
    fl = isomorphism(vacuum ⊗ vacuum', fuse(vacuum, vacuum))
    mpotensors = similar(raw_data(hA))
    for i in 1:L-1
        # println("i======================$i")
        _a = space(hA[i], 3)'
        _b = space(hBp[i], 1)'
        fp = isomorphism(_a ⊗ _b, fuse(_a, _b) )
        @tensor tmp[-1 -2; -3 -4] := conj(fl[1,2,-1]) * hA[i][1, -2, 3, 5] * hBp[i][4,5,2,-4] * fp[3,4,-3]
        mpotensors[i] = tmp
        fl = fp
    end
    # println("here**********************")
    fp = isomorphism(right ⊗ space(hBp[L], 1)', space_r(hA))
    @tensor tmp[-1 -2; -3 -4] := conj(fl[1,2,-1]) * hA[L][1, -2, 3, 5] * hBp[L][4,5,2,-4] * fp[-3,4,3]
    mpotensors[L] = tmp 
    return FiniteMPO(mpotensors)
end

"""
    Base.:*(hA::FiniteMPO, hB::AdjointFiniteMPO)
    a * b†
"""
Base.:*(hA::FiniteMPO, hB::AdjointFiniteMPO) = dot_prod(hA, hB; right=oneunit(spacetype(hA))')

const MPO_APPROX_EQUAL_ATOL = 1.0e-12

"""
    Base.isapprox(a::M, b::M) where {M <: FiniteMPO} 
    Check is two MPOs are approximated equal 
"""
Base.isapprox(a::M, b::M; atol=MPO_APPROX_EQUAL_ATOL) where {M <: FiniteMPO} = distance2(a, b) <= atol


r_RR(psiA::M, h::FiniteMPO, psiB::M) where {M <: Union{FiniteMPS, ExactFiniteMPS}} = isomorphism(
    Matrix{promote_type(scalar_type(psiA), scalar_type(h), scalar_type(psiB))}, space_r(psiA), space_r(h) ⊗ space_r(psiB))


l_LL(psiA::M, h::FiniteMPO, psiB::M) where {M <: Union{FiniteMPS, ExactFiniteMPS}} = isomorphism(
    Matrix{promote_type(scalar_type(psiA), scalar_type(h), scalar_type(psiB))}, space_l(psiA), space_l(h) ⊗ space_l(psiB))


"""
    expectation(psiA::FiniteMPS, h::FiniteMPO, psiB::FiniteMPS)
    compute < psiA | h | psiB >
"""
function expectation(psiA::FiniteMPS{<:MPSTensor{S}}, h::FiniteMPO{<:MPOTensor{S}}, psiB::FiniteMPS{<:MPSTensor{S}}) where {S <: EuclideanSpace}
    (length(psiA) == length(h) == length(psiB)) || throw(DimensionMismatch())
    hold = r_RR(psiA, h, psiB)
    for i in length(psiA):-1:1
        hold = updateright(hold, psiA[i], h[i], psiB[i])
    end
    return scalar(hold)
end
expectation(h::FiniteMPO, psi::FiniteMPS) = expectation(psi, h, psi)


function LinearAlgebra.ishermitian(h::FiniteMPO)
    isempty(h) && throw(ArgumentError("input operator is empty."))
    isstrict(h) || throw(ArgumentError("input operator must be strict."))
    return isapprox(h, FiniteMPO(h'), atol=1.0e-10) 
end




