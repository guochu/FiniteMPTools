

function LinearAlgebra.dot(hA::FiniteMPO, hB::FiniteMPO) 
	(length(hA) == length(hB)) || throw(ArgumentError("dimension mismatch."))
    hold = l_LL(hA)
    for i in 1:length(hA)
        hold = updateleft(hold, hA[i], hB[i])
    end
    return tr(hold)
end
LinearAlgebra.norm(h::FiniteMPO) = sqrt(real(dot(h, h)))
LinearAlgebra.norm(h::AdjointFiniteMPO) = norm(h.parent)
distance2(hA::FiniteMPO, hB::FiniteMPO) = _distance2(hA, hB)
distance(hA::FiniteMPO, hB::FiniteMPO) = _distance(hA, hB)


# get identity operator

"""
    TensorKit.id(m::FiniteMPO)
Retuen an identity MPO from a given MPO
"""
TensorKit.id(m::FiniteMPO) = FiniteMPO([id(Matrix{scalar_type(m)}, oneunit(spacetype(m)) ⊗ space(item, 2) ) for item in raw_data(m)])


l_tr(h::Union{FiniteMPO, FiniteDensityOperatorMPS}) = Tensor(ones,scalar_type(h),oneunit(spacetype(h))')

function LinearAlgebra.tr(h::FiniteMPO)
    isempty(h) && return 0.
    L = length(h)
    hold = l_tr(h)
    for i in 1:L
        hold = updatetraceleft(hold, h[i])
    end
    return scalar(hold)
end


function LinearAlgebra.lmul!(f::Number, h::FiniteMPO)
    if !isempty(h)
        h[1] *= f
    end
    return h
end
LinearAlgebra.rmul!(h::FiniteMPO, f::Number) = lmul!(f, h)

Base.:*(h::FiniteMPO, f::Number) = lmul!(f, copy(h))
Base.:*(h::AdjointFiniteMPO, f::Number) = adjoint(h.parent * conj(f))
Base.:*(f::Number, h::AbstractMPO) = h * f
Base.:/(h::AbstractMPO, f::Number) = h * (1/f)



# this function is very stupid 
function _left_embedders(a::MPOTensor, b::MPOTensor)
    a = permute(a, (1,), (2,3,4))
    b = permute(b, (1,), (2,3,4))
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
function Base.:+(hA::FiniteMPO, hB::FiniteMPO)
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
# adding mpo with adjoint mpo will return an normal mpo
Base.:+(hA::FiniteMPO, hB::AdjointFiniteMPO) = hA + FiniteMPO(hB)
Base.:+(hA::AdjointFiniteMPO, hB::FiniteMPO) = FiniteMPO(hA) + hB
Base.:+(hA::AdjointFiniteMPO, hB::AdjointFiniteMPO) = adjoint(hA.parent + hB.parent)
Base.:-(hA::AbstractMPO, hB::AbstractMPO) = hA + (-1) * hB
Base.:-(h::AbstractMPO) = -1 * h

"""
    Base.:*(h::FiniteMPO, psi::FiniteMPS)
    Base.:*(h::FiniteMPO, psi::FiniteDensityOperatorMPS)
    Base.:*(h::AdjointFiniteMPO, psi::AdjointFiniteMPS)
    Multiplication of mps by an mpo.
"""
function Base.:*(h::FiniteMPO, psi::FiniteMPS)
    (length(h) == length(psi)) || throw(DimensionMismatch())
    isempty(h) && throw(ArgumentError("input operator is empty."))
    r = [@tensor tmp[-1 -2; -3 -4 -5] := a[-1, -3, -4, 1] * b[-2, 1, -5] for (a, b) in zip(raw_data(h), raw_data(psi))]
    left = isomorphism(fuse(space_l(h), space_l(psi)), space_l(h) ⊗ space_l(psi))
    fusion_ts = [isomorphism(space(item, 4)' ⊗ space(item, 5)', fuse(space(item, 4)', space(item, 5)')) for item in r]
    @tensor tmp[-1 -2; -3] := left[-1, 1, 2] * r[1][1,2,-2,3,4] * fusion_ts[1][3,4,-3]
    mpstensors = Vector{typeof(tmp)}(undef, length(h))
    mpstensors[1] = tmp
    for i in 2:length(h)
        @tensor tmp[-1 -2; -3] := conj(fusion_ts[i-1][1,2,-1]) * r[i][1,2,-2,3,4] * fusion_ts[i][3,4,-3]
        mpstensors[i] = tmp
    end
    return FiniteMPS(mpstensors)
end
Base.:*(h::FiniteMPO, psi::FiniteDensityOperatorMPS) = FiniteDensityOperatorMPS(h * psi.data, psi.fusers, psi.I)
Base.:*(h::AdjointFiniteMPO, psi::AdjointFiniteMPS) = (h.parent * psi.parent)'
# Base.:*(h::AdjointFiniteMPO, psi::FiniteMPS) = FiniteMPO(h) * psi

# normal normal mult of two chains of MPOTensor
function _mult_n_n(a::Vector{<:MPOTensor}, b::Vector{<:MPOTensor})
    r = [@tensor tmp[-1 -2 -3; -4 -5 -6] := aj[-1, -3, -4, 1] * bj[-2, 1, -5, -6] for (aj, bj) in zip(a, b)]
    T = eltype(r[1])
    left = isomorphism(Matrix{T}, fuse(space(a[1], 1), space(b[1], 1)), space(a[1], 1) ⊗ space(b[1], 1))
    fusion_ts = [isomorphism(Matrix{T}, space(item, 4)' ⊗ space(item, 5)', fuse(space(item, 4)', space(item, 5)')) for item in r]
    @tensor tmp[-1 -2; -3 -4] := left[-1, 1, 2] * r[1][1,2,-2,3,4,-4] * fusion_ts[1][3,4,-3]
    mpotensors = Vector{typeof(tmp)}(undef, length(a))
    mpotensors[1] = tmp
    for i in 2:length(a)
        @tensor tmp[-1 -2; -3 -4] := conj(fusion_ts[i-1][1,2,-1]) * r[i][1,2,-2,3,4,-4] * fusion_ts[i][3,4,-3]
        mpotensors[i] = tmp
    end
    return mpotensors
end

"""
    Base.:*(hA::M, hB::M) where {M <: FiniteMPO}
    a * b
"""
function Base.:*(hA::FiniteMPO, hB::FiniteMPO) 
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    isempty(hA) && throw(ArgumentError("input operator is empty."))
    return FiniteMPO(_mult_n_n(raw_data(hA), raw_data(hB)))
end

"""
    Base.:*(hA::M, hB::M) where {M <: AdjointFiniteMPO} 
    a† * b†
"""
Base.:*(hA::AdjointFiniteMPO, hB::AdjointFiniteMPO) = (hB.parent * hA.parent)'

function mpo_tensor_adjoint(vj::MPOTensor)
    rj = vj'
    sl = space(rj, 3)'
    ml = isomorphism(Matrix{eltype(vj)}, sl, flip(sl))
    sr = space(rj, 1)
    mr = isomorphism(Matrix{eltype(vj)}, flip(sr), sr)
    @tensor tmp[-1 -2; -3 -4] := ml[1, -1] * rj[2,-2,1,-4] * mr[-3,2]
    return tmp
end


function FiniteMPO(h::AdjointFiniteMPO)
    isstrict(h) || throw(ArgumentError("not strict operator allowed."))
    return FiniteMPO(mpo_tensor_adjoint.(raw_data(h.parent)))
end  
# FiniteMPO(h::AdjointFiniteMPO) = FiniteMPO(mpo_tensor_adjoint.(raw_data(h.parent)))

# mult adjoint and normal
function _mult_a_n(a::Vector{<:MPOTensor}, hB::Vector{<:MPOTensor}, right::EuclideanSpace)
    hAp = adjoint.(a)
    L = length(hAp)
    vacuum = oneunit(right)
    fl = isomorphism(vacuum' ⊗ vacuum, fuse(vacuum, vacuum))
    mpotensors = Vector{Any}(undef, L)
    for i in 1:L-1
        _a = space(hAp[i], 1)'
        _b = space(hB[i], 3)'
        fp = isomorphism(_a ⊗ _b, fuse(_a, _b) )
        @tensor tmp[-1 -2; -3 -4] := conj(fl[1,2,-1]) * hAp[i][3, -2, 1, 5] * hB[i][2,5,4,-4] * fp[3,4,-3]
        mpotensors[i] = tmp
        fl = fp
    end
    fp = isomorphism(right ⊗ space(hB[L], 3)', space(hAp[L], 1))
    @tensor tmp[-1 -2; -3 -4] := conj(fl[1,2,-1]) * hAp[L][3, -2, 1, 5] * hB[L][2,5,4,-4] * fp[-3,4,3]
    mpotensors[L] = tmp    
    return [mpotensors...]
end

function dot_prod(hA::AdjointFiniteMPO, hB::FiniteMPO; right::EuclideanSpace)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    isempty(hA) && throw(ArgumentError("input operator is empty."))    
    return FiniteMPO(_mult_a_n(raw_data(hA.parent),  raw_data(hB), right))
end

"""
    Base.:*(hA::AdjointFiniteMPO, hB::FiniteMPO)
    a† * b
"""
Base.:*(hA::AdjointFiniteMPO, hB::FiniteMPO) = dot_prod(hA, hB; right=oneunit(spacetype(hB))')

# mult normal and adjoint
function _mult_n_a(hA::Vector{<:MPOTensor}, hB::Vector{<:MPOTensor}, right::EuclideanSpace)
    hBp = adjoint.(hB)    
    L = length(hA)
    vacuum = oneunit(right)
    fl = isomorphism(vacuum ⊗ vacuum', fuse(vacuum, vacuum))
    mpotensors = Vector{Any}(undef, L)
    for i in 1:L-1
        _a = space(hA[i], 3)'
        _b = space(hBp[i], 1)'
        fp = isomorphism(_a ⊗ _b, fuse(_a, _b) )
        @tensor tmp[-1 -2; -3 -4] := conj(fl[1,2,-1]) * hA[i][1, -2, 3, 5] * hBp[i][4,5,2,-4] * fp[3,4,-3]
        mpotensors[i] = tmp
        fl = fp
    end
    fp = isomorphism(right ⊗ space(hBp[L], 1)', space(hA[L], 3))
    @tensor tmp[-1 -2; -3 -4] := conj(fl[1,2,-1]) * hA[L][1, -2, 3, 5] * hBp[L][4,5,2,-4] * fp[-3,4,3]
    mpotensors[L] = tmp 
   return [mpotensors...]
end

function dot_prod(hA::FiniteMPO, hB::AdjointFiniteMPO; right::EuclideanSpace)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    isempty(hA) && throw(ArgumentError("input operator is empty."))    
    return FiniteMPO(_mult_n_a(raw_data(hA), raw_data(hB.parent), right))
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
Base.isapprox(a::FiniteMPO, b::FiniteMPO; atol=MPO_APPROX_EQUAL_ATOL) = distance2(a, b) <= atol


r_RR(psiA::M, h::FiniteMPO, psiB::M) where {M <: Union{FiniteMPS, ExactFiniteMPS}} = loose_isometry(
    Matrix{promote_type(scalar_type(psiA), scalar_type(h), scalar_type(psiB))}, space_r(psiA), space_r(h) ⊗ space_r(psiB))


l_LL(psiA::M, h::FiniteMPO, psiB::M) where {M <: Union{FiniteMPS, ExactFiniteMPS}} = loose_isometry(
    Matrix{promote_type(scalar_type(psiA), scalar_type(h), scalar_type(psiB))}, space_l(psiA), space_l(h) ⊗ space_l(psiB))


"""
    expectation(psiA::FiniteMPS, h::FiniteMPO, psiB::FiniteMPS)
    expectation(h::FiniteMPO, psi::FiniteMPS) = expectation(psi, h, psi)
    expectation(h::FiniteMPO, psi::FiniteDensityOperatorMPS) = expectation(psi.I, h, psi.data)
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

# conversion between FiniteMPO and FiniteDensityOperatorMPS
site_dm_to_mpotensor(psij::MPSTensor, fuser::MPSTensor) = @tensor o[-1 -2; -3 -4] := psij[-1,1,-3]*fuser[-2,-4,1]


FiniteMPO(psi::FiniteDensityOperatorMPS) = FiniteMPO([@tensor o[-1 -2; -3 -4] := psi[i][-1,1,-3]*psi.fusers[i][-2,-4,1] for i in 1:length(psi)])
function DensityOperator(h::FiniteMPO)
    T = scalar_type(h)
    fusers = [isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4)) ) for m in raw_data(h)]
    mps = FiniteMPS([@tensor o[-1 -2; -3] := m[-1,1,-3,2] * conj(fj[1,2,-2])  for (fj, m) in zip(fusers, raw_data(h))])

    # this is the only we can do
    tmp = get_trivial_leg(mps[1])
    iden = FiniteMPS([@tensor o[-1 -2; -3] := tmp[-1] * conj(fj[1,1,-2]) * conj(tmp[-3]) for fj in fusers])
    return FiniteDensityOperatorMPS(mps, fusers, iden)
end

expectation(h::FiniteMPO, psi::FiniteDensityOperatorMPS) = expectation(psi.I, h, psi.data)

# kronecker product between MPOs
function _otimes_n_n(x::Vector{<:MPOTensor}, y::Vector{<:MPOTensor}, f)
    (length(x) == length(y)) || throw(DimensionMismatch())
    L = length(x)
    r = [f(xj, yj) for (xj, yj) in zip(x, y)]
    p_f = [isomorphism(Matrix{eltype(m)}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4))) for m in r]
    l_f = PeriodicArray([isomorphism(Matrix{eltype(m)}, fuse(space(m, 1), space(m, 3)), space(m, 1) ⊗ space(m, 3) ) for m in r])
    v=[@tensor o[-1 -2; -3 -4] := l_f[i][-1,1,3]*r[i][1,2,3,4,5,6,7,8]*conj(p_f[i][2,4,-2])*conj(l_f[i+1][-3,5,7])*p_f[i][6,8,-4] for i in 1:L]
    return v
end

function _otimes(x::FiniteMPO, y::FiniteMPO, f)
    (isstrict(x) && isstrict(y)) || throw(ArgumentError("only strict MPOs allowed."))
    v = _otimes_n_n(raw_data(x), raw_data(y), f)
    return FiniteMPO(v)
end

# function _otimes(x::FiniteMPO, y::FiniteMPO, f)
#     (length(x) == length(y)) || throw(DimensionMismatch())
#     (isstrict(x) && isstrict(y)) || throw(ArgumentError("only strict MPOs allowed."))
#     L = length(x)
#     T = promote_type(scalar_type(x), scalar_type(y))
#     r = [f(xj, yj) for (xj, yj) in zip(raw_data(x), raw_data(y))]

#     p_f = [isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4))) for m in r]
#     l_f = PeriodicArray([isomorphism(Matrix{T}, fuse(space(m, 1), space(m, 3)), space(m, 1) ⊗ space(m, 3) ) for m in r])
#     v=[@tensor o[-1 -2; -3 -4] := l_f[i][-1,1,3]*r[i][1,2,3,4,5,6,7,8]*conj(p_f[i][2,4,-2])*conj(l_f[i+1][-3,5,7])*p_f[i][6,8,-4] for i in 1:L]

#     return FiniteMPO(v)
# end

"""
    _otimes_n_a(x::Vector{<:MPOTensor}, y::Vector{<:MPOTensor}, f; right=nothing)
if f = ⊠, right is not used. (Example: one of x or y is strict)
if f = ⊗, if right is nothing, simply fusing the boundaries, else right should be a space, meaning selecting right after fusing the boundaries. 
(Example: both x and y are not strict, but the result is strict after selecting vacuum)
"""
function _otimes_n_a(x::Vector{<:MPOTensor}, y::Vector{<:MPOTensor}, f; right=oneunit(spacetype(y[1]))')
    L = length(x)
    r = [f(xj, permute(yj', (3,4), (1,2)) ) for (xj, yj) in zip(x, y)]
    T = eltype(r[1])

    p_f = [isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4))) for m in r]
    l_f = PeriodicArray([isomorphism(Matrix{T}, fuse(space(m, 1), space(m, 3)), space(m, 1) ⊗ space(m, 3) ) for m in r])

    if f === ⊗
        # we should do something special at the end for ⊗
        # a sector has to be chosen, I choose the vacuum sector
        v=[@tensor o[-1 -2; -3 -4] := l_f[i][-1,1,3]*r[i][1,2,3,4,5,6,7,8]*conj(p_f[i][2,4,-2])*conj(l_f[i+1][-3,5,7])*p_f[i][6,8,-4] for i in 1:L-1]
        i = L
        if !isnothing(right)
            # default oneunit(spacetype(r[i]))'
            right_ts = isomorphism(Matrix{T}, right ⊗ space(r[i], 5)', space(r[i], 7) ) 
            @tensor tmp[-1 -2; -3 -4] := l_f[i][-1,1,3]*r[i][1,2,3,4,5,6,7,8]*conj(p_f[i][2,4,-2])*right_ts[-3,5,7]*p_f[i][6,8,-4]        
        else
             right_ts = isomorphism(Matrix{T}, fuse(space(r[i], 5), space(r[i], 7)) , space(r[i], 5) ⊗ space(r[i], 7) ) 
            @tensor tmp[-1 -2; -3 -4] := l_f[i][-1,1,3]*r[i][1,2,3,4,5,6,7,8]*conj(p_f[i][2,4,-2])*right_ts[-3,5,7]*p_f[i][6,8,-4]              
        end
        push!(v, tmp)
    else
        v=[@tensor o[-1 -2; -3 -4] := l_f[i][-1,1,3]*r[i][1,2,3,4,5,6,7,8]*conj(p_f[i][2,4,-2])*conj(l_f[i+1][-3,5,7])*p_f[i][6,8,-4] for i in 1:L]
    end
    return v   
end

# we allow the case where x, y are nonstrict and f=⊗
function _otimes(x::FiniteMPO, y::ConjugateFiniteMPO, f; kwargs...)
    ((f === ⊠) || (f === ⊗)) || throw(ArgumentError("fuser should be ⊗ or ⊠."))
    (length(x) == length(y)) || throw(DimensionMismatch())
    yp = y.parent
    if f === ⊠
        (isstrict(x) && isstrict(yp)) || throw(ArgumentError("only strict MPOs allowed."))
    end       
    v = _otimes_n_a(raw_data(x), raw_data(yp), f; kwargs...)
    return FiniteMPO(v)
end

"""
    TensorKit.:⊗(x::FiniteMPO, y::Union{FiniteMPO, ConjugateFiniteMPO}) 
x⊗y or x⊗conj(y)
"""
TensorKit.:⊗(x::FiniteMPO, y::Union{FiniteMPO, ConjugateFiniteMPO}; kwargs...) = _otimes(x, y, ⊗; kwargs...)
TensorKit.:⊗(x::Union{AdjointFiniteMPO, ConjugateFiniteMPO}, args...) = error("A† ⊗ B not defined, only A ⊗ B and A ⊗ B† allowed.")
TensorKit.:⊠(x::FiniteMPO, y::Union{FiniteMPO, ConjugateFiniteMPO}; kwargs...) = _otimes(x, y, ⊠)
TensorKit.:⊠(x::Union{AdjointFiniteMPO, ConjugateFiniteMPO}, args...) = error("A† ⊠ B not defined, only A ⊠ B and A ⊠ B† allowed.")


