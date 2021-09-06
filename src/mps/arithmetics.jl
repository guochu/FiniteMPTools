


function LinearAlgebra.dot(psiA::FiniteMPS, psiB::FiniteMPS) 
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

distance2(a::FiniteMPS, b::FiniteMPS) = _distance2(a, b)
distance(a::FiniteMPS, b::FiniteMPS) = _distance(a, b)


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
function Base.:+(psiA::FiniteMPS, psiB::FiniteMPS) 
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
Base.:-(psiA::FiniteMPS, psiB::FiniteMPS) = psiA + (-1) * psiB
Base.:-(psi::FiniteMPS) = -1 * psi

const MPS_APPROX_EQUAL_ATOL = 1.0e-14
Base.isapprox(psiA::M, psiB::M; atol=MPS_APPROX_EQUAL_ATOL) where {M <: FiniteMPS} = distance2(psiA, psiB) <= atol


# kronecker operations for MPSs

function is_diagonal_sector(sector::Sector)
    ss = sector.sectors
    L = length(ss)
    (mod(L, 2) == 0) || throw(DimensionMismatch())
    return ss[1:div(L, 2)] == dual.(ss[div(L, 2)+1:L])
end
select_diagonal(s::S) where {S <: EuclideanSpace} = S(sector=>dim(s, sector) for sector in sectors(s) if is_diagonal_sector(sector))

function _boxtimes_iden_mps(::Type{T}, physpaces::Vector{S}, sector::Sector) where {T, S <: GradedSpace}
    L = length(physpaces)
    left = oneunit(S)
    right = S(sector=>1)
    virtualpaces = Vector{S}(undef, L+1)
    virtualpaces[1] = left
    for i in 2:L
        virtualpaces[i] = select_diagonal(fuse(virtualpaces[i-1], physpaces[i-1]))
        virtualpaces[i] = S(item=>1 for item in sectors(virtualpaces[i]))
    end
    virtualpaces[L+1] = right
    for i in L:-1:2
        virtualpaces[i] = infimum(virtualpaces[i], select_diagonal(fuse(physpaces[i]', virtualpaces[i+1])) )
        virtualpaces[i] = S(item=>1 for item in sectors(virtualpaces[i]))
    end
    return FiniteMPS([TensorMap(ones, T, virtualpaces[i] ⊗ physpaces[i], virtualpaces[i+1]) for i in 1:L])
end

function _boxtimes_iden_mps(::Type{T}, physpaces::Vector{S}, sector::Sector) where {T, S <: Union{ComplexSpace, CartesianSpace}}
    L = length(physpaces)
    mpstensors = Vector{Any}(undef, L)
    for i in 1:L
        d2 = dim(physpaces[i])
        d = round(Int, sqrt(d2))
        (d*d == d2) || throw(ArgumentError("space can not be interpretted as open."))
        mpstensors[i] = TensorMap(reshape(one(zeros(T, d, d)), 1, d2, 1), oneunit(S) ⊗ physpaces[i], oneunit(S) ) 
    end
    return FiniteMPS([mpstensors...])
end

"""
    infinite_temperature_state(::Type{T}, physpaces::Vector{S}; sector::Sector, fuser=⊠) where {T<:Number, S<:EuclideanSpace}
    create an infinite temperature state (identity) as mps, namely |I⟩
"""
function infinite_temperature_state(::Type{T}, physpaces::Vector{S}; sector::Sector, fuser=⊠) where {T<:Number, S<:EuclideanSpace}
     ((fuser === ⊠) || (fuser === ⊗)) || throw(ArgumentError("fuser should be ⊗ or ⊠."))
     phy_fusers = [isomorphism(Matrix{T}, ph ⊗ ph', fuse(ph, ph') ) for ph in physpaces]
     if fuser === ⊠
         iden = _boxtimes_iden_mps(T, physical_spaces, sector)
     else
        tmp = Tensor(ones,T, oneunit(S))
        iden = FiniteMPS([@tensor o[-1 -2; -3] := tmp[-1] * conj(fj[1,1,-2]) * conj(tmp[-3]) for fj in phy_fusers])
     end
     return FiniteDensityOperatorMPS(iden, phy_fusers, iden)
end
infinite_temperature_state(physpaces::Vector{S}; kwargs...) where {S<:EuclideanSpace} = infinite_temperature_state(ComplexF64, physpaces; kwargs...)

# initializing density operator
# f is ⊗ or ⊠
function _otimes(x::FiniteMPS, y::AdjointFiniteMPS, f, trunc::TruncationScheme)
    (length(x) == length(y)) || throw(DimensionMismatch())
    isempty(x) && throw(ArgumentError("input is empty."))
    ((f === ⊠) || (f === ⊗)) || throw(ArgumentError("fuser should be ⊗ or ⊠."))
    yp = y.parent
    (isstrict(x) && isstrict(yp)) || throw(ArgumentError("only strict MPSs allowed."))
    T = promote_type(scalar_type(x), scalar_type(yp))
    L = length(x)

    # first site
    m = f(x[1], permute(yp[1]', (2, 3), (1,)))
    left = isomorphism(Matrix{T}, fuse(space(m, 1), space(m, 3)),  space(m, 1) ⊗ space(m, 3) )
    phy_fuser = isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4)) )
    phy_fusers = [phy_fuser]

    @tensor tmp[-1 -2; -3 -4] := left[-1, 1, 3] * m[1, 2, 3, 4, -3, -4] * conj(phy_fuser[2,4,-2])
    u, s, v, err = tsvd!(tmp, trunc=trunc)
    r = Vector{typeof(u)}(undef, L)
    r[1] = u
    v = s * v
    # middle sites
    for i in 2:L-1
        m = f(x[i], permute(yp[i]', (2, 3), (1,)))
        phy_fuser = isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4)) )
        push!(phy_fusers, phy_fuser)
        @tensor tmp[-1 -2; -3 -4] := v[-1, 1, 3] * m[1, 2, 3, 4, -3, -4] * conj(phy_fuser[2,4,-2])
        u, s, v, err = tsvd!(tmp, trunc=trunc)
        r[i] = u
        v = s * v
    end

    # last site
    m = f(x[L], permute(yp[L]', (2, 3), (1,)))
    phy_fuser = isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4)) )
    push!(phy_fusers, phy_fuser)

    right = isomorphism(Matrix{T}, space(m, 5)' ⊗ space(m, 6)', fuse(space(m, 5)', space(m, 6)') )
    if f === ⊗
        # a sector has to be chosen, I choose the vacuum sector
        right = isomorphism(Matrix{T}, space(m, 5)', space(m, 6) ⊗ oneunit(spacetype(m)) ) 
    end
    @tensor tmp[-1 -2; -3] := v[-1, 1, 3] * m[1, 2, 3, 4, 5, 6] * conj(phy_fuser[2,4,-2]) * right[5, 6, -3]
    r[L] = tmp
    mpsout = FiniteMPS(r)
    rightorth!(mpsout, trunc=trunc)

    if f === ⊠
        iden = _boxtimes_iden_mps(T, physical_spaces(mpsout), sector(mpsout))
    else
        # f === ⊗
        tmp = get_trivial_leg(mpsout[1])
        iden = FiniteMPS([@tensor o[-1 -2; -3] := tmp[-1] * conj(fj[1,1,-2]) * conj(tmp[-3]) for fj in phy_fusers])
    end
    return FiniteDensityOperatorMPS(mpsout, phy_fusers, iden)
end


# f is ⊗ or ⊠
function _otimes(x::FiniteMPS, y::FiniteMPS, f, trunc::TruncationScheme)
    (length(x) == length(y)) || throw(DimensionMismatch())
    isempty(x) && throw(ArgumentError("input is empty."))
    (isstrict(x) && isstrict(y)) || throw(ArgumentError("only strict MPSs allowed."))
    T = promote_type(scalar_type(x), scalar_type(y))
    L = length(x)

    # first site
    m = f(x[1], y[1])
    left = isomorphism(Matrix{T}, fuse(space(m, 1), space(m, 3)),  space(m, 1) ⊗ space(m, 3) )
    phy_fuser = isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4)) )

    @tensor tmp[-1 -2; -3 -4] := left[-1, 1, 3] * m[1, 2, 3, 4, -3, -4] * conj(phy_fuser[2,4,-2])
    u, s, v, err = tsvd!(tmp, trunc=trunc)
    r = Vector{typeof(u)}(undef, L)
    r[1] = u
    v = s * v
    # middle sites
    for i in 2:L-1
        m = f(x[i], y[i])
        phy_fuser = isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4)) )
        @tensor tmp[-1 -2; -3 -4] := v[-1, 1, 3] * m[1, 2, 3, 4, -3, -4] * conj(phy_fuser[2,4,-2])
        u, s, v, err = tsvd!(tmp, trunc=trunc)
        r[i] = u
        v = s * v
    end

    # last site
    m = f(x[L], y[L])
    phy_fuser = isomorphism(Matrix{T}, space(m, 2) ⊗ space(m, 4), fuse(space(m, 2), space(m, 4)) )
    right = isomorphism(Matrix{T}, space(m, 5)' ⊗ space(m, 6)', fuse(space(m, 5)', space(m, 6)') )
    @tensor tmp[-1 -2; -3] := v[-1, 1, 3] * m[1, 2, 3, 4, 5, 6] * conj(phy_fuser[2,4,-2]) * right[5, 6, -3]
    r[L] = tmp
    mpsout = FiniteMPS(r)
    rightorth!(mpsout, trunc=trunc)
    return mpsout
end

# This is how to generate density operator from unitary mps

"""
    TensorKit.:⊗(x::FiniteMPS, y::FiniteMPS; trunc::TruncationScheme=default_truncation(spacetype(x))) 
    TensorKit.:⊗(x::FiniteMPS, y::AdjointFiniteMPS; trunc::TruncationScheme=default_truncation(spacetype(x)))
    kronecker of two mps is another mps
    kronecker of an mps with a conjugate mps is a density operator
    kronecker of a conjugate mps with an mps is not deifned
    A bare kronecker operator may produce a huge MPS, thus we truncate it and return
"""
TensorKit.:⊗(x::FiniteMPS, y::FiniteMPS; trunc::TruncationScheme=default_truncation(spacetype(x))) = _otimes(x, y, ⊗, trunc)
TensorKit.:⊗(x::FiniteMPS, y::AdjointFiniteMPS; trunc::TruncationScheme=default_truncation(spacetype(x))) = _otimes(x, y, ⊗, trunc)
TensorKit.:⊗(x::AdjointFiniteMPS, y::FiniteMPS; kwargs...) = error("<x| ⊗ |y> not defined, you may want to reverse the order to do |x> ⊗ <y| instead?")

"""
    TensorKit.:⊠(x::FiniteMPS, y::AdjointFiniteMPS; trunc::TruncationScheme=default_truncation(spacetype(x)))
    A bare kronecker operator may produce a huge MPS, thus we truncate it and return
"""
TensorKit.:⊠(x::FiniteMPS, y::AdjointFiniteMPS; trunc::TruncationScheme=default_truncation(spacetype(x))) = _otimes(x, y, ⊠, trunc)
TensorKit.:⊠(x::FiniteMPS, y::FiniteMPS; trunc::TruncationScheme=default_truncation(spacetype(x))) = _otimes(x, y, ⊠, trunc)
TensorKit.:⊠(x::AdjointFiniteMPS, y::FiniteMPS; kwargs...) = error("<x| ⊠ |y> not defined, you may want to reverse the order to do |x> ⊠ <y| instead?")

"""
    DensityOperator(psi::FiniteMPS; fuser=⊠, kwargs...)
    return ρ = |ψ⟩⟨ψ|
"""
DensityOperator(psi::FiniteMPS; fuser=⊠, kwargs...) = fuser(psi, psi'; kwargs...)
