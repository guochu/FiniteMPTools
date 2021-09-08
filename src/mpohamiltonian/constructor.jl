



function generate_Fmat(fvec::Vector{<:Number}, n::Int)
    L = length(fvec)
    (L >= n) || error("number of sites must be larger than number of terms in expansion.")
    F = zeros(eltype(fvec), L-n+1, n)
    for j in 1:n
        for i in 1:L-n+1
            F[i, j] = fvec[i + j - 1]
        end
    end
    return F
end

generate_Fmat(f, L::Int, n::Int) = generate_Fmat([f(k) for k in 1:L], n)

function exponential_expansion(fmat::AbstractMatrix)
    s1, n = size(fmat)
    (s1 >= n) || error("wrong input, try increase L, or decrease the tolerance.")
    L = s1 - 1 + n
    _u, _v = LinearAlgebra.qr(fmat)
    U = Matrix(_u)
    V = Matrix(_v)
    U1 = U[1:L-n, :]
    U2 = U[(s1-L+n+1):s1, :]
    m = LinearAlgebra.pinv(U1) * U2
    lambdas = LinearAlgebra.eigvals(m)
    (length(lambdas) == n) || error("something wrong..")
    m = zeros(eltype(lambdas), L, n)
    for j in 1:n
        for i in 1:L
            m[i, j] = lambdas[j]^i
        end
    end
    fvec = zeros(eltype(fmat), L)
    for i in 1:n
        fvec[i] = fmat[1, i]
    end
    for i in n+1:L
        fvec[i] = fmat[i-n+1, n]
    end
    xs = m \ fvec
    # err = norm(m * xs - fvec)
    err = maximum(abs.(m * xs - fvec))
    return  xs, lambdas, err
end

exponential_expansion_n(f, L::Int, n::Int) = exponential_expansion(generate_Fmat(f, L, n))
function exponential_expansion(f, L::Int; atol::Real=1.0e-5)
    for n in 1:L
        xs, lambdas, err = exponential_expansion_n(f, L, n)
        if err <= atol
            # println("converged $n iterations, error is $err.")
            return xs, lambdas
        end
        if n >= L-n+1
            @warn "can not converge to $atol with size $L, try increase L, or decrease the tolerance."
            return xs, lambdas
        end
    end
    error("can not find a good approximation.")
end



abstract type AbstractLongRangeTerm end

# coeff * α^n, α must be in [0, 1]
struct ExponentialDecayTerm{M1 <: AbstractSiteOperator, M<:ScalarSiteOp, M2 <: AbstractSiteOperator, T <:Number} <: AbstractLongRangeTerm
    a::M1
    m::M
    b::M2
    α::T
    coeff::T
end

function ExponentialDecayTerm(a::Union{ScalarSiteOp, SiteOp}, m::ScalarSiteOp, b::AbstractSiteOperator; α::Number=1., coeff::Number=1.) 
    T = promote_type(typeof(α), typeof(coeff))
    return ExponentialDecayTerm(a, m, b, convert(T, α), convert(T, coeff))
end
function ExponentialDecayTerm(a::Union{ScalarSiteOp, SiteOp}, b::AbstractSiteOperator; α::Number=1., coeff::Number=1.) 
	T = promote_type(eltype(a), eltype(b), typeof(α), typeof(coeff))
    m = ScalarSiteOp(id(Matrix{T}, physical_space(a)))
	return ExponentialDecayTerm(a, m, b, α=α, coeff=coeff)
end

coeff(x::ExponentialDecayTerm) = x.coeff
scalar_type(::Type{ExponentialDecayTerm{M1, M, M2, T}}) where {M1, M, M2, T} = promote_type(eltype(M1), eltype(M), eltype(M2), T)
scalar_type(x::ExponentialDecayTerm) = scalar_type(typeof(x))

_op_adjoint(a::ScalarSiteOp, m::ScalarSiteOp, b::ScalarSiteOp) = (a', m', b')
_op_adjoint(a::SiteOp, m::ScalarSiteOp, b::SiteOp) = (SiteOp(mpo_tensor_adjoint(a.op)), m', SiteOp(mpo_tensor_adjoint(b.op)))
_op_adjoint(a::SiteOp, m::ScalarSiteOp, b::AdjointSiteOp) = (SiteOp(mpo_tensor_adjoint(a.op)), m', SiteOp(mpo_tensor_adjoint(b.parent.op))' )

Base.adjoint(x::ExponentialDecayTerm) = ExponentialDecayTerm(_op_adjoint(x.a, x.m, x.b)...; α=conj(x.α), coeff=conj(coeff(x)))

struct GenericDecayTerm{M1 <: AbstractSiteOperator, M<:ScalarSiteOp, M2<:AbstractSiteOperator, F, T <: Number} <: AbstractLongRangeTerm
    a::M1
    m::M
    b::M2
    f::F
    coeff::T
end

coeff(x::GenericDecayTerm) = x.coeff
GenericDecayTerm(a::Union{ScalarSiteOp, SiteOp}, m::ScalarSiteOp, b::AbstractSiteOperator; 
    f, coeff::Union{Number, Function, Coefficient}=1.) = GenericDecayTerm(a, m, b, f, coeff)
function GenericDecayTerm(a::Union{ScalarSiteOp, SiteOp}, b::AbstractSiteOperator; f, coeff::Union{Number, Function, Coefficient}=1.) 
    T = promote_type(eltype(a), eltype(b), typeof(f(0.)), typeof(coeff))
    m = ScalarSiteOp(id(Matrix{T}, physical_space(a)))
    return GenericDecayTerm(a, m, b, f=f, coeff=coeff)
end

scalar_type(x::GenericDecayTerm{M1, M, M2, F, T}) where {M1, M, M2, F, T} = promote_type(eltype(M1), eltype(M), eltype(M2), T, typeof(x.f(0.)))

Base.adjoint(x::GenericDecayTerm) = GenericDecayTerm(_op_adjoint(x.a, x.m, x.b)...; α=y->conj(x.f(y)), coeff=conj(coeff(x)))

"""
    coeff * n^α, α must be negative (diverging otherwise)
"""
PowerlawDecayTerm(a::Union{ScalarSiteOp, SiteOp}, m::ScalarSiteOp, b::AbstractSiteOperator; α::Number=1., coeff::Number=1.) = GenericDecayTerm(a, m, b, f=x->x^α, coeff=coeff)
PowerlawDecayTerm(a::Union{ScalarSiteOp, SiteOp}, b::AbstractSiteOperator; α::Number=1., coeff::Number=1.) = GenericDecayTerm(a, b, f=x->x^α, coeff=coeff)



# L is the number of sites
function exponential_expansion(x::GenericDecayTerm{M, T, F}; len::Int, atol::Real=1.0e-5) where {M, T, F}
    xs, lambdas = exponential_expansion(x.f, len, atol=atol)
    r = []
    for (c, alpha) in zip(xs, lambdas)
        push!(r, ExponentialDecayTerm(x.a, x.m, x.b; α=alpha, coeff=c * coeff(x)))
    end
    return [r...]
end

function _longrange_schurmpo_util(h1, h2s::Vector{<:ExponentialDecayTerm})
    isempty(h2s) && throw(ArgumentError("empty interactions."))
	pspace = physical_space(h2s[1].a)
	N = length(h2s)
	T = Float64
	for item in h2s
		T = promote_type(T, scalar_type(item))
	end
	cell = Matrix{Any}(undef, N+2, N+2)
	for i in 1:length(cell)
		cell[i] = zero(T)
	end
	# diagonals
	iden = id(Matrix{T}, pspace)
	cell[1, 1] = iden
	cell[end, end] = iden
	cell[1, end] = h1
	for i in 1:N
        if isa(h2s[i].a, ScalarSiteOp)
            b_iden = id(Matrix{T}, oneunit(spacetype(h2s[i].a)))
        else
            b_iden = id(Matrix{T}, space_r(h2s[i].a)')
        end
        m = raw_data(h2s[i].m)
        @tensor iden[-1 -2; -3 -4] := b_iden[-1, -3] * m[-2, -4]
		cell[i+1, i+1] = h2s[i].α * iden
		cell[1, i+1] = h2s[i].coeff * raw_data(h2s[i].a) 
		cell[i+1, end] = h2s[i].α * raw_data(h2s[i].b)
	end
	return SchurMPOTensor(cell)
end
SchurMPOTensor(h1::ScalarSiteOp, h2s::Vector{<:ExponentialDecayTerm}) = _longrange_schurmpo_util(raw_data(h1), h2s)
SchurMPOTensor(h2s::Vector{<:ExponentialDecayTerm}) = _longrange_schurmpo_util(0., h2s)










