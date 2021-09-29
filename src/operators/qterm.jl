abstract type AbstractQuantumTerm end


function _is_positions_stricted_order(pos::Vector{Int})
	(length(Set(pos)) == length(pos)) || throw(ArgumentError("duplicate positions not allowed"))
	return sort(pos) == pos
end


struct QTerm{M <: MPOTensor} <: AbstractQuantumTerm 
	positions::Vector{Int}
	op::Vector{M}
	coeff::AbstractCoefficient


function QTerm{M}(pos::Vector{Int}, m::Vector, v::Coefficient) where {M <: MPOTensor}
	# checks
	(length(pos) == length(m)) || throw(DimensionMismatch())
	isempty(m) && throw(ArgumentError("no input."))
	_is_positions_stricted_order(pos)
	all(_check_mpo_tensor_dir.(m)) || throw(SpaceMismatch())
	_check_mpo_space(m, strict=false)
	return new{M}(pos, m, v)
end

end

positions(x::QTerm) = x.positions
op(x::QTerm) = x.op
coeff(x::QTerm) = x.coeff

"""
	QTerm(pos::Vector{Int}, m::Vector{<:AbstractTensorMap}; coeff::Union{Number, Function, Coefficient}=1.) 
 	entrance to construct a single quantum term
"""
# entrance point
function QTerm(pos::Vector{Int}, m::Vector{<:AbstractTensorMap{S}}, v::Coefficient) where {S<:EuclideanSpace}
	L = length(m)
	T = promote_type(scalar_type(v), eltype.(m)...)
	M = tensormaptype(S, 2, 2, T)
	mpotensors = Vector{M}(undef, L)
	right = oneunit(S)
	for i in L:-1:1
		mj = m[i]
		if isa(mj, MPSBondTensor)
			tmp = id(Matrix{T}, right)
			@tensor mj[-1 -2; -3 -4] := tmp[-1, -3] * mj[-2, -4]
		end
		mpotensors[i] = mj
		right = space(mj, 1)
	end	
	return QTerm{M}(pos, mpotensors, v)
end 

QTerm(pos::Vector{Int}, m::Vector{<:AbstractTensorMap}; coeff::Union{Number, Function, Coefficient}=1.) = QTerm(pos, m, Coefficient(coeff))
QTerm(pos::Tuple, m::Vector{<:AbstractTensorMap}; coeff::Union{Number, Function, Coefficient}=1.) = QTerm([pos...], m; coeff=coeff)
QTerm(pos::Vector{Int}, m::Vector{<:AbstractSiteOperator}; coeff::Union{Number, Function, Coefficient}=1.) = QTerm(pos, raw_data.(m); coeff=coeff)

(x::QTerm)(t::Number) = QTerm(positions(x), op(x), coeff(x)(t))

QTerm(pos::Int, m::AbstractTensorMap; coeff::Union{Number, Function, Coefficient}=1.) = QTerm([pos], [m], coeff=coeff)

function _parse_pairs(x::Pair...)
	pos = Int[]
	ms = []
	for (a, b) in x
		push!(pos, a)
		push!(ms, b)
	end
	return pos, [ms...]
end

function QTerm(x::Pair{Int}...; coeff::Union{Number, Function, Coefficient}=1.) 
	pos, ms = _parse_pairs(x...)
	return QTerm(pos, ms; coeff=coeff)
end 


Base.copy(x::QTerm) = QTerm(copy(positions(x)), copy(op(x)), copy(coeff(x)))
Base.isempty(x::QTerm) = isempty(op(x))

Base.:*(s::QTerm, m::Union{Number, Coefficient}) = QTerm(positions(s), op(s), coeff(s) * m)
Base.:*(m::Union{Number, Coefficient}, s::QTerm) = s * m
Base.:/(s::QTerm, m::Union{Number, Coefficient}) = QTerm(positions(s), op(s), coeff(s) / m)
Base.:+(s::QTerm) = s
Base.:-(s::QTerm) = QTerm(positions(s), op(s), -coeff(s))


TensorKit.spacetype(::Type{QTerm{M}}) where {M <: MPOTensor} = spacetype(M)
TensorKit.spacetype(x::QTerm) = spacetype(typeof(x))

nterms(s::QTerm) = length(op(s))
is_constant(s::QTerm) = is_constant(coeff(s))
function scalar_type(x::QTerm)
	T = scalar_type(coeff(x))
	for m in op(x)
		T = promote_type(T, eltype(m))
	end
	return T
end 

function _interaction_range(x::Union{Vector{Int}, Tuple})::Int
	(length(x) == 0) && return 0
	(length(x)==1) && return 1
	return x[end] - x[1] + 1
end

interaction_range(x::QTerm) = _interaction_range(positions(x))


shift(m::QTerm, i::Int) = QTerm(positions(m) .+ i, op(m), coeff(m))

function is_zero(x::QTerm) 
	is_zero(coeff(x)) && return true
	isempty(x) && return true
	for item in op(x)
	    is_zero(item) && return true
	end
	return false
end

function isstrict(t::QTerm)
	isempty(t) && return true
	is_zero(t) && return true
	m = op(t)[end]
	return space(m, 3)' == oneunit(space(m, 3))
end


function _join(v::Vector{<:MPOTensor})
	isempty(v) && throw(ArgumentError())
	util = Tensor(ones,eltype(v[1]),oneunit(space(v[1],1)))
	if length(v) == 1
		@tensor r[-1 ; -2] := conj(util[1]) * v[1][1,-1,2,-2] * util[2]
	elseif length(v) == 2
		@tensor r[-1 -2; -3 -4] := conj(util[1]) * v[1][1,-1,2,-3] * v[2][2,-2,3,-4] * util[3]
	elseif length(v) == 3
		@tensor r[-1 -2 -3; -4 -5 -6] := conj(util[1]) * v[1][1,-1,2,-4] * v[2][2,-2,3,-5] * v[3][3,-3,4,-6] * util[4]
	elseif length(v) == 4
		@tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := conj(util[1]) * v[1][1,-1,2,-5] * v[2][2,-2,3,-6] * v[3][3,-3,4,-7] * v[4][4,-4,5,-8] * util[5]
	else
		throw(ArgumentError("only support up to 4-body terms."))
	end
	return r
end

function _join_ops(m::QTerm)
	is_constant(m) || error(ArgumentError("functional term not allowed."))
	is_zero(m) && return nothing
	isstrict(m) || error(ArgumentError("only number conserving term allowed."))
	return _join(op(m)) * value(coeff(m))
end


function _coerce_qterms(x::QTerm, y::QTerm)
	T = promote_type(scalar_type(x), scalar_type(y))
    opx = op(x)
    opy = op(y)
    pos = positions(x)
    vacuum = oneunit(spacetype(x))
    if !(positions(x) == positions(y))
    	new_pos = sort([Set(vcat(positions(x), positions(y)))...])
    	new_opx = []
    	new_opy = []
    	for pos in new_pos
    		pos_x = findfirst(a->a==pos, positions(x))
    		pos_y = findfirst(a->a==pos, positions(x))
    		if isnothing(pos_x) && !(isnothing(pos_y))
    			push!(new_opx, id(Matrix{T}, vacuum ⊗ space(opy[pos_y], 2) ))
    			push!(new_opy, opy[pos_y])
    		elseif !(isnothing(pos_x)) && isnothing(pos_y)
    			push!(new_opx, opx[pos_x])
    			push!(new_opy, id(Matrix{T}, vacuum ⊗ space(opx[pos_x], 2)))
    		elseif !(isnothing(pos_x)) && !(isnothing(pos_y))
    			push!(new_opx, opx[pos_x])
    			push!(new_opy, opy[pos_y])
    		else
    			throw(ArgumentError("why here?"))
    		end
    	end
    	opx = [new_opx...]
    	opy = [new_opy...]
    	pos = new_pos
    end
    return pos, opx, opy
end


# Qterm adjoint
struct AdjointQTerm{M <: MPOTensor} <: AbstractQuantumTerm 
	parent::QTerm{M}
end

Base.adjoint(m::QTerm) = AdjointQTerm(m)
Base.adjoint(m::AdjointQTerm) = m.parent
positions(m::AdjointQTerm) = positions(m.parent)
coeff(m::AdjointQTerm) = conj(coeff(m.parent))
scalar_type(m::AdjointQTerm) = scalar_type(m.parent)
is_constant(s::AdjointQTerm) = is_constant(s.parent)


function QTerm(m::AdjointQTerm)
	isstrict(m.parent) || throw(ArgumentError("can not convert non-strict QTerm adjoint into a QTerm."))
	QTerm(positions(m.parent), mpo_tensor_adjoint.(op(m.parent)), conj(coeff(m.parent)) )
end 


"""
	Base.:*(x::QTerm, y::AdjointQTerm)
multiplication between two QTerms
"""
function Base.:*(x::QTerm, y::AdjointQTerm)
	pos, opx, opy = _coerce_qterms(x, y.parent)
	v = _mult_n_a(opx, opy, oneunit(spacetype(x))')
	return QTerm(pos, v, coeff=coeff(x)*coeff(y))
end
function Base.:*(x::AdjointQTerm, y::QTerm)
	pos, opx, opy = _coerce_qterms(x.parent, y)
	v = _mult_a_n(opx, opy, oneunit(spacetype(y))')
	return QTerm(pos, v, coeff=coeff(x)*coeff(y))
end
function Base.:*(x::QTerm, y::QTerm)
	pos, opx, opy = _coerce_qterms(x, y.parent)
	v = _mult_n_n(opx, opy)
	return QTerm(pos, v, coeff=coeff(x)*coeff(y))
end
Base.:*(x::AdjointQTerm, y::AdjointQTerm) = adjoint(y.parent * x.parent)




