
abstract type AbstractQuantumOperator end

struct QOperator{S <: EuclideanSpace, M<:MPOTensor} <: AbstractQuantumOperator
	physpaces::Vector{Union{S, Missing}}
	data::Dict{Tuple{Int, Vararg{Int, N} where N}, Vector{Tuple{Vector{M}, AbstractCoefficient}}}
end
QOperator{S, M}() where {S <: EuclideanSpace, M <: MPOTensor} = QOperator{S, M}(Vector{Vector{Union{S, Missing}}}(), 
	Dict{Tuple{Int, Vararg{Int, N} where N}, Vector{Tuple{Vector{M}, AbstractCoefficient}}}())
QOperator{S, M}(physpaces::Vector{Union{S, Missing}}) where {S <: EuclideanSpace, M <: MPOTensor} = QOperator{S, M}(
	physpaces, Dict{Tuple{Int, Vararg{Int, N} where N}, Vector{Tuple{Vector{M}, AbstractCoefficient}}}())

function QOperator{S, M}(ms::Vector{<:QTerm}) where {S <: EuclideanSpace, M <: MPOTensor}
	r = QOperator{S, M}()
	for item in ms
		add!(r, item)
	end
	return r
end

function QOperator(ms::Vector{<:QTerm})
	isempty(ms) && throw(ArgumentError("no terms."))
	S = spacetype(ms[1])
	T = Float64
	for item in ms
		T = promote_type(T, scalar_type(item))
	end
	M = tensormaptype(S, 2, 2, T)
	return QOperator{S, M}(ms)
end

scalar_type(::Type{QOperator{S, M}}) where {S <: EuclideanSpace, M <: MPOTensor} = scalar_type(M)
scalar_type(x::QOperator) = scalar_type(typeof(x))

# QOperator{S}(physpaces::Vector) where {S <: EuclideanSpace} = QOperator{S, TensorMap{S, 2, 2}}(physpaces)
# QOperator{S}() where {S <: EuclideanSpace} = QOperator{S, TensorMap{S, 2, 2}}()
# QOperator{S,T}() where {S <: EuclideanSpace, T <: Number} = QOperator{S, tensormaptype(S, 2, 2, T)}()
# QOperator(physpaces::Vector{Union{S, Missing}}) where {S <: EuclideanSpace} = QOperator{S}(physpaces)


TensorKit.spacetype(::Type{QOperator{S, M}}) where {S, M} = S
TensorKit.spacetype(x::QOperator) = spacetype(typeof(x))
TensorKit.space(x::QOperator) = x.physpaces
TensorKit.space(x::QOperator, i::Int) = x.physpaces[i]

raw_data(x::QOperator) = x.data
Base.copy(x::QOperator) = QOperator(copy(space(x)), copy(raw_data(x)))

Base.empty!(s::QOperator) = empty!(raw_data(s))
Base.isempty(s::QOperator) = isempty(raw_data(s))
Base.keys(s::QOperator) = keys(raw_data(s))
Base.similar(s::QOperator{S, M}) where {S <: EuclideanSpace, M <: MPOTensor} = QOperator{S, M}()
Base.length(x::QOperator) = length(space(x))

function is_constant(x::QOperator)
	for (k, v) in raw_data(x)
		for (m, c) in v
			is_constant(c) || return false
		end
	end
	return true
end

interaction_range(x::QOperator) = maximum([_interaction_range(k) for k in keys(x)])

function (x::QOperator)(t::Number)
	r = typeof(raw_data(x))()
	for (k, v) in raw_data(x)
		vr =  typeof(v)()
		for (m, c) in v
			push!(vr, (m, coeff(c(t)) ))
		end
		r[k] = vr
	end
	return QOperator(copy(space(x)), r)
end 

function Base.:*(x::QOperator, y::Number) 
	r = typeof(raw_data(x))()
	for (k, v) in raw_data(x)
		vr =  typeof(v)()
		for (t, c) in v
			push!(vr, (t, c * y))
		end
		r[k] = vr
	end
	return QOperator(copy(space(x)), r)
end

Base.:*(y::Number, x::QOperator) = x * y
Base.:/(x::QOperator, y::Number) = x * (1 / y)
Base.:+(x::QOperator) = x
Base.:-(x::QOperator) = x * (-1)

function _merge_spaces(x, y)
	L = max(length(x), length(y))
	r = copy(x)
	resize!(r, L)
	for i in 1:L
		if i <= length(y)
			if ismissing(r[i])
				if !ismissing(y[i])
					r[i] = y
				end
			else
				if !ismissing(y[i])
					(r[i] == y[i]) || throw(SpaceMismatch())
				end
			end
		end
	end
	return r
end

function Base.:+(x::M, y::M) where {M <: QOperator}
	z = QOperator(_merge_spaces(space(x), space(y)), copy(raw_data(x)))
	for (k, v) in raw_data(y)
	   	tmp = get!(raw_data(z), k, typeof(v)())
	   	append!(tmp, v)
	end	
	return z
end
Base.:-(x::M, y::M) where {M <: QOperator} = x + (-y)

function shift(x::QOperator, n::Int)
	s = similar(space(x), length(x) + n)
	s[1:n] .= missing
	s[(n+1):end] .= space(x)
	return QOperator(s, typeof(raw_data(x))( k .+ n =>v for (k, v) in raw_data(x)) )
end 


"""
	add!(x::QOperator{M}, m::QTerm{M}) where {M <: MPOTensor}
	adding a new term into the quantum operator
"""
function add!(x::QOperator, m::QTerm) 
	(spacetype(x) == spacetype(m)) || throw(SpaceMismatch())
	is_zero(m) && return
	pos = Tuple(positions(m))
	L = length(x)
	if pos[end] > L
		resize!(x.physpaces, pos[end])
		for i in L+1:pos[end]
			x.physpaces[i] = missing
		end
	end
	for i in 1:length(pos)
		if ismissing(space(x, pos[i]))
			x.physpaces[pos[i]] = space(op(m)[i], 2)
		else
			(x.physpaces[pos[i]] == space(op(m)[i], 2)) || throw(SpaceMismatch())
		end
	end
	x_data = raw_data(x)
	v = get!(x_data, pos, valtype(x_data)())
	push!(v, (op(m), coeff(m)))
	return x
end  

function isstrict(x::QOperator)
	for (k, v) in raw_data(x)
		for (m, c) in v
			isstrict(QTerm(k, m, coeff=c)) || return false
		end
	end
	return true
end

function qterms(x::QOperator) 
	r = []
	for (k, v) in raw_data(x)
		for (m, c) in v
			a = QTerm(k, m, coeff=c)
			if !is_zero(a)
				push!(r, a)
			end
		end
	end
	return r
end

function qterms(x::QOperator, k::Tuple)
	r = []
	v = get(raw_data(x), k, nothing)
	if isnothing(v)
		return r
	else
		for (m, c) in v
			a = QTerm(k, m, coeff=c)		
			if !is_zero(a)
				push!(r, a)
			end
		end
	end
	return r
end

function _expm(x::QOperator{S}, dt::Number) where {S <: EuclideanSpace}
	is_constant(x) || throw(ArgumentError("input operator should be constant."))
	r = QuantumCircuit{S}()
	for k in keys(x)
	    m = qterms(x, k)
	    v = nothing
	    for item in m
	    	tmp = _join_ops(item)
	    	if !isnothing(tmp)
	    		if isnothing(v)
	    			v = tmp
	    		else
	    			v += tmp
	    		end
	    	end
	    end
	    if !isnothing(v)
	    	push!(r, QuantumGate(k, exp!(v * dt)))
	    end
	end
	return r
end

function _absorb_one_bodies(physpaces::Vector, h::Dict)
	r = typeof(h)()
	L = length(physpaces)
	(L >= 2) || throw(ArgumentError("operator should at least have two sites."))
	any(ismissing.(physpaces)) && throw(ArgumentError("hamiltonian has missing spaces."))
	for (key, value) in h
		if length(key)==1
			i = key[1]
			if i < L
				iden = id(oneunit(physpaces[i+1]) ⊗ physpaces[i+1])
				m = typeof(value)( [([ item[1], iden], c) for (item, c) in value ] )
				hj = get(r, (i, i+1), nothing)
				if hj == nothing
					r[(i, i+1)] = m
				else
					append!(r[(i, i+1)], m)
				end
			else
				iden = id(oneunit(physpaces[i-1]) ⊗ physpaces[i-1])
				m = typeof(value)( [([ iden, item[1]], c) for (item, c) in value ] )
				hj = get(r, (i-1, i), nothing)
				if hj == nothing
					r[(i-1, i)] = m
				else
					append!(r[(i-1, i)], m)
				end
			end
		else
			hj = get(r, key, nothing)
			if hj == nothing
				r[key] = value
			else
				append!(r[key], value)
			end
		end
	end
	return r
end

function absorb_one_bodies(h::QOperator) 
	r = _absorb_one_bodies(space(h), raw_data(h))
	return QOperator(copy(space(h)), r)
end
