


function purified_thermalize(m::QTerm)
	pos, opx, opy = _coerce_qterms(m, id(m))
	v = _otimes_n_n(opx, opy, ⊠)
	return QTerm(pos, v, coeff=coeff(m))
end
purified_thermalize(m::AdjointQTerm) = purified_thermalize(m.parent)'
purified_thermalize(m::QuantumOperator) = QuantumOperator([purified_thermalize(item) for item in qterms(m)])
purified_thermalize(m::FiniteMPO) = m ⊠ id(m)
purified_thermalize(m::AdjointFiniteMPO) = purified_thermalize(m.parent)'

function purified_infinite_temperature_state(::Type{T}, physpaces::Vector{S}) where {T <: Number, S <: Union{CartesianSpace, ComplexSpace}}
    L = length(physpaces)
    mpstensors = Vector{Any}(undef, L)
    for i in 1:L
        d = dim(physpaces[i])
        d2 = d * d
        mpstensors[i] = TensorMap(reshape(one(zeros(T, d, d)), 1, d2, 1), oneunit(S) ⊗ (fuse(physpaces[i] ⊠ physpaces[i])), oneunit(S) ) 
    end
    return FiniteMPS([mpstensors...])
end


function purified_infinite_temperature_state(::Type{T}, physpaces::Vector{S}; left::S=oneunit(S), right::S=oneunit(S)) where {T <: Number, S <: GradedSpace}
    L = length(physpaces)
    leftspace = left ⊠ left
    S2 = typeof(S)
    physpaces2 = [S2(sector ⊠ sector => 1 for sector in sectors(item)) for item in physpaces]
    virtualpaces = Vector{S2}(undef, L+1)
    virtualpaces[1] = leftspace
    for i in 2:L
        virtualpaces[i] =  fuse(virtualpaces[i-1], physpaces2[i-1])
    end
    virtualpaces[L+1] = right ⊠ right
    for i in L:-1:2
        virtualpaces[i] = infimum(virtualpaces[i], select_diagonal(fuse(physpaces2[i]', virtualpaces[i+1])) )
        # virtualpaces[i] = S(item=>1 for item in sectors(virtualpaces[i]))
    end
    return FiniteMPS([TensorMap(ones, T, virtualpaces[i] ⊗ physpaces2[i], virtualpaces[i+1]) for i in 1:L])
end

_get_identity_state(h::Union{QuantumOperator, FiniteMPO}; kwargs...) = purified_infinite_temperature_state(scalar_type(h), physical_spaces(h); kwargs...)
function purified_thermal_state(h::Union{QuantumOperator, FiniteMPO}; β::Real, stepper::AbstractStepper=TEBDStepper(stepsize=0.05, order=4), kwargs...)
	(isa(stepper, TEBDStepper) && isa(h, FiniteMPO)) && throw(ArgumentError("TEBD can not be used with MPO."))
	beta = convert(Float64, β) / 2
	((beta >= 0.) && (beta != Inf)) || throw(ArgumentError("β expected to be finite."))	
	state = _get_identity_state(h; kwargs...)
	canonicalize!(state, normalize=true)
	(beta == 0.) && return state
	superh = purified_thermalize(h)
	delta = 0.1

	nsteps, delta = compute_step_size(beta, delta)
	local cache
	for i in 1:nsteps
		tspan = ( -(i-1) * delta, -i * delta )
		stepper = change_tspan_dt(stepper, tspan=tspan)
		(@isdefined cache) || (cache = timeevo_cache(superh, stepper, state))
		state, cache = timeevo!(state, superh, stepper, cache)
		canonicalize!(state, normalize=true)
	end
	# return normalize!(state)
	return state
end
function exact_purified_thermal_state(h::Union{QuantumOperator, FiniteMPO}; β::Real, kwargs...)
	beta = convert(Float64, β) / 2
	((beta >= 0.) && (beta != Inf)) || throw(ArgumentError("β expected to be finite."))	
	state = _get_identity_state(h; kwargs...)
	canonicalize!(state, normalize=true)
	state = ExactFiniteMPS(state)
	(beta == 0.) && return state
	superh = purified_thermalize(h)

	state = exact_timeevolution(FiniteMPO(superh), -beta, state, ishermitian=true)

	return normalize!(state)
end
# thermal_state(h::Union{QuantumOperator, FiniteMPO}; kwargs...) = purified_thermal_state(h; kwargs...)


