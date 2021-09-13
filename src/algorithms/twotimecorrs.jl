# two time correlation for pure state and density matrices

function _check_times(ts::Vector{<:Real})
	isempty(ts) && error("no times.")
	(ts[1] >= 0.) || error("the first time step should be non-negative.")
	for i in 2:length(ts)
		(ts[i] > ts[i-1]) || error("times must be monotonically increasing.")
	end
end

function _unitary_tt_corr_util(h, A, B, C, state, times, stepper)
	if isnothing(C)
		state_right = copy(state)
	else
		state_right = C * state
	end
	if isnothing(A)
		state_left = copy(state)
	else
		state_left = A' * state
	end

	result = []
	for i in 1:length(times)
		local cache_left, cache_right		
		tspan = (i == 1) ? (0., -im*times[1]) : (-im*times[i-1], -im*times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			stepper = change_tspan_dt(stepper, tspan=tspan)
			(@isdefined cache_left) || (cache_left = timeevo_cache(h, stepper, state_left))
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper, state_right))
			state_left, cache_left = timeevo!(state_left, h, stepper, cache_left)
			state_right, cache_right = timeevo!(state_right, h, stepper, cache_right)
		end
		push!(result, expectation(state_left, B, state_right))
	end
	return [result...]
end

function unitart_twotime_corr(h::QuantumOperator, a::Union{FiniteMPO, MPOHamiltonian}, b::Union{FiniteMPO, MPOHamiltonian}, 
	state::FiniteMPS, times::Vector{<:Real}; stepper::AbstractStepper=TEBDStepper(tspan=(0., -0.01*im), stepsize=0.01), reverse::Bool=false)
	_check_times(times)
	# isa(stepper, TDVPStepper) && (h = FiniteMPO(h, alg=SVDCompression()))
	reverse ? _unitary_tt_corr_util(h, a, b, nothing, state, times, stepper) : _unitary_tt_corr_util(h, nothing, a, b, state, times, stepper)
end

function _open_tt_corr_util(h, A, B, C, state, times, stepper)
	# isa(stepper, TDVPStepper) && (h = FiniteMPO(h, alg=SVDCompression()))
	if isnothing(C)
		state_right = copy(state)
	else
		state_right = C * state
	end
	if !isnothing(A)
		state_right = A * state_right
	end
	result = []
	# cache_right = timeevo_cache(h, stepper, state_right)
	for i in 1:length(times)
		local cache_right	
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if tspan[2] - tspan[1] > 0.
			stepper = change_tspan_dt(stepper, tspan=tspan)
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper, state_right))
			state_right, cache_right = timeevo!(state_right, h, stepper, cache_right)
		end
		push!(result, expectation(B, state_right) )
	end
	return [result...]
end

"""
	correlation_2op_1t(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::FiniteMPS, times::Vector{<:Real}, stepper::AbstractStepper; 
	reverse::Bool=false) 
	for a unitary system with hamiltonian h, compute <a(t)b> if revere=false and <a b(t)> if reverse=true
	for an open system with superoperator h, and a, b to be normal operators, compute <a(t)b> if revere=false and <a b(t)> if reverse=true.
	For open system see definitions of <a(t)b> or <a b(t)> on Page 146 of Gardiner and Zoller (Quantum Noise)
"""
correlation_2op_1t(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::FiniteMPS, times::Vector{<:Real}; kwargs...) = unitart_twotime_corr(
	h, FiniteMPO(a, alg=SVDCompression()), FiniteMPO(b, alg=SVDCompression()), state, times; kwargs...)
function correlation_2op_1t(h::SuperOperatorBase, a::QuantumOperator, b::QuantumOperator, state::FiniteDensityOperatorMPS, times::Vector{<:Real}; 
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	mpo_a = FiniteMPO(a, alg=SVDCompression())
	mpo_b = FiniteMPO(b, alg=SVDCompression())
	iden = id(mpo_a)
	reverse ? _open_tt_corr_util(h, h.fuser(iden, conj(FiniteMPO(mpo_a'))), h.fuser(mpo_b, conj(iden)), nothing, state, times, stepper) : _open_tt_corr_util(h
		, nothing, h.fuser(mpo_a, conj(iden)), h.fuser(mpo_b, conj(iden)), state, times, stepper)
end


function _exact_unitary_tt_corr_util(h, A, B, C, state, times)
	if isnothing(C)
		state_right = copy(state)
	else
		state_right = C * state
	end
	if isnothing(A)
		state_left = copy(state)
	else
		state_left = A' * state
	end	
	state_left = ExactFiniteMPS(state_left)
	state_right = ExactFiniteMPS(state_right)
	(state_left.center == state_right.center) || error("something wrong.")
	# isa(h, QuantumOperator) && (h = FiniteMPO(h))
	left, right = init_h_center(h, state_right)

	result = []
	for i in 1:length(times)
		tspan = (i == 1) ? (0., -im*times[1]) : (-im*times[i-1], -im*times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			state_left = _exact_timeevolution_util(h, tspan[2]-tspan[1], state_left, left, right, ishermitian=true)
			state_right = _exact_timeevolution_util(h, tspan[2]-tspan[1], state_right, left, right, ishermitian=true)
		end
		push!(result, expectation(FiniteMPS(state_left), B, FiniteMPS(state_right)))
	end
	return [result...]
end

function _exact_open_tt_corr_util(h, A, B, C, state, times)
	if isnothing(C)
		state_right = copy(state)
	else
		state_right = C * state
	end
	if !isnothing(A)
		state_right = A * state_right
	end
	state_right = ExactFiniteMPS(state_right.data)
	# isa(h, QuantumOperator) && (h = FiniteMPO(h))
	left, right = init_h_center(h, state_right)

	result = []
	for i in 1:length(times)
		local cache_right	
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if tspan[2] - tspan[1] > 0.
			state_right = _exact_timeevolution_util(h, tspan[2]-tspan[1], state_right, left, right, ishermitian=false)
		end
		push!(result, expectation(B, FiniteDensityOperatorMPS(FiniteMPS(state_right), state.fusers, state.I)))
	end
	return [result...]
end

# exact diagonalization, used for small systems or debug
function exact_correlation_2op_1t(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::FiniteMPS, times::Vector{<:Real}; reverse::Bool=false)
	_check_times(times)
	is_constant(h) || throw(ArgumentError("cony onstant hamiltonian supported."))
	a = FiniteMPO(a)
	b = FiniteMPO(b)
	h = FiniteMPO(h)
	reverse ? _exact_unitary_tt_corr_util(h, a, b, nothing, state, times) : _exact_unitary_tt_corr_util(h, nothing, a, b, state, times)
end
function exact_correlation_2op_1t(h::SuperOperatorBase, a::QuantumOperator, b::QuantumOperator, state::FiniteDensityOperatorMPS, times::Vector{<:Real}; reverse::Bool=false)
	_check_times(times)
	mpo_a = FiniteMPO(a, alg=SVDCompression())
	mpo_b = FiniteMPO(b, alg=SVDCompression())
	iden = id(mpo_a)
	reverse ? _exact_open_tt_corr_util(FiniteMPO(h), h.fuser(iden, conj(FiniteMPO(mpo_a'))), h.fuser(mpo_b, conj(iden)), nothing, state, times) : _exact_open_tt_corr_util(
		FiniteMPO(h), nothing, h.fuser(mpo_a, conj(iden)), h.fuser(mpo_b, conj(iden)), state, times)
end







