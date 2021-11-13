# two time correlation for pure state and density matrices
# the complexity arises from the fact that the A, B operators may change the sector of the state

_to_n(x::AdjointFiniteMPO) = FiniteMPO(mpo_tensor_adjoint.(raw_data(x.parent)))
_to_a(x::FiniteMPO) = adjoint(FiniteMPO(mpo_tensor_adjoint.(raw_data(x))))

_time_reversal(t::Number) = -conj(t)
_time_reversal(a::Tuple{<:Number, <:Number}) = (_time_reversal(a[1]), _time_reversal(a[2]))

function _unitary_tt_corr_at_b(h, A::AdjointFiniteMPO, B::FiniteMPO, state, times, stepper)
	state_right = B * state
	canonicalize!(state_right)
	state_left = copy(state)

	result = scalar_type(state)[]
	local cache_left, cache_right	
	for i in 1:length(times)	
		# println("state norm $(norm(state_left)), $(norm(state_right)).")
		# tspan = (i == 1) ? (0., -im*times[1]) : (-im*times[i-1], -im*times[i])
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			stepper_right = change_tspan_dt(stepper, tspan=tspan)
			stepper_left = change_tspan_dt(stepper, tspan=_time_reversal(tspan))
			(@isdefined cache_left) || (cache_left = timeevo_cache(h, stepper_left, state_left))
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper_right, state_right))
			state_left, cache_left = timeevo!(state_left, h, stepper_left, cache_left)
			state_right, cache_right = timeevo!(state_right, h, stepper_right, cache_right)
			# println("step size $(cache_left.stepper.stepsize), $(cache_right.stepper.stepsize)")
		end
		push!(result, dot(A' * state_left, state_right) / dim(space_r(state_right)) )
	end
	return result
end
_unitary_tt_corr_at_b(h, A::FiniteMPO, B::AdjointFiniteMPO, state, times, stepper) = _unitary_tt_corr_at_b(h, _to_a(A), _to_n(B), state, times, stepper)


function _unitary_tt_corr_a_bt(h, A::AdjointFiniteMPO, B::FiniteMPO, state, times, stepper)
	state_right = copy(state)
	state_left = A' * state
	canonicalize!(state_left)

	result = scalar_type(state)[]
	local cache_left, cache_right	
	for i in 1:length(times)	
		# tspan = (i == 1) ? (0., -im*times[1]) : (-im*times[i-1], -im*times[i])
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if abs(tspan[2] - tspan[1]) > 0.
			stepper_right = change_tspan_dt(stepper, tspan=tspan)
			stepper_left = change_tspan_dt(stepper, tspan=_time_reversal(tspan))
			(@isdefined cache_left) || (cache_left = timeevo_cache(h, stepper_left, state_left))
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper_right, state_right))
			state_left, cache_left = timeevo!(state_left, h, stepper_left, cache_left)
			state_right, cache_right = timeevo!(state_right, h, stepper_right, cache_right)
		end
		push!(result, expectation(state_left, B, state_right) / dim(space_r(state_left)) )
	end
	return result
end
_unitary_tt_corr_a_bt(h, A::FiniteMPO, B::AdjointFiniteMPO, state, times, stepper) = _unitary_tt_corr_a_bt(h, _to_a(A), _to_n(B), state, times, stepper)



"""
	correlation_2op_1t(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::FiniteMPS, times::Vector{<:Real}, stepper::AbstractStepper; 
	reverse::Bool=false) 
	for a unitary system with hamiltonian h, compute <a(t)b> if revere=false and <a b(t)> if reverse=true
	for an open system with superoperator h, and a, b to be normal operators, compute <a(t)b> if revere=false and <a b(t)> if reverse=true.
	For open system see definitions of <a(t)b> or <a b(t)> on Page 146 of Gardiner and Zoller (Quantum Noise)
"""
function correlation_2op_1t(h::Union{QuantumOperator, AbstractMPO}, a::AbstractMPO, b::AbstractMPO, state::FiniteMPS, times::Vector{<:Real};
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	if scalar_type(state) <: Real
		state = complex(state)
	end
	times = -im .* times
	reverse ? _unitary_tt_corr_a_bt(h, a, b, state, times, stepper) : _unitary_tt_corr_at_b(h, a, b, state, times, stepper)
end

"""
	correlation_2op_1τ(h::QuantumOperator, a::QuantumOperator, b::QuantumOperator, state::FiniteMPS, times::Vector{<:Real}, stepper::AbstractStepper; 
	reverse::Bool=false) 
	for a unitary system with hamiltonian h, compute <a(τ)b> if revere=false and <a b(τ)> if reverse=true
"""
function correlation_2op_1τ(h::Union{QuantumOperator, AbstractMPO}, a::AbstractMPO, b::AbstractMPO, state::FiniteMPS, times::Vector{<:Real};
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	times = -times
	reverse ? _unitary_tt_corr_a_bt(h, a, b, state, times, stepper) : _unitary_tt_corr_at_b(h, a, b, state, times, stepper)
end



# function _open_tt_corr_util(h, A, B, C, state, times, stepper)
# 	if isnothing(C)
# 		state_right = copy(state)
# 	else
# 		state_right = C * state
# 	end
# 	if !isnothing(A)
# 		state_right = A * state_right
# 	end
# 	result = scalar_type(state_right)[]
# 	# cache_right = timeevo_cache(h, stepper, state_right)
# 	local cache_right
# 	for i in 1:length(times)	
# 		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
# 		if tspan[2] - tspan[1] > 0.
# 			stepper = change_tspan_dt(stepper, tspan=tspan)
# 			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper, state_right))
# 			state_right, cache_right = timeevo!(state_right, h, stepper, cache_right)
# 		end
# 		push!(result, dot(B' * state_right.I, state_right.data) )
# 	end
# 	return result
# end 

function _open_tt_corr_at_b(h, A::AdjointFiniteMPO, B::FiniteMPO, state, times, stepper)
	state_right = B * state
	canonicalize!(state_right)

	result = scalar_type(state_right)[]
	local cache_right
	for i in 1:length(times)	
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if tspan[2] - tspan[1] > 0.
			stepper = change_tspan_dt(stepper, tspan=tspan)
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper, state_right))
			state_right, cache_right = timeevo!(state_right, h, stepper, cache_right)
		end
		push!(result, dot(A' * state_right.I, state_right.data) / dim(space_r(state_right)) )
	end
	return result
end 

function _open_tt_corr_a_bt(h, A::FiniteMPO, B::FiniteMPO, state, times, stepper)
	state_right = A * state
	canonicalize!(state_right)
	fuser = h.fuser

	result = scalar_type(state_right)[]
	local cache_right
	local iden
	for i in 1:length(times)	
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if tspan[2] - tspan[1] > 0.
			stepper = change_tspan_dt(stepper, tspan=tspan)
			(@isdefined cache_right) || (cache_right = timeevo_cache(h, stepper, state_right))
			state_right, cache_right = timeevo!(state_right, h, stepper, cache_right)
		end
		# push!(result, dot(B' * state_right.I, state_right.data) )
		
		# (@isdefined iden) || iden =  
		if fuser === ⊠
			tmp = B * state_right.data
			(@isdefined iden) || (iden = _boxtimes_iden_mps(scalar_type(tmp), physical_spaces(tmp), space_l(tmp), space_r(tmp)'))
			push!(result, dot(iden, tmp) / dim(space_r(tmp)) )
		else
			# we need to multiply frobeniusschur factor here since one of the operator is inverted 
			push!(result, dot(_to_a(B)' * state_right.I, state_right.data) * frobeniusschur(sector(state_right.data)) / dim(space_r(state_right)) )
		end
	end
	return result
end 
# _open_tt_corr_a_bt(h, A::AdjointFiniteMPO, B::FiniteMPO, state, times, stepper) = _open_tt_corr_a_bt(h, _to_n(A), _to_a(B), state, times, stepper)

# function correlation_2op_1t(h::SuperOperatorBase, a::QuantumOperator, b::QuantumOperator, state::FiniteDensityOperatorMPS, times::Vector{<:Real}; 
# 	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
# 	mpo_a = FiniteMPO(a, alg=SVDCompression())
# 	mpo_b = FiniteMPO(b, alg=SVDCompression())
# 	iden = id(mpo_a)
# 	reverse ? _open_tt_corr_util(h, h.fuser(iden, conj(FiniteMPO(mpo_a'))), h.fuser(mpo_b, conj(iden)), nothing, state, times, stepper) : _open_tt_corr_util(h
# 		, nothing, h.fuser(mpo_a, conj(iden)), h.fuser(mpo_b, conj(iden)), state, times, stepper)
# end

function correlation_2op_1t(h::SuperOperatorBase, a::AdjointFiniteMPO, b::FiniteMPO, state::FiniteDensityOperatorMPS, times::Vector{<:Real}; 
	stepper::AbstractStepper=TEBDStepper(tspan=(0., 0.01), stepsize=0.01), reverse::Bool=false)
	# (h.fuser === ⊠) || throw(ArgumentError("only fuser ⊠ is supported here."))
	iden = id(b)
	mpo_b = h.fuser(b, conj(iden), right=nothing)
	if reverse
		return _open_tt_corr_a_bt(h, h.fuser(iden, conj(a'), right=nothing), mpo_b, state, times, stepper)
	else
		return _open_tt_corr_at_b(h, adjoint(h.fuser(a', conj(iden), right=nothing)), mpo_b, state, times, stepper)
	end
end
correlation_2op_1t(h::SuperOperatorBase, a::FiniteMPO, b::AdjointFiniteMPO, state::FiniteDensityOperatorMPS, times::Vector{<:Real}; kwargs...) = correlation_2op_1t(
	h, _to_a(a), _to_n(b), state, times; kwargs...)

function _exact_unitary_tt_corr_at_b(h, A::AdjointFiniteMPO, B::FiniteMPO, state, times)
	state_right = B * state
	state_left = copy(state)

	state_left = ExactFiniteMPS(state_left)
	state_right = ExactFiniteMPS(state_right)
	# (state_left.center == state_right.center) || error("something wrong.")
	sr_hleft, sr_hright = init_h_center(h, state_right)
	sl_hleft, sl_hright = init_h_center(h, state_left)

	result = scalar_type(state)[]
	for i in 1:length(times)	
		tspan_right = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		tspan_left = _time_reversal(tspan_right)
		if abs(tspan_right[2] - tspan_right[1]) > 0.
			state_left = _exact_timeevolution_util(h, tspan_left[2]-tspan_left[1], state_left, sl_hleft, sl_hright, ishermitian=true)
			state_right = _exact_timeevolution_util(h, tspan_right[2]-tspan_right[1], state_right, sr_hleft, sr_hright, ishermitian=true)
		end
		push!(result, dot(A' * FiniteMPS(state_left), FiniteMPS(state_right)) / dim(space_r(state_right)) )
	end
	return result
end
_exact_unitary_tt_corr_at_b(h, A::FiniteMPO, B::AdjointFiniteMPO, state, times) = _exact_unitary_tt_corr_at_b(h, _to_a(A), _to_n(B), state, times)


function _exact_unitary_tt_corr_a_bt(h, A::AdjointFiniteMPO, B::FiniteMPO, state, times)
	state_right = copy(state)
	state_left = A' * state

	state_left = ExactFiniteMPS(state_left)
	state_right = ExactFiniteMPS(state_right)
	# (state_left.center == state_right.center) || error("something wrong.")
	sr_hleft, sr_hright = init_h_center(h, state_right)
	sl_hleft, sl_hright = init_h_center(h, state_left)

	result = scalar_type(state)[]
	local cache_left, cache_right	
	for i in 1:length(times)	
		tspan_right = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		tspan_left = _time_reversal(tspan_right)
		if abs(tspan_right[2] - tspan_right[1]) > 0.
			state_left = _exact_timeevolution_util(h, tspan_left[2]-tspan_left[1], state_left, sl_hleft, sl_hright, ishermitian=true)
			state_right = _exact_timeevolution_util(h, tspan_right[2]-tspan_right[1], state_right, sr_hleft, sr_hright, ishermitian=true)
		end
		push!(result, expectation(FiniteMPS(state_left), B, FiniteMPS(state_right)) / dim(space_r(state_left)) )
	end
	return result
end
_exact_unitary_tt_corr_a_bt(h, A::FiniteMPO, B::AdjointFiniteMPO, state, times) = _exact_unitary_tt_corr_a_bt(h, _to_a(A), _to_n(B), state, times)

# exact diagonalization, used for small systems or debug
function exact_correlation_2op_1t(h::AbstractMPO, a::AbstractMPO, b::AbstractMPO, state::FiniteMPS, times::Vector{<:Real}; reverse::Bool=false)
	if scalar_type(state) <: Real
		state = complex(state)
	end
	times = -im .* times
	reverse ? _exact_unitary_tt_corr_a_bt(h, a, b, state, times) : _exact_unitary_tt_corr_at_b(h, a, b, state, times)
end
function exact_correlation_2op_1t(h::QuantumOperator, a::AbstractMPO, b::AbstractMPO, state::FiniteMPS, times::Vector{<:Real}; reverse::Bool=false)
	return exact_correlation_2op_1t(FiniteMPO(h), a, b, state, times, reverse=reverse)
end
function exact_correlation_2op_1τ(h::AbstractMPO, a::AbstractMPO, b::AbstractMPO, state::FiniteMPS, times::Vector{<:Real}; reverse::Bool=false)
	times = -times
	reverse ? _exact_unitary_tt_corr_a_bt(h, a, b, state, times) : _exact_unitary_tt_corr_at_b(h, a, b, state, times)
end
function exact_correlation_2op_1τ(h::QuantumOperator, a::AbstractMPO, b::AbstractMPO, state::FiniteMPS, times::Vector{<:Real}; reverse::Bool=false)
	return exact_correlation_2op_1τ(FiniteMPO(h), a, b, state, times, reverse=reverse)
end

# function _exact_open_tt_corr_util(h, A, B, C, state, times)
# 	if isnothing(C)
# 		state_right = copy(state)
# 	else
# 		state_right = C * state
# 	end
# 	if !isnothing(A)
# 		state_right = A * state_right
# 	end
# 	state_right = ExactFiniteMPS(state_right.data)
# 	# isa(h, QuantumOperator) && (h = FiniteMPO(h))
# 	left, right = init_h_center(h, state_right)

# 	result = scalar_type(state)[]
# 	for i in 1:length(times)
# 		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
# 		if tspan[2] - tspan[1] > 0.
# 			state_right = _exact_timeevolution_util(h, tspan[2]-tspan[1], state_right, left, right, ishermitian=false)
# 		end
# 		# push!(result, expectation(B, FiniteDensityOperatorMPS(FiniteMPS(state_right), state.fusers, state.I)))
# 		push!(result, dot(B' * state.I, FiniteMPS(state_right) ) )
# 	end
# 	return result
# end


function _exact_open_tt_corr_at_b(h, A, B, state, times)
	state_right = B * state
	state_right = ExactFiniteMPS(state_right.data)
	

	left, right = init_h_center(h, state_right)
	result = scalar_type(state)[]
	for i in 1:length(times)
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if tspan[2] - tspan[1] > 0.
			state_right = _exact_timeevolution_util(h, tspan[2]-tspan[1], state_right, left, right, ishermitian=false)
		end
		# push!(result, expectation(B, FiniteDensityOperatorMPS(FiniteMPS(state_right), state.fusers, state.I)))
		push!(result, dot(A' * state.I, FiniteMPS(state_right)) / dim(space_r(state_right)) )
	end
	return result
end

function _exact_open_tt_corr_a_bt(h, A::FiniteMPO, B::FiniteMPO, state, times, fuser)
	state_right = A * state
	state_right = ExactFiniteMPS(state_right.data)
	left, right = init_h_center(h, state_right)
	# fuser = h.fuser

	result = scalar_type(state)[]
	local iden
	for i in 1:length(times)
		tspan = (i == 1) ? (0., times[1]) : (times[i-1], times[i])
		if tspan[2] - tspan[1] > 0.
			state_right = _exact_timeevolution_util(h, tspan[2]-tspan[1], state_right, left, right, ishermitian=false)
		end
		# push!(result, expectation(B, FiniteDensityOperatorMPS(FiniteMPS(state_right), state.fusers, state.I)))
		# push!(result, dot(B' * state.I, FiniteMPS(state_right) ) )

		if fuser === ⊠
			tmp = B * FiniteMPS(state_right)
			(@isdefined iden) || (iden = _boxtimes_iden_mps(scalar_type(tmp), physical_spaces(tmp), space_l(tmp), space_r(tmp)'))
			push!(result, dot(iden, tmp) / dim(space_r(tmp)) )
		else
			push!(result, dot(_to_a(B)' * state.I, FiniteMPS(state_right)) * frobeniusschur(sector(state_right)) / dim(space_r(state_right)) )
		end
	end
	return result
end

function exact_correlation_2op_1t(h::SuperOperatorBase, a::AdjointFiniteMPO, b::FiniteMPO, state::FiniteDensityOperatorMPS, times::Vector{<:Real}; reverse::Bool=false)
	iden = id(b)
	mpo_b = h.fuser(b, conj(iden), right=nothing)
	if reverse
		return _exact_open_tt_corr_a_bt(FiniteMPO(h), h.fuser(iden, conj(a'), right=nothing), mpo_b, state, times, h.fuser)
	else
		return _exact_open_tt_corr_at_b(FiniteMPO(h), adjoint(h.fuser(a', conj(iden), right=nothing)), mpo_b, state, times)
	end
end
exact_correlation_2op_1t(h::SuperOperatorBase, a::FiniteMPO, b::AdjointFiniteMPO, state::FiniteDensityOperatorMPS, times::Vector{<:Real}; kwargs...) = exact_correlation_2op_1t(
	h, _to_a(a), _to_n(b), state, times; kwargs...)

# function exact_correlation_2op_1t(h::SuperOperatorBase, a::QuantumOperator, b::QuantumOperator, state::FiniteDensityOperatorMPS, times::Vector{<:Real}; reverse::Bool=false)
# 	# _check_times(times)
# 	mpo_a = FiniteMPO(a, alg=SVDCompression())
# 	mpo_b = FiniteMPO(b, alg=SVDCompression())
# 	iden = id(mpo_a)
# 	reverse ? _exact_open_tt_corr_util(FiniteMPO(h), h.fuser(iden, conj(FiniteMPO(mpo_a'))), h.fuser(mpo_b, conj(iden)), nothing, state, times) : _exact_open_tt_corr_util(
# 		FiniteMPO(h), nothing, h.fuser(mpo_a, conj(iden)), h.fuser(mpo_b, conj(iden)), state, times)
# end







