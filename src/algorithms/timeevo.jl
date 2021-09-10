

function compute_step_size(t::Number, dt::Number)
	n = round(Int, abs(t / dt)) 
	if n == 0
		n = 1
	end
	return n, t / n
end


abstract type AbstractStepper end

struct TEBDStepper{T<:Number, C <: TensorKit.TruncationScheme} <: AbstractStepper
	tspan::Tuple{T, T}
	stepsize::T
	n::Int
	order::Int
	trunc::C
end

function TEBDStepper(; tspan::Tuple{<:Number, <:Number}, stepsize::Number, order::Int=2, trunc::TensorKit.TruncationScheme=default_truncation())
	ti, tf = tspan
	δ = tf - ti
	n, stepsize = compute_step_size(δ, stepsize)
	T = promote_type(typeof(ti), typeof(tf), typeof(stepsize))
	return TEBDStepper((convert(T, ti), convert(T, tf)), convert(T, stepsize), n, order, trunc)
end

function Base.getproperty(x::TEBDStepper, s::Symbol)
	if s == :δ
		return x.tspan[2] - x.tspan[1]
	else
		getfield(x, s)
	end
end


struct TDVPStepper{T<:Number, A<:TDVPAlgorithm} <: AbstractStepper
	tspan::Tuple{T, T}
	n::Int
	alg::A	
end

function Base.getproperty(x::TDVPStepper, s::Symbol)
	if s == :δ
		return x.tspan[2] - x.tspan[1]
	elseif s == :order
		return 2
	elseif s == :stepsize
		return x.alg.stepsize
	else
		getfield(x, s)
	end
end

function _change_stepsize(x::TDVP1, stepsize::Number)
	if !(x.stepsize ≈ stepsize)
		x = TDVP1(stepsize=stepsize, ishermitian=x.ishermitian, verbosity=x.verbosity)
	end
	return x
end
function _change_stepsize(x::TDVP2, stepsize::Number)
	if !(x.stepsize ≈ stepsize)
		x = TDVP2(stepsize=x.stepsize, ishermitian=x.ishermitian, trunc=x.trunc, verbosity=x.verbosity)
	end
	return x
end
function _change_stepsize(x::TDVP1S, stepsize::Number)
	if !(x.stepsize ≈ stepsize)
		x = TDVP1S(stepsize=x.stepsize, ishermitian=x.ishermitian, trunc=x.trunc, expan=x.expan, verbosity=x.verbosity)
	end
	return x
end

function TDVPStepper(;tspan::Tuple{<:Number, <:Number}, alg::TDVPAlgorithm)
	ti, tf = tspan
	δ = tf - ti	
	n, stepsize = compute_step_size(δ, alg.stepsize)
	T = promote_type(typeof(ti), typeof(tf), typeof(stepsize))
	return TDVPStepper((convert(T, ti), convert(T, tf)), n, _change_stepsize(alg, stepsize))
end

mutable struct HomogeousTEBDCache{H<:QuantumOperator, C<:QuantumCircuit, S<:TEBDStepper} <: AbstractCache
	h::H
	circuit::C
	stepper::S
end

function HomogeousTEBDCache(h::QuantumOperator, stepper::TEBDStepper)
	is_constant(h) || throw(ArgumentError("const hamiltonian expected."))
	circuit = fuse_gates(repeat(trotter_propagator(h, (0., stepper.stepsize), order=stepper.order, stepsize=stepper.stepsize), stepper.n))
	return HomogeousTEBDCache(h, circuit, stepper)
end

function recalculate!(x::HomogeousTEBDCache, h::QuantumOperator, stepper::TEBDStepper)
	if !((x.h === h) && (stepper.δ==x.stepper.δ) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) )
		is_constant(h) || throw(ArgumentError("const hamiltonian expected."))
		return HomogeousTEBDCache(fuse_gates(repeat(trotter_propagator(h, (0., stepper.stepsize), order=stepper.order, stepsize=stepper.stepsize), stepper.n)), h, stepper)
	else
		return x
	end
end

# function recalculate!(x::HomogeousTEBDCache, stepper::TEBDStepper)
# 	if !((stepper.δ==x.stepper.δ) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) )
# 		x.circuit = fuse_gates(repeat(trotter_propagator(x.h, (0., stepper.stepsize), order=stepper.order, stepsize=stepper.stepsize), stepper.n))
# 		x.stepper = stepper
# 	end
# end

function make_step!(h::QuantumOperator, stepper::TEBDStepper, state::FiniteMPS, x::HomogeousTEBDCache)
	recalculate!(x, h, stepper)
	apply!(x.circuit, state, trunc=x.stepper.trunc)
	return state
end


mutable struct InhomogenousTEBDCache{H<:QuantumOperator, C<:QuantumCircuit, S<:TEBDStepper} <: AbstractCache
	h::H
	circuit::C
	stepper::S	
end

function InhomogenousTEBDCache(h::QuantumOperator, stepper::TEBDStepper)
	circuit = fuse_gates(trotter_propagator(h, stepper.tspan, order=stepper.order, stepsize=stepper.stepsize))
	return InhomogenousTEBDCache(h, circuit, stepper)
end


function recalculate!(x::InhomogenousTEBDCache, h::QuantumOperator, stepper::TEBDStepper)
	if !((x.h === h) && (stepper.tspan==x.stepper.tspan) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) )
		return InhomogenousTEBDCache(fuse_gates(trotter_propagator(h, stepper.tspan, order=stepper.order, stepsize=stepper.stepsize)), h, stepper)
	else
		return x
	end
end

# function recalculate!(x::InhomogenousTEBDCache, stepper::TEBDStepper)
# 	if !((stepper.tspan==x.stepper.tspan) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) )
# 		x.circuit = fuse_gates(trotter_propagator(x.h, stepper.tspan, order=stepper.order, stepsize=stepper.stepsize))
# 		x.stepper = stepper
# 	end
# end

function make_step!(h::QuantumOperator, stepper::TEBDStepper, state::FiniteMPS, x::InhomogenousTEBDCache)
	x = recalculate!(x, h, stepper)
	apply!(x.circuit, state, trunc=x.stepper.trunc)
	return state, x
end

TEBDCache(h::QuantumOperator, stepper::TEBDStepper) = is_constant(h) ? HomogeousTEBDCache(h, stepper) : InhomogenousTEBDCache(h, stepper)


mutable struct HomogeousTDVPCache{E<:ExpectationCache, S<:TDVPStepper}
	env::E
	stepper::S
end

HomogeousTDVPCache(h::Union{FiniteMPO, MPOHamiltonian}, stepper::TDVPStepper, state::FiniteMPS) = HomogeousTDVPCache(environments(h, state), stepper)
TDVPCache(h::Union{FiniteMPO, MPOHamiltonian}, stepper::TDVPStepper, state::FiniteMPS) = HomogeousTDVPCache(h, stepper, state)
# TDVPCache(h::QuantumOperator, stepper::TDVPStepper, state::FiniteMPS) = TDVPCache(FiniteMPO(h, alg=SVDCompression()), stepper, state)

function recalculate!(x::HomogeousTDVPCache, h::Union{FiniteMPO, MPOHamiltonian}, stepper::TDVPStepper, state::FiniteMPS)
	if !((x.env.state === state) && (x.env.h === h))
		return HomogeousTDVPCache(environments(h, state), stepper)
	else
		return HomogeousTDVPCache(x.env, stepper)
	end
end

function make_step!(h::Union{FiniteMPO, MPOHamiltonian}, stepper::TDVPStepper, state::FiniteMPS, x::HomogeousTDVPCache)
	x = recalculate!(x, h, stepper, state)
	for i in 1:stepper.n
		sweep!(x.env, stepper.alg)
	end
	return state, x
end
# function make_step!(h::Union{FiniteMPO, MPOHamiltonian}, stepper::TDVPStepper, state::FiniteDensityOperatorMPS, x::HomogeousTDVPCache)
# 	make_step!(h, stepper, state.data, x)
# 	return state
# end 

timeevo!(state::FiniteMPS, h::QuantumOperator, stepper::TEBDStepper, cache=TEBDCache(h, stepper)) = make_step!(h, stepper, state, cache)
function timeevo!(state::FiniteDensityOperatorMPS, h::SuperOperatorBase, stepper::TEBDStepper, cache=TEBDCache(h.data, stepper))
	make_step!(h.data, stepper, state.data, cache)
	return state, cache
end 
timeevo!(state::FiniteMPS, h::Union{FiniteMPO, MPOHamiltonian}, stepper::TDVPStepper, cache=TDVPCache(h, stepper, state)) = make_step!(h, stepper, state, cache)
function timeevo!(state::FiniteDensityOperatorMPS, h::Union{FiniteMPO, MPOHamiltonian}, stepper::TDVPStepper, cache=TDVPCache(h, stepper, state.data))
	make_step!(h, stepper, state.data, cache)
	return state, cache
end



