# push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")

using Test
using TensorKit
using FiniteMPTools

# function hubbard_chain(L, J, U, p)
# 	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
# 	adagJW = adag * JW
# 	a = adag'

# 	terms = []
# 	for i in 1:L
# 		push!(terms, QTerm(i => nn, coeff=U))
# 	end
# 	for i in 1:L-1
# 		m = QTerm(i=>adagJW, i+1=>a, coeff=-J)
# 		push!(terms, m)
# 		push!(terms, m')
# 	end

# 	observers = [QTerm(i=>nn) for i in 1:L]
# 	return QuantumOperator([terms...]), observers
# end

function long_range_hubbard_chain_mpo(L, J, U, alpha, p)
	# a, adag, nn, JW, JWa, adagJW = fermionic_site_ops()
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
	adagJW = adag * JW
	a = adag'
	terms = []
	for i in 1:L
		push!(terms, QTerm(i => nn, coeff=U))
	end
	for i in 1:L-1
		m = QTerm(i=>adagJW, i+1=>a, coeff=-J)
		push!(terms, m)
		push!(terms, m')
	end

	for i in 1:L
	    for j in i+1:L
	    	coeff = exp(-alpha*(j-i))
	    	push!(terms,  QTerm(i=>nn, j=>nn, coeff=coeff) )
	    end
	end

	observers = [QTerm(i=>nn) for i in 1:L]
	return FiniteMPO(QuantumOperator([terms...])), observers
end

function long_range_hubbard_chain_mpo_ham(L, J, U, alpha, p)
	# a, adag, nn, JW, JWa, adagJW = fermionic_site_ops()
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
	adagJW = adag * JW

	nn = nn.op
	iden = one(nn)
	adagJW = adagJW.op
	a = FiniteMPTools.raw_data(adag')

	adagJW_d = FiniteMPTools.mpo_tensor_adjoint(adagJW)
	a_d = FiniteMPTools.mpo_tensor_adjoint(a)
	# println(typeof(a_d))

	coeff = exp(-alpha)

	mpot = SchurMPOTensor([one(nn) -J * adagJW -J * adagJW_d coeff * nn U * nn; 0. 0. 0. 0. a; 0. 0. 0. 0. a_d; 0. 0. 0. coeff*iden nn; 0. 0. 0. 0. iden])

	observers = [QTerm(i=>nn) for i in 1:L]
	return MPOHamiltonian([mpot]), observers
end

function compare_mpo_ham(L)
	J = 1.0
	U = 1.2
	alpha = 0.5

	Errs = Float64[]

	mpo1, observers =long_range_hubbard_chain_mpo(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	ham2, observers = long_range_hubbard_chain_mpo_ham(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())

	push!(Errs, distance(mpo1, FiniteMPO(ham2, L)))

	mpo1, observers =long_range_hubbard_chain_mpo(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_u1())
	ham2, observers = long_range_hubbard_chain_mpo_ham(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_u1())

	push!(Errs, distance(mpo1, FiniteMPO(ham2, L)))

	mpo1, observers =long_range_hubbard_chain_mpo(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_dense())
	ham2, observers = long_range_hubbard_chain_mpo_ham(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_dense())

	push!(Errs, distance(mpo1, FiniteMPO(ham2, L)))

	return maximum(Errs)
end


function power_law_hubbard_chain_mpo(L, J, U, alpha, p)
	# a, adag, nn, JW, JWa, adagJW = fermionic_site_ops()
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
	adagJW = adag * JW
	a = adag'
	terms = []
	for i in 1:L
		push!(terms, QTerm(i => nn, coeff=U))
	end


	for i in 1:L
	    for j in i+1:L
	    	coeff = exp(-(j-i)) 
	    	push!(terms,  QTerm(i=>nn, j=>nn, coeff=coeff) )
	    	t = QTerm(i=>adagJW, j=>a, coeff=-J*(j-i)^(-alpha) )
	    	push!(terms, t)
	    	push!(terms, t')
	    end
	end

	observers = [QTerm(i=>nn) for i in 1:L]
	return QuantumOperator([terms...]), observers
end

function power_law_hubbard_chain_mpo_ham(L, J, U, alpha, p)
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
	adagJW = adag * JW

	m = GenericDecayTerm(adagJW, adag', f=x->x^(-alpha), coeff=-J)
	m2 = ExponentialDecayTerm(nn, nn, α = exp(-1), coeff=1.)

	mm = exponential_expansion(m, len=L, atol=1.0e-8)
	mm = vcat(mm, adjoint.(mm))
	mm = vcat(mm, [m2])

	# mm = [m2]
	mpot = SchurMPOTensor(U * nn, mm)

	observers = [QTerm(i=>nn) for i in 1:L]
	return MPOHamiltonian([mpot]), observers
end

function exp_law_hubbard_chain_mpo_f(L, J, alpha, p)
	# a, adag, nn, JW, JWa, adagJW = fermionic_site_ops()
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
	adagJW = adag * JW
	a = adag'
	terms = []

	for i in 1:L
	    for j in i+1:L
	    	coeff = alpha^(j-i) 
	    	pos = collect(i:j)
	    	op_v = vcat(vcat([adagJW], [JW for k in (i+1):(j-1)]), [a])
	    	t = QTerm(pos, op_v, coeff=-J * coeff)
	    	push!(terms, t)
	    	push!(terms, t')
	    end
	end
	observers = [QTerm(i=>nn) for i in 1:L]
	return QuantumOperator([terms...]), observers
end

function exp_law_hubbard_chain_mpo_ham_f(L, J, alpha, p)
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
	adagJW = adag * JW

	m = ExponentialDecayTerm(adagJW, JW, adag', α=alpha, coeff=-J)

	# mm = [m2]
	mpot = SchurMPOTensor([m, m'])

	observers = [QTerm(i=>nn) for i in 1:L]
	return MPOHamiltonian([mpot]), observers
end

function power_law_hubbard_chain_mpo_f(L, J, alpha, p)
	# a, adag, nn, JW, JWa, adagJW = fermionic_site_ops()
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
	adagJW = adag * JW
	a = adag'
	terms = []

	for i in 1:L
	    for j in i+1:L
	    	coeff = (j-i)^alpha
	    	pos = collect(i:j)
	    	op_v = vcat(vcat([adagJW], [JW for k in (i+1):(j-1)]), [a])
	    	t = QTerm(pos, op_v, coeff=-J * coeff)
	    	push!(terms, t)
	    	push!(terms, t')
	    end
	end
	observers = [QTerm(i=>nn) for i in 1:L]
	return QuantumOperator([terms...]), observers
end

function power_law_hubbard_chain_mpo_ham_f(L, J, alpha, p)
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]
	adagJW = adag * JW

	m = PowerlawDecayTerm(adagJW, JW, adag', α=alpha, coeff=-J)

	mm = exponential_expansion(m, len=L, atol=1.0e-8)
	mm = vcat(mm, adjoint.(mm))

	mpot = SchurMPOTensor(mm)

	observers = [QTerm(i=>nn) for i in 1:L]
	return MPOHamiltonian([mpot]), observers
end

function test_power_law_ham(L)
	J = 1.1
	U = 1.2
	alpha = 4.0


	h1, observers =power_law_hubbard_chain_mpo(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())

	mpo1 = FiniteMPO(h1, alg=Deparallelise(tol=1.0e-11))
	mpo2 = FiniteMPO(h1, alg=SVDCompression(tol=1.0e-11))


	h2, observers = power_law_hubbard_chain_mpo_ham(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())

	mpo3 = FiniteMPO(h2, L)

	return max(distance(mpo1, mpo2), distance(mpo1, mpo3), distance(mpo2, mpo3))
end

function test_exp_law_ham_f(L)
	J = 1.1
	alpha = exp(-1)	

	h1, observers = exp_law_hubbard_chain_mpo_f(L, J, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())

	mpo1 = FiniteMPO(h1, alg=Deparallelise(tol=1.0e-11))
	# println(bond_dimensions(mpo1))

	h2, observers = exp_law_hubbard_chain_mpo_ham_f(L, J, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())

	mpo2 = FiniteMPO(h2, L)
	# println(bond_dimensions(mpo2))

	# println(distance(mpo1, mpo2))
	return distance(mpo1, mpo2)
end

function test_power_law_ham_f(L)
	J = 1.1
	alpha = -4.

	h1, observers = power_law_hubbard_chain_mpo_f(L, J, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())

	mpo1 = FiniteMPO(h1, alg=Deparallelise(tol=1.0e-11))
	# println(bond_dimensions(mpo1))

	h2, observers = power_law_hubbard_chain_mpo_ham_f(L, J, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())

	mpo2 = FiniteMPO(h2, L)
	# println(bond_dimensions(mpo2))

	# println(distance(mpo1, mpo2))
	return distance(mpo1, mpo2)
end

println("-----------test mpo hamiltonain-----------------")
@testset "long range mpo hamiltonian" begin
	for L in [5, 6]
		@test compare_mpo_ham(L) < 1.0e-5
	end
	for L in [10, 11]
		@test test_power_law_ham(L) < 1.0e-3
		@test test_exp_law_ham_f(L) < 1.0e-3
		@test test_power_law_ham_f(L) < 1.0e-3
	end
end


function hubbard_ladder(L, J1, J2, U, p)
	adag, nn, JW = p["+"], p["n↑n↓"], p["JW"]

	adagJW = adag * JW
	a = adag'

	terms = []
	for i in 1:L
		push!(terms, QTerm(i => nn, coeff=U))
	end
	for i in 1:L-1
		m = QTerm(i=>adagJW, i+1=>a, coeff=-J1)
		push!(terms, m)
		push!(terms, m')
	end
	for i in 1:L-2
		m = QTerm(i=>adagJW, i+1=>JW, i+2=>a, coeff=-J2)
		push!(terms, m)
		push!(terms, m')
	end

	observers = [QTerm(i=>nn) for i in 1:L]
	return QuantumOperator([terms...]), observers
end

function initial_state_u1_su2(L)
	physpace = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)

	init_state = [(-0.5, 0) for i in 1:L]
	for i in 2:2:L
		init_state[i] = (0.5, 0)
	end
	n = sum([item[1] for item in init_state])

	right = Rep[U₁×SU₂]((n, 0)=>1)
	state = prodmps(ComplexF64, physpace, init_state, right=right )

	return state, first(sectors(right))
end


function initial_state_u1_u1(L)
	physpace = Rep[U₁×U₁]((0, 0)=>1, (0, 1)=>1, (1, 0)=>1, (1, 1)=>1)

	init_state = [(0, 0) for i in 1:L]
	for i in 2:2:L
		init_state[i] = (1, 1)
	end
	n1 = sum([item[1] for item in init_state])
	n2 = sum([item[2] for item in init_state])

	right = Rep[U₁×U₁]((n1, n2)=>1)
	state = prodmps(ComplexF64, physpace, init_state, right=right )
	return state, first(sectors(right))
end


function initial_state_dense(L)
	init_state = [0 for i in 1:L]
	for i in 2:2:L
		init_state[i] = 3
	end
	return prodmps(ComplexF64, [4 for i in 1:L], init_state)
end 

function do_dmrg(dmrg, alg)
	dmrg_sweeps = 5
	# Evals, delta = compute!(dmrg, alg)
	Evals = Float64[]
	for i in 1:dmrg_sweeps
		Evals, delta = sweep!(dmrg, alg)
	end
	return Evals[end]
end

function test_ground_state(L)
	J = 1.
	J2 = 1.2
	U = 1.37

	all_Es = Float64[]
	# hubbard chain u1 u1 dmrg
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_u1())
	mpo = FiniteMPO(ham)
	state, sector = initial_state_u1_u1(L)
	push!(all_Es, do_dmrg(environments(mpo, copy(state)), DMRG2()) )
	push!(all_Es, do_dmrg(environments(mpo, copy(state)), DMRG1S()) )

	E, _st = exact_diagonalization(mpo, sector=sector, num=1, ishermitian=true)
	push!(all_Es, E[1])

	# hubbard chain u1 su2 dmrg
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	mpo = FiniteMPO(ham)
	state, sector = initial_state_u1_su2(L)
	push!(all_Es, do_dmrg(environments(mpo, copy(state)), DMRG2()) )
	push!(all_Es, do_dmrg(environments(mpo, copy(state)), DMRG1S()) )

	E, _st = exact_diagonalization(mpo, sector=sector, num=1, ishermitian=true)
	push!(all_Es, E[1])
	# println("energies $all_Es")

	return maximum(abs.(all_Es .- all_Es[1])) 
end

function test_ground_state_2(L)
	J = 1.
	J2 = 1.2
	U = 1.37
	alpha = 0.45

	all_Es = Float64[]
	# hubbard chain u1 u1 dmrg

	mpo1, observers =long_range_hubbard_chain_mpo(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_u1())
	mpo2, observers = long_range_hubbard_chain_mpo_ham(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_u1())

	state, sector = initial_state_u1_u1(L)
	push!(all_Es, do_dmrg(environments(mpo1, copy(state)), DMRG2()) )
	push!(all_Es, do_dmrg(environments(mpo2, copy(state)), DMRG2()) )
	push!(all_Es, do_dmrg(environments(mpo2, copy(state)), DMRG1S()) )

	E, _st = exact_diagonalization(mpo1, sector=sector, num=1, ishermitian=true)
	push!(all_Es, E[1])
	E, _st = exact_diagonalization(mpo2, sector=sector, num=1, len=L, ishermitian=true)
	push!(all_Es, E[1])


	# hubbard chain u1 su2 dmrg
	mpo1, observers =long_range_hubbard_chain_mpo(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	mpo2, observers = long_range_hubbard_chain_mpo_ham(L, J, U, alpha, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	state, sector = initial_state_u1_su2(L)

	push!(all_Es, do_dmrg(environments(mpo1, copy(state)), DMRG2()) )
	push!(all_Es, do_dmrg(environments(mpo2, copy(state)), DMRG2()) )
	push!(all_Es, do_dmrg(environments(mpo2, copy(state)), DMRG1S()) )

	E, _st = exact_diagonalization(mpo1, sector=sector, num=1, ishermitian=true)
	push!(all_Es, E[1])
	E, _st = exact_diagonalization(mpo2, sector=sector, num=1, len=L, ishermitian=true)
	push!(all_Es, E[1])


	return maximum(abs.(all_Es .- all_Es[1])) 
end

function test_excitations(L)
	J = 1.
	J2 = 1.1
	U = 1.4


	# hubbard chain u1 u1 dmrg
	U1_Es = Float64[]
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_u1())
	mpo = FiniteMPO(ham)

	state, sector = initial_state_u1_u1(L)
	dmrg = environments(mpo, copy(state))
	do_dmrg(dmrg, DMRG2())
	gs_state = dmrg.state

	push!(U1_Es, do_dmrg(environments(mpo, copy(state), [gs_state]), DMRG2()) )
	push!(U1_Es, do_dmrg(environments(mpo, copy(state), [gs_state]), DMRG1S()) )

	E, _st = exact_diagonalization(mpo, sector=sector, num=2, ishermitian=true)
	push!(U1_Es, E[2])
	# println(U1_Es)

	# hubbard chain u1 su2 dmrg
	SU2_Es = Float64[]
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	mpo = FiniteMPO(ham)

	state, sector = initial_state_u1_su2(L)
	dmrg = environments(mpo, copy(state))
	do_dmrg(dmrg, DMRG2())
	gs_state = dmrg.state

	push!(SU2_Es, do_dmrg(environments(mpo, copy(state), [gs_state]), DMRG2()) )
	push!(SU2_Es, do_dmrg(environments(mpo, copy(state), [gs_state]), DMRG1S()) )

	E, _st = exact_diagonalization(mpo, sector=sector, num=2, ishermitian=true)
	push!(SU2_Es, E[2])

	# println(SU2_Es)
	return max(maximum(abs.(U1_Es .- U1_Es[1])), maximum(abs.(SU2_Es .- SU2_Es[1])) ) 
end

function do_tdvp(dmrg, alg, n, obs)
	for i in 1:n
		sweep!(dmrg, alg)
	end
	return real([expectation(item, dmrg.state, iscanonical=false) for item in obs])
end

function do_exact_evo(mpo, state, t, obs)
	state = exact_timeevolution(mpo, t, ExactFiniteMPS(state), ishermitian=true)
	return real([expectation(item, FiniteMPS(state), iscanonical=false) for item in obs])
end


function test_tdvp(L)
	J = 1.
	J2 = 1.2
	U = 0.7

	dt = 0.01
	dmrg_sweeps = 50

	# hubbard chain u1 u1 tdvp
	all_obs = Vector{Float64}[]
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_u1())
	mpo = FiniteMPO(ham)

	state, sector = initial_state_u1_u1(L)

	push!(all_obs, do_exact_evo(mpo, state, -im*dt*dmrg_sweeps, observers) )

	push!(all_obs, do_tdvp(environments(mpo, copy(state)), TDVP2(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers) )
	push!(all_obs, do_tdvp(environments(mpo, copy(state)), TDVP1S(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers) )


	# hubbard chain u1 su2 tdvp
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	mpo = FiniteMPO(ham)

	state, sector = initial_state_u1_su2(L)
	push!(all_obs, do_exact_evo(mpo, state, -im*dt*dmrg_sweeps, observers) )

	push!(all_obs, do_tdvp(environments(mpo, copy(state)), TDVP2(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers) )
	push!(all_obs, do_tdvp(environments(mpo, copy(state)), TDVP1S(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers) )

	# hubbard chain dense tdvp
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_dense())
	mpo = FiniteMPO(ham)

	state = initial_state_dense(L)
	push!(all_obs, do_exact_evo(mpo, state, -im*dt*dmrg_sweeps, observers) )

	push!(all_obs, do_tdvp(environments(mpo, copy(state)), TDVP2(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers) )
	push!(all_obs, do_tdvp(environments(mpo, copy(state)), TDVP1S(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers) )


	return maximum([maximum(abs.(all_obs[i] - all_obs[1])) for i in 1:length(all_obs)])

end

function do_tebd(circ, state, obs)
	apply!(circ, state, trunc=MPSTruncation(100, 1.0e-8))
	return real([expectation(item, state, iscanonical=true) for item in obs])
end

function test_tebd(L)
	J = 1.
	J2 = 1.2
	U = 0.7

	dt = 0.01
	dmrg_sweeps = 50

	# hubbard chain u1 u1 tebd
	all_obs = Vector{Float64}[]
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_u1())
	mpo = FiniteMPO(ham)

	state, sector = initial_state_u1_u1(L)

	push!(all_obs, do_exact_evo(mpo, state, -im*dt*dmrg_sweeps, observers) )
	
	circuit = fuse_gates(trotter_propagator(ham, (0., -im * dmrg_sweeps*dt), stepsize=dt, order=4))
	push!(all_obs, do_tebd(circuit, copy(state), observers))

	# hubbard chain u1 su2 tebd
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	state, sector = initial_state_u1_su2(L)
	circuit = fuse_gates(trotter_propagator(ham, (0., -im * dmrg_sweeps*dt), stepsize=dt, order=4))
	push!(all_obs, do_tebd(circuit, copy(state), observers))

	# hubbard chain dense tebd
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_dense())
	state = initial_state_dense(L)
	circuit = fuse_gates(trotter_propagator(ham, (0., -im * dmrg_sweeps*dt), stepsize=dt, order=4))
	push!(all_obs, do_tebd(circuit, copy(state), observers))


	# println(all_obs)
	return maximum([maximum(abs.(all_obs[i] - all_obs[1])) for i in 1:length(all_obs)])
end

function _prepare_rand_state(state; D)
	rand_state = randommps(ComplexF64, physical_spaces(state), sector=sector(state), D=D)
	canonicalize!(rand_state, normalize=true)
	return rand_state
end

function test_expectation_values_1(L)
	J = 1.
	J2 = 1.2
	U = 0.8

	ham_vals = Float64[]
	# u1 u1
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_u1())
	state, sect = initial_state_u1_u1(L)
	canonicalize!(state, normalize=true)
	mpo = FiniteMPO(ham)

	push!(ham_vals, real(expectation(ham, state, iscanonical=true)))
	push!(ham_vals, real(expectation(ham, state, iscanonical=false)))
	push!(ham_vals, real(expectation(mpo, state)) )

	rho = DensityOperator(state, fuser=⊗)
	push!(ham_vals, real(tr( mpo * FiniteMPO(rho))) )
	push!(ham_vals, real(expectation(mpo ⊗ conj(id(mpo)) , rho) ))

	rho = DensityOperator(state, fuser=⊠)
	push!(ham_vals, real(expectation(mpo ⊠ conj(id(mpo)) , rho) ))

	# u1 su2
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	state, sect = initial_state_u1_su2(L)
	canonicalize!(state, normalize=true)
	mpo = FiniteMPO(ham)

	push!(ham_vals, real(expectation(ham, state, iscanonical=true)))
	push!(ham_vals, real(expectation(ham, state, iscanonical=false)))
	push!(ham_vals, real(expectation(mpo, state)) )

	rho = DensityOperator(state, fuser=⊗)
	push!(ham_vals, real(tr( mpo * FiniteMPO(rho))) )
	push!(ham_vals, real(expectation(mpo ⊗ conj(id(mpo)) , rho) ))

	rho = DensityOperator(state, fuser=⊠)
	push!(ham_vals, real(expectation(mpo ⊠ conj(id(mpo)), rho) ))


	# dense
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_dense())
	state = initial_state_dense(L)
	canonicalize!(state, normalize=true)
	mpo = FiniteMPO(ham)

	push!(ham_vals, real(expectation(ham, state, iscanonical=true)))
	push!(ham_vals, real(expectation(ham, state, iscanonical=false)))
	push!(ham_vals, real(expectation(mpo, state)) )

	rho = DensityOperator(state, fuser=⊗)
	push!(ham_vals, real(tr( mpo * FiniteMPO(rho))) )
	push!(ham_vals, real(expectation(mpo ⊗ conj(id(mpo)) , rho) ))

	rho = DensityOperator(state, fuser=⊠)
	push!(ham_vals, real(expectation(mpo ⊠ conj(id(mpo)) , rho) ))

	return maximum(abs.(ham_vals .- ham_vals[1]))
end

function test_expectation_values_2(L)
	J = 1.
	J2 = 1.2
	U = 0.8

	ham_vals = Float64[]
	Errs = Float64[]
	# u1 u1
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_u1())
	state, sect = initial_state_u1_u1(L)
	rand_state = _prepare_rand_state(state, D=10)
	mpo = FiniteMPO(ham)

	push!(ham_vals, real(expectation(ham, rand_state, iscanonical=true)) )
	push!(ham_vals, real(expectation(ham, rand_state, iscanonical=false)) )

	for fuser in [⊗, ⊠]
		rho = DensityOperator(rand_state, fuser=fuser)
		push!(ham_vals, real(expectation(fuser(mpo, conj(id(mpo)) ), rho)) )
	end
	push!(Errs, maximum(abs.(ham_vals .- ham_vals[1])) )

	# u1 su2
	ham_vals = Float64[]
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_u1_su2())
	state, sect = initial_state_u1_su2(L)
	rand_state = _prepare_rand_state(state, D=10)
	mpo = FiniteMPO(ham)

	push!(ham_vals, real(expectation(ham, rand_state, iscanonical=true)) )
	push!(ham_vals, real(expectation(ham, rand_state, iscanonical=false)) )

	for fuser in [⊗, ⊠]
		rho = DensityOperator(rand_state, fuser=fuser)
		push!(ham_vals, real(expectation(fuser(mpo, conj(id(mpo))), rho)) )
	end
	push!(Errs, maximum(abs.(ham_vals .- ham_vals[1])) )


	# dense
	ham_vals = Float64[]
	ham, observers = hubbard_ladder(L, J, J2, U, FiniteMPTools.spinal_fermion_site_ops_dense())
	state = initial_state_dense(L)
	rand_state = _prepare_rand_state(state, D=10)
	mpo = FiniteMPO(ham)

	push!(ham_vals, real(expectation(ham, rand_state, iscanonical=true)) )
	push!(ham_vals, real(expectation(ham, rand_state, iscanonical=false)) )

	for fuser in [⊗, ⊠]
		rho = DensityOperator(rand_state, fuser=fuser)
		push!(ham_vals, real(expectation(fuser(mpo, conj(id(mpo)) ), rho)) )
	end
	push!(Errs, maximum(abs.(ham_vals .- ham_vals[1])) )	
	
	return maximum(Errs)
end

println("----------------test algorithms--------------------")

@testset "expectation values" begin
	for L in [5, 6]
		@test test_expectation_values_1(L) < 1.0e-8
		@test test_expectation_values_2(L) < 1.0e-8
	end
end


@testset "ground state dmrg algorithms" begin
	for L in [5, 6]
		@test test_ground_state(L) < 1.0e-8
		@test test_ground_state_2(L) < 1.0e-8
	end
end

@testset "excitations dmrg algorithms" begin
	for L in [5, 6]
		@test test_excitations(L) < 1.0e-8
	end
end

@testset "tdvp" begin
	for L in [5, 6]
		@test test_tdvp(L) < 1.0e-4
	end
end

@testset "tebd" begin
	for L in [5, 6]
		@test test_tebd(L) < 1.0e-6
	end
end



