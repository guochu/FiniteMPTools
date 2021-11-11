abstract type DMRGAlgorithm end

@with_kw struct DMRG1 <: DMRGAlgorithm 
	D::Int = Defaults.D
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	verbosity::Int = Defaults.verbosity
end

function calc_galerkin(m::Union{ExpectationCache, ProjectedExpectationCache}, site::Int)
	mpsj = m.mps[site]
	try
		return norm(leftnull(mpsj)' * ac_prime(mpsj, m.mpo[site], m.hstorage[site], m.hstorage[site+1]))
	catch
		return norm(permute(ac_prime(mpsj, m.mpo[site], m.hstorage[site], m.hstorage[site+1]), (1,), (2,3)) * rightnull(permute(mpsj, (1,), (2,3) ) )' )
	end
end

# delayed evaluation of galerkin error.
function leftsweep!(m::ExpectationCache, alg::DMRG1)
	# try increase the bond dimension if the bond dimension of the state is less than D given by alg
	increase_bond!(m, D=alg.D)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	Energies = Float64[]
	delta = 0.
	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")
		# galerkin error
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be left canonical
		Q, R = leftorth!(vecs[1], alg=QR())
		mps[site] = Q
		mps[site+1] = @tensor tmp[-1 -2; -3] := R[-1, 1] * mps[site+1][1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		updateleft!(m, site)
	end
	return Energies, delta
end

function rightsweep!(m::ExpectationCache, alg::DMRG1)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	Energies = Float64[]
	delta = 0.
	for site in length(mps):-1:2
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")
		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")		
		# galerkin error
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be right canonical
		L, Q = rightorth(vecs[1], (1,), (2,3), alg=LQ())
		mps[site] = permute(Q, (1,2), (3,))
		mps[site-1] = @tensor tmp[-1 -2; -3] := mps[site-1][-1, -2, 1] * L[1, -3]
		# hstorage[site] = updateright(hstorage[site+1], mps[site], mpo[site], mps[site])
		updateright!(m, site)
	end
	return Energies, delta
end


@with_kw struct DMRG2{C<:TensorKit.TruncationScheme} <: DMRGAlgorithm
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol	
	verbosity::Int = Defaults.verbosity
	trunc::C = default_truncation()
end


function leftsweep!(m::ExpectationCache, alg::DMRG2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	Energies = Float64[]
	delta = 0.
	for site in 1:length(mps)-2
		(alg.verbosity > 2) && println("sweeping from left to right at bond: $site.")
		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		eigvals, vecs = eigsolve(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), twositemps, 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on bond $site is $(Energies[end]).")				
		# prepare mps site tensor to be left canonical
		u, s, v, err = stable_tsvd!(vecs[1], trunc=trunc)
		normalize!(s)
		mps[site] = u
		mps[site+1] = @tensor tmp[-1 -2; -3] := s[-1, 1] * v[1, -2, -3]
		# compute error
		err_1 = @tensor twositemps[1,2,3,4]*conj(u[1,2,5])*conj(s[5,6])*conj(v[6,3,4])
        delta = max(delta,abs(1-abs(err_1)))
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		updateleft!(m, site)
	end
	return Energies, delta
end

function rightsweep!(m::ExpectationCache, alg::DMRG2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	Energies = Float64[]
	delta = 0.
	for site in length(mps)-1:-1:1
		(alg.verbosity > 2) && println("sweeping from right to left at bond: $site.")
		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		eigvals, vecs = eigsolve(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), twositemps, 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on bond $site is $(Energies[end]).")	
		# prepare mps site tensor to be right canonical
		u, s, v, err = stable_tsvd!(vecs[1], trunc=trunc)	
		normalize!(s)
		mps[site] = @tensor tmp[-1 -2; -3] := u[-1, -2, 1] * s[1, -3]
		mps[site+1] = permute(v, (1,2), (3,))
		mps.s[site+1] = s
		# compute error
		err_1 = @tensor twositemps[1,2,3,4]*conj(u[1,2,5])*conj(s[5,6])*conj(v[6,3,4])
        delta = max(delta,abs(1-abs(err_1)))
		# hstorage[site+1] = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])
		updateright!(m, site+1)
	end
	return Energies, delta
end



@with_kw  struct DMRG1S{C<:TensorKit.TruncationScheme, E<:SubspaceExpansionScheme} <: DMRGAlgorithm
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol	
	verbosity::Int = Defaults.verbosity
	trunc::C = default_truncation()
	expan::E = default_expansion()
end

function leftsweep!(m::ExpectationCache, alg::DMRG1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	Energies = Float64[]
	delta = 0.
	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		# subspace expansion
		right_expansion!(m, site, alg.expan, trunc)
		# end of subspace expansion

		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")
		# galerkin error
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be left canonical
		Q, R = leftorth!(vecs[1], alg=QR())
		mps[site] = Q
		mps[site+1] = @tensor tmp[-1 -2; -3] := R[-1, 1] * mps[site+1][1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		updateleft!(m, site)
	end
	return Energies, delta
end

function rightsweep!(m::ExpectationCache, alg::DMRG1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	Energies = Float64[]
	delta = 0.
	for site in length(mps):-1:2
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")

		# subspace expansion
		left_expansion!(m, site, alg.expan, trunc)
		# end of subspace expansion

		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")		
		# galerkin error
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be right canonical
		L, Q = rightorth(vecs[1], (1,), (2,3), alg=LQ())
		mps[site] = permute(Q, (1,2), (3,))
		mps[site-1] = @tensor tmp[-1 -2; -3] := mps[site-1][-1, -2, 1] * L[1, -3]
		# hstorage[site] = updateright(hstorage[site+1], mps[site], mpo[site], mps[site])
		updateright!(m, site)
	end
	return Energies, delta
end

"""
	compute!(env::AbstractCache, alg::DMRGAlgorithm)
	execute dmrg iterations
"""
function compute!(env::AbstractCache, alg::DMRGAlgorithm)
	all_energies = Float64[]
	iter = 0
	delta = 2 * alg.tol
	# do a first sweep anyway?
	# Energies, delta = sweep!(env, alg)
	while iter < alg.maxiter && delta > alg.tol
		Energies, delta = sweep!(env, alg)
		append!(all_energies, Energies)
		iter += 1
		(alg.verbosity > 2) && println("finish the $iter-th sweep with error $delta", "\n")
	end
	return all_energies, delta
end

"""
	return the ground state
	ground_state!(state::FiniteMPS, h::Union{MPOHamiltonian, FiniteMPO}, alg::DMRGAlgorithm)
"""
ground_state!(state::FiniteMPS, h::Union{MPOHamiltonian, FiniteMPO}, alg::DMRGAlgorithm=DMRG1S()) = compute!(environments(h, state), alg)


