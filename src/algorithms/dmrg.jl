abstract type AbstractDMRGAlgorithm end


struct DMRG1 <: AbstractDMRGAlgorithm 
	verbosity::Int
end

DMRG1(;verbosity::Int=1) = DMRG1(verbosity) 

function leftsweep!(m::FiniteEnv, alg::DMRG1)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	Energies = Float64[]
	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")
		# prepare mps site tensor to be left canonical
		Q, R = leftorth!(vecs[1], alg=QR())
		mps[site] = Q
		mps[site+1] = @tensor tmp[-1 -2; -3] := R[-1, 1] * mps[site+1][1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		updateleft!(m, site)
	end
	return Energies
end

function rightsweep!(m::FiniteEnv, alg::DMRG1)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	Energies = Float64[]
	for site in length(mps):-1:2
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")
		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")		
		# prepare mps site tensor to be right canonical
		L, Q = rightorth(vecs[1], (1,), (2,3), alg=LQ())
		mps[site] = permute(Q, (1,2), (3,))
		mps[site-1] = @tensor tmp[-1 -2; -3] := mps[site-1][-1, -2, 1] * L[1, -3]
		# hstorage[site] = updateright(hstorage[site+1], mps[site], mpo[site], mps[site])
		updateright!(m, site)
	end
	return Energies
end


struct DMRG2{C<:TensorKit.TruncationScheme} <: AbstractDMRGAlgorithm
	trunc::C
	verbosity::Int
end
DMRG2(;trunc::TruncationScheme=default_truncation(), verbosity::Int=1) = DMRG2(trunc, verbosity)


function leftsweep!(m::FiniteEnv, alg::DMRG2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	Energies = Float64[]
	for site in 1:length(mps)-2
		(alg.verbosity > 2) && println("sweeping from left to right at bond: $site.")
		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		eigvals, vecs = eigsolve(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), twositemps, 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on bond $site is $(Energies[end]).")				
		# prepare mps site tensor to be left canonical
		u, s, v, err = tsvd!(vecs[1], trunc=trunc)
		mps[site] = u
		mps[site+1] = @tensor tmp[-1 -2; -3] := s[-1, 1] * v[1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		updateleft!(m, site)
	end
	return Energies	
end

function rightsweep!(m::FiniteEnv, alg::DMRG2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	Energies = Float64[]
	for site in length(mps)-1:-1:1
		(alg.verbosity > 2) && println("sweeping from right to left at bond: $site.")
		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		eigvals, vecs = eigsolve(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), twositemps, 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on bond $site is $(Energies[end]).")	
		# prepare mps site tensor to be right canonical
		u, s, v, err = tsvd!(vecs[1], trunc=trunc)	
		mps[site] = @tensor tmp[-1 -2; -3] := u[-1, -2, 1] * s[1, -3]
		mps[site+1] = permute(v, (1,2), (3,))
		mps.s[site+1] = s
		# hstorage[site+1] = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])
		updateright!(m, site+1)
	end
	return Energies
end



struct DMRG1S{C<:TensorKit.TruncationScheme, E<:SubspaceExpansionScheme} <: AbstractDMRGAlgorithm
	trunc::C
	expan::E
	verbosity::Int
end

DMRG1S(;trunc::TruncationScheme=default_truncation(), expan::SubspaceExpansionScheme=default_expansion(), verbosity::Int=1) = DMRG1S(trunc, expan, verbosity)


function leftsweep!(m::FiniteEnv, alg::DMRG1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	Energies = Float64[]
	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		# subspace expansion
		right_expansion!(m, site, alg.expan, trunc)
		# end of subspace expansion

		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")
		# prepare mps site tensor to be left canonical
		Q, R = leftorth!(vecs[1], alg=QR())
		mps[site] = Q
		mps[site+1] = @tensor tmp[-1 -2; -3] := R[-1, 1] * mps[site+1][1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		updateleft!(m, site)
	end
	return Energies
end

function rightsweep!(m::FiniteEnv, alg::DMRG1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	Energies = Float64[]
	for site in length(mps):-1:2
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")

		# subspace expansion
		left_expansion!(m, site, alg.expan, trunc)
		# end of subspace expansion

		eigvals, vecs = eigsolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), mps[site], 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")		
		# prepare mps site tensor to be right canonical
		L, Q = rightorth(vecs[1], (1,), (2,3), alg=LQ())
		mps[site] = permute(Q, (1,2), (3,))
		mps[site-1] = @tensor tmp[-1 -2; -3] := mps[site-1][-1, -2, 1] * L[1, -3]
		# hstorage[site] = updateright(hstorage[site+1], mps[site], mpo[site], mps[site])
		updateright!(m, site)
	end
	return Energies
end

