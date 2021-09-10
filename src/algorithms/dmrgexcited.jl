


function _project!(y, projectors)
	for p in projectors
		y = axpy!(-dot(y, p), p, y)
	end	
	return y
end


function leftsweep!(m::ProjectedExpectationCache, alg::DMRG1)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	cstorages = m.cenvs
	projectors = m.projectors
	Energies = Float64[]
	delta = 0.
	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		p1 = [c_proj(projectors[l][site], cstorages[l][site], cstorages[l][site+1]) for l in 1:length(cstorages)]
		sitemps = _project!(copy(mps[site]), p1)
		eigvals, vecs = eigsolve(x->_project!(ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), p1), sitemps, 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be left canonical
		Q, R = leftorth!(vecs[1], alg=QR())
		mps[site] = Q
		mps[site+1] = @tensor tmp[-1 -2; -3] := R[-1, 1] * mps[site+1][1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		# for l in 1:length(cstorages)
		#     cstorages[l][site+1] = updateleft(cstorages[l][site], mps[site], projectors[l][site])
		# end
		updateleft!(m, site)
	end
	return Energies, delta
end

function rightsweep!(m::ProjectedExpectationCache, alg::DMRG1)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	cstorages = m.cenvs
	projectors = m.projectors
	Energies = Float64[]
	delta = 0.
	for site in length(mps):-1:2
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")
		p1 = [c_proj(projectors[l][site], cstorages[l][site], cstorages[l][site+1]) for l in 1:length(cstorages)]
		sitemps = _project!(copy(mps[site]), p1)
		eigvals, vecs = eigsolve(x->_project!(ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), p1), sitemps, 1, :SR, Lanczos())
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")		
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be right canonical
		L, Q = rightorth(vecs[1], (1,), (2,3), alg=LQ())
		mps[site] = permute(Q, (1,2), (3,))
		mps[site-1] = @tensor tmp[-1 -2; -3] := mps[site-1][-1, -2, 1] * L[1, -3]
		# hstorage[site] = updateright(hstorage[site+1], mps[site], mpo[site], mps[site])
		# for l in 1:length(cstorages)
		#     cstorages[l][site] = updateright(cstorages[l][site+1],  mps[site], projectors[l][site])
		# end
		updateright!(m, site)
	end
	return Energies, delta
end

function _c2_prime_2(x1::MPSTensor{S}, x2::MPSTensor{S}, cleft::AbstractTensorMap{S, 1, 1}, cright::AbstractTensorMap{S, 1, 1}) where {S <: EuclideanSpace}
	@tensor tmp[-1 -2; -3 -4] := cleft[-1, 1] * x1[1, -2, 2] * x2[2, -3, 3] * cright[-4, 3]
end


function leftsweep!(m::ProjectedExpectationCache, alg::DMRG2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	cstorages = m.cenvs
	projectors = m.projectors
	trunc = alg.trunc
	Energies = Float64[]
	delta = 0.
	for site in 1:length(mps)-2
		(alg.verbosity > 2) && println("sweeping from left to right at bond: $site.")
		p2 = [_c2_prime_2(projectors[l][site], projectors[l][site+1], cstorages[l][site], cstorages[l][site+2]) for l in 1:length(cstorages)]
		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		eigvals, vecs = eigsolve(x->_project!(ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), p2), _project!(twositemps, p2), 1, :SR, Lanczos())
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
		# for l in 1:length(cstorages)
		#     cstorages[l][site+1] = updateleft(cstorages[l][site], mps[site], projectors[l][site])
		# end
		updateleft!(m, site)
	end
	return Energies, delta
end


function rightsweep!(m::ProjectedExpectationCache, alg::DMRG2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	cstorages = m.cenvs
	projectors = m.projectors
	trunc = alg.trunc
	Energies = Float64[]
	delta = 0.
	for site in length(mps)-1:-1:1
		(alg.verbosity > 2) && println("sweeping from right to left at bond: $site.")
		p2 = [_c2_prime_2(projectors[l][site], projectors[l][site+1], cstorages[l][site], cstorages[l][site+2]) for l in 1:length(cstorages)]
		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		eigvals, vecs = eigsolve(x->_project!(ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), p2), _project!(twositemps, p2), 1, :SR, Lanczos())
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
		# for l in 1:length(cstorages)
		#     cstorages[l][site+1] = updateright(cstorages[l][site+2], mps[site+1], projectors[l][site+1])
		# end
		updateright!(m, site+1)
	end
	return Energies, delta
end


function leftsweep!(m::ProjectedExpectationCache, alg::DMRG1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	cstorages = m.cenvs
	projectors = m.projectors
	trunc = alg.trunc
	Energies = Float64[]
	delta = 0.
	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		right_expansion!(m, site, alg.expan, trunc)

		p1 = [c_proj(projectors[l][site], cstorages[l][site], cstorages[l][site+1]) for l in 1:length(cstorages)]
		sitemps = _project!(copy(mps[site]), p1)

		eigvals, vecs = eigsolve(x->_project!(ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), p1), sitemps, 1, :SR, Lanczos())

		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")
		delta = max(delta, calc_galerkin(m, site) )
		# prepare mps site tensor to be left canonical
		Q, R = leftorth!(vecs[1], alg=QR())
		mps[site] = Q
		mps[site+1] = @tensor tmp[-1 -2; -3] := R[-1, 1] * mps[site+1][1, -2, -3]
		# hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		# for l in 1:length(cstorages)
		#     cstorages[l][site+1] = updateleft(cstorages[l][site], mps[site], projectors[l][site])
		# end
		updateleft!(m, site)
	end
	return Energies, delta
end

function rightsweep!(m::ProjectedExpectationCache, alg::DMRG1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	cstorages = m.cenvs
	projectors = m.projectors
	trunc = alg.trunc
	Energies = Float64[]
	delta = 0.
	for site in length(mps):-1:2
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")
		left_expansion!(m, site, alg.expan, trunc)
		p1 = [c_proj(projectors[l][site], cstorages[l][site], cstorages[l][site+1]) for l in 1:length(cstorages)]
		sitemps = _project!(copy(mps[site]), p1)

		eigvals, vecs = eigsolve(x->_project!(ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), p1), sitemps, 1, :SR, Lanczos())
		
		push!(Energies, eigvals[1])
		(alg.verbosity > 2) && println("Energy after optimization on site $site is $(Energies[end]).")	
		delta = max(delta, calc_galerkin(m, site) )	
		# prepare mps site tensor to be right canonical
		L, Q = rightorth(vecs[1], (1,), (2,3), alg=LQ())
		mps[site] = permute(Q, (1,2), (3,))
		mps[site-1] = @tensor tmp[-1 -2; -3] := mps[site-1][-1, -2, 1] * L[1, -3]
		# hstorage[site] = updateright(hstorage[site+1], mps[site], mpo[site], mps[site])
		# for l in 1:length(cstorages)
		#     cstorages[l][site] = updateright(cstorages[l][site+1],  mps[site], projectors[l][site])
		# end
		updateright!(m, site)
	end
	return Energies, delta
end


function sweep!(m::Union{ExpectationCache, ProjectedExpectationCache}, alg::DMRGAlgorithm=DMRG2(trunc=default_truncation(spacetype(m.mps))); kwargs...)
	Energies1, delta1 = leftsweep!(m, alg; kwargs...)
	Energies2, delta2 = rightsweep!(m, alg; kwargs...)
	return vcat(Energies1, Energies2), max(delta1, delta2)
end

