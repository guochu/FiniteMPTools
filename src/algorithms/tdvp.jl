abstract type TDVPAlgorithm end

struct TDVP1{T} <: TDVPAlgorithm
	stepsize::T
	D::Int 
	ishermitian::Bool
	verbosity::Int
end

TDVP1(; stepsize::Number, D::Int=Defaults.D, ishermitian::Bool=false, verbosity::Int=1) = TDVP1(stepsize, D, ishermitian, verbosity)

function _leftsweep!(m::ExpectationCache, alg::TDVP1)
	increase_bond!(m, D=alg.D)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	driver = isherm ? Lanczos() : Arnoldi()

	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		tmp, info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver)

		mps[site], v = leftorth!(tmp, alg=QR())
		hnew = updateleft(hstorage[site], mps[site], mpo[site], mps[site])

		v, info = exponentiate(x->c_prime(x, hnew, hstorage[site+1]), -dt/2, v, driver)
		mps[site+1] = @tensor tmp[-1 -2; -3] := v[-1, 1] * mps[site+1][1, -2, -3]
		hstorage[site+1] = hnew
	end
	site = length(mps)
	mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt, mps[site], driver)
end

function _rightsweep!(m::ExpectationCache, alg::TDVP1)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	driver = isherm ? Lanczos() : Arnoldi()
	
	for site in length(mps)-1:-1:1
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")

		v, Q = rightorth(mps[site+1], (1,), (2,3), alg=LQ()) 
		mps[site+1] = permute(Q, (1, 2), (3,))
		hnew = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])

		v, info = exponentiate(x->c_prime(x, hstorage[site+1], hnew), -dt/2, v, driver)
		hstorage[site+1] = hnew
		mps[site] = mps[site] * v

		mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver)
	end
end

struct TDVP2{T, C<:TensorKit.TruncationScheme} <: TDVPAlgorithm
	stepsize::T
	ishermitian::Bool	
	trunc::C
	verbosity::Int
end
TDVP2(;ishermitian::Bool=false, verbosity::Int=1, trunc::TruncationScheme=default_truncation(), stepsize::Number) = TDVP2(stepsize, ishermitian, trunc, verbosity)



function _leftsweep!(m::ExpectationCache, alg::TDVP2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	dt = alg.stepsize
	isherm = alg.ishermitian
	driver = isherm ? Lanczos() : Arnoldi()

	for site in 1:length(mps)-2
		(alg.verbosity > 2) && println("sweeping from left to right at bond: $site.")
		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		twositemps, info = exponentiate(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt/2, twositemps, driver)
		u, s, v, err = stable_tsvd!(twositemps, trunc=trunc)
		mps[site] = u

		@tensor sitemps[-1 -2; -3] := s[-1, 1] * v[1, -2, -3]
		hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		mps[site+1], info = exponentiate(x->ac_prime(x, mpo[site+1], hstorage[site+1], hstorage[site+2]), -dt/2, sitemps, driver)
	end

	site = length(mps)-1
	(alg.verbosity > 2) && println("sweeping from left to right at bond: $site.")
	@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
	twositemps, info = exponentiate(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt, twositemps, driver)
	u, s, v, err = stable_tsvd!(twositemps, trunc=trunc)
	mps[site] = u * s
	mps[site+1] = permute(v, (1,2), (3,))

	hstorage[site+1] = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])
	mps.s[site+1] = s
end


function _rightsweep!(m::ExpectationCache, alg::TDVP2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	dt = alg.stepsize
	isherm = alg.ishermitian
	driver = isherm ? Lanczos() : Arnoldi()
	for site in length(mps)-2:-1:1
		(alg.verbosity > 2)  && println("sweeping from right to left at bond: $site.")
		mps[site+1], info = exponentiate(x->ac_prime(x, mpo[site+1], hstorage[site+1], hstorage[site+2]), -dt/2, mps[site+1], driver)

		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		twositemps, info = exponentiate(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt/2, twositemps, driver)
		u, s, v, err = stable_tsvd!(twositemps, trunc=trunc)
		mps[site] = u * s
		mps[site+1] = permute(v, (1,2), (3,))
		hstorage[site+1] = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])
		mps.s[site+1] = s
	end
end


struct TDVP1S{T, C<:TensorKit.TruncationScheme, E<:SubspaceExpansionScheme} <: TDVPAlgorithm
	stepsize::T
	ishermitian::Bool	
	trunc::C
	expan::E
	verbosity::Int
end
TDVP1S(;ishermitian::Bool=false, verbosity::Int=1, stepsize::Number, trunc::TruncationScheme=default_truncation(), expan::SubspaceExpansionScheme=default_expansion()) = TDVP1S(
	stepsize, ishermitian, trunc, expan, verbosity)


function _leftsweep!(m::ExpectationCache, alg::TDVP1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	trunc = alg.trunc
	driver = isherm ? Lanczos() : Arnoldi()

	for site in 1:length(mps)-1
		(alg.verbosity > 2) && println("sweeping from left to right at site: $site.")
		right_expansion!(m, site, alg.expan, trunc)

		tmp, info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver)

		mps[site], v = leftorth!(tmp, alg=QR())
		hnew = updateleft(hstorage[site], mps[site], mpo[site], mps[site])

		v, info = exponentiate(x->c_prime(x, hnew, hstorage[site+1]), -dt/2, v, driver)
		mps[site+1] = @tensor tmp[-1 -2; -3] := v[-1, 1] * mps[site+1][1, -2, -3]
		hstorage[site+1] = hnew
	end
	site = length(mps)
	mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt, mps[site], driver)
end

function _rightsweep!(m::ExpectationCache, alg::TDVP1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	trunc = alg.trunc
	driver = isherm ? Lanczos() : Arnoldi()
	
	for site in length(mps)-1:-1:1
		(alg.verbosity > 2) && println("sweeping from right to left at site: $site.")

		v, Q = rightorth(mps[site+1], (1,), (2,3), alg=LQ()) 
		mps[site+1] = permute(Q, (1, 2), (3,))
		hnew = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])

		v, info = exponentiate(x->c_prime(x, hstorage[site+1], hnew), -dt/2, v, driver)
		hstorage[site+1] = hnew
		mps[site] = mps[site] * v

		if site > 1
			left_expansion!(m, site, alg.expan, trunc)
		end

		mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver)
	end
end



function sweep!(m::ExpectationCache, alg::TDVPAlgorithm; kwargs...)
	_leftsweep!(m, alg; kwargs...)
	_rightsweep!(m, alg; kwargs...)
end



