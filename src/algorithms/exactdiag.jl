

function init_h_center(mpo::Union{FiniteMPO, MPOHamiltonian}, mps::ExactFiniteMPS)
	# (length(mpo) == length(mps)) || throw(DimensionMismatch())
	(spacetype(mpo) == spacetype(mps)) || throw(SpaceMismatch())
	isstrict(mpo) || throw(ArgumentError("operator must be strict."))
	center = mps.center
	right = r_RR(mps, mpo, mps)
	L = length(mps)
	left = l_LL(mps, mpo, mps)
	for i in L:-1:center+1
		right = updateright(right, mps[i], mpo[i], mps[i])
	end
	for i in 1:center-1
		left = updateleft(left, mps[i], mpo[i], mps[i])
	end
	return left, right
end

_default_len(h::FiniteMPO) = length(h)
_default_len(h::MPOHamiltonian) = period(h)

function exact_diagonalization(h::Union{FiniteMPO, MPOHamiltonian}; sector::Sector= first(sectors(oneunit(spacetype(h)))), 
	len::Int=_default_len(h), ishermitian::Bool, num::Int=1, which=:SR) 
	driver = ishermitian ? Lanczos() : Arnoldi()

	physpaces = physical_spaces(h)
	if isa(h, MPOHamiltonian)
		physpaces = [physpaces[i] for i in 1:len]
	end

	mps = ExactFiniteMPS(randn, scalar_type(h), physpaces, sector=sector)
	middle_site = mps.center

	left, right = init_h_center(h, mps)

	vals,vecs,info = eigsolve(x->ac_prime(x, h[middle_site], left, right), mps[middle_site], num, which, driver)

	(info.converged >= num) || @warn "only $(info.converged) converged."
	states = Vector{typeof(mps)}(undef, num)
	for i in 1:num
		states[i] = copy(mps)
		states[i][middle_site] = vecs[i]
	end
	return vals[1:num], states
end

function _exact_timeevolution_util(h::Union{FiniteMPO, MPOHamiltonian}, t::Number, psi::ExactFiniteMPS, left, right; ishermitian::Bool)
	driver = ishermitian ? Lanczos() : Arnoldi()
	middle_site = psi.center
	mpsj, info = exponentiate(x->ac_prime(x, h[middle_site], left, right), t, psi[middle_site], driver)
	(info.converged >= 1) || error("fail to converge.")
	psi_2 = copy(psi)
	psi_2[middle_site] = mpsj
	return psi_2
end

function exact_timeevolution(h::Union{FiniteMPO, MPOHamiltonian}, t::Number, psi::ExactFiniteMPS; ishermitian::Bool)
	left, right = init_h_center(h, psi)
	return _exact_timeevolution_util(h, t, psi, left, right; ishermitian=ishermitian)
end

