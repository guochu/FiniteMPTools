
"""
	The convention is that the creation operator on the left of the annihilation operator
"""
function spinal_fermion_site_ops_u1_su2()
	ph = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)
	bh = Rep[U₁×SU₂]((0.5, 0.5)=>1)
	vh = oneunit(ph)
	adag = TensorMap(zeros, Float64, vh ⊗ ph ← bh ⊗ ph)
	blocks(adag)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = ones(1,1)
	blocks(adag)[Irrep[U₁](0.5) ⊠ Irrep[SU₂](0)] = sqrt(2) * ones(1,1) 

	onsite_interact = TensorMap(zeros, Float64, ph ← ph)
	blocks(onsite_interact)[Irrep[U₁](0.5) ⊠ Irrep[SU₂](0)] = ones(1, 1)

	JW = TensorMap(ones, Float64, ph ← ph)
	blocks(JW)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = -ones(1, 1)

	# adagJW = TensorMap(zeros, Float64, vh ⊗ ph ← bh ⊗ ph)
	# blocks(adagJW)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = ones(1,1)
	# blocks(adagJW)[Irrep[U₁](0.5) ⊠ Irrep[SU₂](0)] = -sqrt(2) * ones(1,1) 

	# hund operators
	# c↑† ⊗ c↓†
	bhr = Rep[U₁×SU₂]((1, 0)=>1)
	adagadag = TensorMap(ones, Float64, vh ⊗ ph ← bhr ⊗ ph)

	# c↑† ⊗ c↓, this is a spin 1 sector operator!!!
	bhr = Rep[U₁×SU₂]((0, 1)=>1)
	adaga = TensorMap(zeros, Float64, vh ⊗ ph ← bhr ⊗ ph)
	blocks(adaga)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = ones(1, 1) * (-sqrt(3) / 2)

	n = ScalarSiteOp(SiteOp(adag) * SiteOp(adag)')
	return Dict("+"=>SiteOp(adag), "++"=>SiteOp(adagadag), "+-"=>SiteOp(adaga), "n↑n↓"=>ScalarSiteOp(onsite_interact), "JW"=>ScalarSiteOp(JW), "n"=>n)
end

function spinal_fermion_site_ops_u1_u1()
	p = spin_half_matrices()
	p = Dict(k=>abelian_matrix_from_dense(v) for (k, v) in p)
	sp, z = p["+"], p["z"]

	I2 = one(z)
	n = (z + I2) / 2
	JW1 = -z

	adagup = SiteOp(_convert_to_tensor_map(kron(sp, I2))[1])
	adagdown = SiteOp(_convert_to_tensor_map(kron(JW1, sp))[1])
	# adagdown = SiteOp(_convert_to_tensor_map(kron(I2, sp))[1])

	adag = adagup + adagdown

	adagadag = SiteOp(_convert_to_tensor_map(kron(sp*JW1, sp))[1])

	tmp = -_convert_to_tensor_map(kron(sp*JW1, sp'))[1] / sqrt(2)
	mm = 0.5 * (_convert_to_tensor_map(kron(n, I2))[1] - _convert_to_tensor_map(kron(I2, n))[1])
	tmp2 = _convert_to_tensor_map(kron(sp*JW1, sp')')[1] / sqrt(2)
	adaga = SiteOp(tmp) + SiteOp(mm) + SiteOp(tmp2)

	occupy = ScalarSiteOp(adag * adag')
	return Dict("+"=>adag, "++"=>adagadag, "+-"=>adaga, "n↑n↓"=>ScalarSiteOp(_convert_to_tensor_map(kron(n, n))[1]), 
		"JW"=>ScalarSiteOp(_convert_to_tensor_map(kron(JW1, JW1))[1]), "n"=>occupy)
end

function spinal_fermion_site_ops_dense()
	p = spin_half_matrices()
	sp, z = p["+"], p["z"]

	I2 = one(z)
	n = (z + I2) / 2
	JW1 = -z

	adagup = SiteOp(kron(sp, I2))
	adagdown = SiteOp(kron(JW1, sp))
	# adagdown = SiteOp(kron(I2, sp))

	adag = adagup + adagdown

	adagadag = SiteOp(kron(sp*JW1, sp))

	tmp = -kron(sp*JW1, sp') / sqrt(2)
	mm = 0.5 * (kron(n, I2) - kron(I2, n))
	adaga = SiteOp(tmp) + SiteOp(mm) + SiteOp(Matrix(-tmp'))

	occupy = ScalarSiteOp(adag * adag')
	return Dict("+"=>adag, "++"=>adagadag, "+-"=>adaga, "n↑n↓"=>ScalarSiteOp(kron(n, n)), "JW"=>ScalarSiteOp(kron(JW1, JW1)), "n"=>occupy)
end

