module FiniteMPTools

using Logging: @warn
using Parameters, KrylovKit, TensorKit
import TensorKit, LinearAlgebra

# auxiliary
export MPSTruncation, default_truncation, Coefficient, value, scalar_type, is_constant
export AbelianMatrix, abelian_matrix_from_dense

# mps
export AbstractMPS, FiniteMPS, iscanonical, canonicalize!, bond_dimension, bond_dimensions, distance2, distance, space_l, space_r, sector
export physical_spaces, FiniteDensityOperatorMPS, DensityOperator, infinite_temperature_state, prodmps, randommps

# # infinitemps
# export InfiniteMPS

# mpo
export AbstractMPO, FiniteMPO, prodmpo, expectation, SiteOp, ScalarSiteOp, svdcompress!
export MPOCompression, SVDCompression, Deparallelise, compress!

# mpohamiltonian
export SchurMPOTensor, MPOHamiltonian, period, odim, ExponentialDecayTerm, GenericDecayTerm, PowerlawDecayTerm, exponential_expansion

# environments
export environments

# circuit
export QuantumGate, QuantumCircuit, apply!, positions, fuse_gates

# operators, easier interface for building quantum operators incrementally, and used for TEBD. Should it really be here in this package?
export QTerm, QuantumOperator, add!, isstrict, qterms, superoperator, add_unitary!, add_dissipation!

# algorithms
export trotter_propagator, DMRG1, DMRG2, DMRG1S, TDVP1, TDVP2, TDVP1S, leftsweep!, rightsweep!, sweep!, compute!, ground_state!
export SubspaceExpansionScheme, CHExpansion, OptimalExpansion
export DMRGAlgorithm, TDVPAlgorithm
export ExactFiniteMPS, exact_diagonalization, exact_timeevolution
# time evolve stepper
export timeevo!, AbstractStepper, TEBDStepper, TDVPStepper, change_tspan_dt, TEBDCache, TDVPCache, timeevo_cache
# two-time correlations
export correlation_2op_1t, exact_correlation_2op_1t, correlation_2op_1τ
# thermal state
export purified_thermalize, purified_infinite_temperature_state, purified_thermal_state

# utilities
export boson_matrices, spin_half_matrices, spin_matrices


#default settings
module Defaults
	const maxiter = 100
	const tolgauge = 1e-14
	const tol = 1e-12
	const verbosity = 1
	import KrylovKit: GMRES
	const solver = GMRES(tol=1e-12, maxiter=100)
end


# auxiliary
include("auxiliary/truncation.jl")
include("auxiliary/deparlise.jl")
include("auxiliary/distance.jl")
include("auxiliary/coeff.jl")
include("auxiliary/periodicarray.jl")
include("auxiliary/stable_svd.jl")
include("auxiliary/others.jl")

include("auxiliary/abelianmatrix.jl")

# mps
include("mps/abstractdefs.jl")
include("mps/transfer.jl")
include("mps/bondview.jl")
include("mps/finitemps.jl")
include("mps/adjointmps.jl")
include("mps/density_operator.jl")
include("mps/exactmps.jl")
include("mps/orth.jl")
include("mps/initializers.jl")
include("mps/arithmetics.jl")

# # infinitemps
# include("infinitemps/orth.jl")
# include("infinitemps/infinitemps.jl")
# include("infinitemps/transfer.jl")
# include("infinitemps/arithmetics.jl")

# mpo
include("mpo/abstractdefs.jl")
include("mpo/transfer.jl")
include("mpo/finitempo.jl")
include("mpo/adjointmpo.jl")
include("mpo/compress.jl")
include("mpo/initializers.jl")
include("mpo/arithmetics.jl")
include("mpo/deparlise.jl")
include("mpo/siteop.jl")

# mpo hamiltonian
include("mpohamiltonian/abstractsitempo.jl")
include("mpohamiltonian/genericmpotensor.jl")
include("mpohamiltonian/schurmpotensor.jl")
include("mpohamiltonian/mpohamiltonian.jl")
include("mpohamiltonian/transfer.jl")
include("mpohamiltonian/arithmetics.jl")
include("mpohamiltonian/constructor.jl")

# environments
include("envs/abstractdefs.jl")
include("envs/finiteenv.jl")
include("envs/overlap.jl")
# include("envs/infiniteenv.jl")

# circuit for TEBD
include("circuit/abstractdefs.jl")
include("circuit/gate.jl")
include("circuit/circuit.jl")
include("circuit/apply_gates.jl")
include("circuit/gate_fusion.jl")

# operators
include("operators/qterm.jl")
include("operators/abelian_qterms.jl")
include("operators/quantumoperator.jl")
include("operators/superoperator.jl")
include("operators/expecs.jl")
include("operators/tompo.jl")


# algorithms
include("algorithms/tebd.jl")
include("algorithms/derivatives.jl")
include("algorithms/expansion/expansion.jl")
include("algorithms/dmrg.jl")
include("algorithms/dmrgexcited.jl")
include("algorithms/tdvp.jl")
include("algorithms/timeevo.jl")
include("algorithms/twotimecorrs.jl")
include("algorithms/thermalstate.jl")
include("algorithms/exactdiag.jl")

# utilities
include("utilities/spin_siteops.jl")
include("utilities/boson_siteops.jl")
include("utilities/fermion_siteops.jl")
include("utilities/models.jl")


# function _precompile_()
# 	@assert precompile(spinal_fermion_site_ops_u1_su2, ())
# 	@assert precompile(spinal_fermion_site_ops_u1_u1, ())
# 	@assert precompile(spinal_fermion_site_ops_dense, ())
# 	# ph = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)
# end

# _precompile_()

end
