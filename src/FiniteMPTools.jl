module FiniteMPTools

using Logging: @warn
using KrylovKit, TensorKit
import TensorKit, LinearAlgebra

# auxiliary
export MPSTruncation, default_truncation, Coefficient, value, scalar_type, is_constant
export boson_matrices, spin_half_matrices, spin_matrices, AbelianMatrix, abelian_matrix_from_dense

# mps
export AbstractMPS, FiniteMPS, iscanonical, canonicalize!, bond_dimension, bond_dimensions, distance2, distance, space_l, space_r, physical_spaces
export prodmps, randommps

# # infinitemps
# export InfiniteMPS

# mpo
export AbstractMPO, FiniteMPO, prodmpo, deparallelise!, deparallelise, expectation, SiteOp

# mpohamiltonian
export SchurMPOTensor, MPOHamiltonian, period, odim

# environments
export environments

# circuit
export QuantumGate, QuantumCircuit, apply!, positions, fuse_gates

# operators, easier interface for building quantum operators incrementally, and used for TEBD. Should it really be here in this package?
export QTerm, QOperator, add!, isstrict, qterms

# algorithms
export trotter_propagator, infinite_trotter_propagator, DMRG1, DMRG2, DMRG1S, TDVP1, TDVP2, TDVP1S, leftsweep!, rightsweep!, sweep!
export SubspaceExpansionScheme, CHExpansion
export ExactFiniteMPS, exact_diagonalization, exact_timeevolution


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

include("auxiliary/smatrix.jl")
include("auxiliary/abelian_particles.jl")

# mps
include("mps/abstractdefs.jl")
include("mps/transfer.jl")
include("mps/bondview.jl")
include("mps/finitemps.jl")
include("mps/exactmps.jl")
include("mps/adjointmps.jl")
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

# environments
include("envs/abstractdefs.jl")
include("envs/finiteenv.jl")
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
include("operators/qoperator.jl")
include("operators/expecs.jl")
include("operators/tompo.jl")


# algorithms
include("algorithms/tebd.jl")
include("algorithms/derivatives.jl")
include("algorithms/expansion/expansion.jl")
include("algorithms/dmrg.jl")
include("algorithms/dmrgexcited.jl")
include("algorithms/tdvp.jl")
include("algorithms/exactdiag.jl")

end
