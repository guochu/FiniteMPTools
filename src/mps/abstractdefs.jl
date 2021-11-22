


const MPSTensor{S} = AbstractTensorMap{S, 2, 1} where {S<:EuclideanSpace}

# The bond tensor are just the singlur vector but has to be stored as a general matrix since  TensorKit does not specialize for Diagonal Matrices
const MPSBondTensor{S} = AbstractTensorMap{S, 1, 1} where {S<:EuclideanSpace}

const NonSymmetricMPSTensor{S} = MPSTensor{S} where {S <: Union{ComplexSpace, CartesianSpace}}

abstract type AbstractMPS end


