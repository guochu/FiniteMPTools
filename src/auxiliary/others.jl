
# some dirty functions which should not be exposed to users.
get_trivial_leg(m::AbstractTensorMap) = Tensor(ones,eltype(m),oneunit(space(m,1)))

# do we really need this one?
function _add_legs(m::AbstractTensorMap{S, 1, 1}) where {S <: EuclideanSpace}
	util=get_trivial_leg(m)
	@tensor m4[-1 -2; -3 -4] := util[-1] * m[-2, -4] * conj(util[-3])
	return m4
end



loose_isometry(cod::TensorKit.EuclideanTensorSpace, dom::TensorKit.EuclideanTensorSpace) =
    loose_isometry(Matrix{Float64}, cod, dom)
loose_isometry(P::TensorKit.EuclideanTensorMapSpace) = loose_isometry(codomain(P), domain(P))
loose_isometry(A::Type{<:DenseMatrix}, P::TensorKit.EuclideanTensorMapSpace) =
    loose_isometry(A, codomain(P), domain(P))
loose_isometry(A::Type{<:DenseMatrix}, cod::TensorKit.EuclideanTensorSpace, dom::TensorKit.EuclideanTensorSpace) =
    loose_isometry(A, convert(ProductSpace, cod), convert(ProductSpace, dom))
function loose_isometry(::Type{A},
                    cod::ProductSpace{S},
                    dom::ProductSpace{S}) where {A<:DenseMatrix, S<:EuclideanSpace}
    t = TensorMap(s->A(undef, s), cod, dom)
    for (c, b) in blocks(t)
        TensorKit._one!(b)
    end
    return t
end

