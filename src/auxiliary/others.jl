
# some dirty functions which should not be exposed to users.
get_trivial_leg(m::AbstractTensorMap) = Tensor(ones,eltype(m),oneunit(space(m,1)))

# do we really need this one?
function _add_legs(m::AbstractTensorMap{S, 1, 1}) where {S <: EuclideanSpace}
	util=get_trivial_leg(m)
	@tensor m4[-1 -2; -3 -4] := util[-1] * m[-2, -4] * conj(util[-3])
	return m4
end