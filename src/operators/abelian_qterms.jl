


QTerm(pos::Vector{Int}, m::Vector{<:AbstractMatrix}; coeff::Union{Number, Function, Coefficient}=1.) = QTerm(pos, _to_tensor_map.(m); coeff=coeff)

function _convert_to_tensor_map(m::AbelianMatrix{S}, left::S) where {S <: GradedSpace}
	phy = space(m)
	right_sectors = sectortype(S)[]
	for (k, v) in raw_data(m)
		ko, ki = k
		for _l in sectors(left)
			(dim(left, _l) == 1) || throw(ArgumentError("wrong leftind."))
			_r, = first(ko ⊗ _l) ⊗ conj(ki)
			if !(_r in right_sectors)
				push!(right_sectors, _r)
			end
		end
	end
	right = S(item=>1 for item in right_sectors)
	r = TensorMap(zeros, left ⊗ phy ← right ⊗ phy )
	for (k, v) in raw_data(m)
		ko, ki = k
		for _l in sectors(left)
			_r, = first(ko ⊗ _l) ⊗ conj(ki)
			copyto!(r[(_l, ko, conj(_r), conj(ki))], v)
		end
	end
	return r, right
end

function QTerm(pos::Vector{Int}, m::Vector{<:AbelianMatrix{S}}; coeff::Union{Number, Function, Coefficient}=1.) where S
	left = oneunit(S)
	ms = []
	for item in m
		# println(typeof(left))
		mj, left = _convert_to_tensor_map(item, left)
		push!(ms, mj)
	end
	return QTerm(pos, [ms...]; coeff=coeff)
end

