# It is nontrivial to give a reasonable random MPS ansatz in presence of symmetry


function FiniteMPS(f, ::Type{T}, physpaces::Vector{S}, virtualpaces::Vector{S}) where {T <: Number, S <: EuclideanSpace}
	L = length(physpaces)
	(length(virtualpaces) == L+1) || throw(DimensionMismatch())
	# (dim(virtualpaces[end]) == 1) || throw(DimensionMismatch())
	(dim(virtualpaces[1]) == 1) || throw(DimensionMismatch())
	any([dim(item)==0 for item in virtualpaces]) &&  @warn "auxiliary space is empty."
	mpstensors = [TensorMap(f, T, virtualpaces[i] ⊗ physpaces[i] ← virtualpaces[i+1]) for i in 1:L]
	return FiniteMPS(mpstensors)
end

function FiniteMPS(f, ::Type{T}, physpaces::Vector{S}, maxvirtualspace::S; left::S=oneunit(S), right::S=oneunit(S)) where {T <: Number, S <: EuclideanSpace}
	L = length(physpaces)
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = infimum(fuse(virtualpaces[i-1], physpaces[i-1]), maxvirtualspace)
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(physpaces[i]', virtualpaces[i+1]))
	end
	return FiniteMPS(f, T, physpaces, virtualpaces)
end


# for symmetric state
"""
	prodmps(::Type{T}, physectors::Vector{S}) where {T <: Number, S <: Sector}
	generate product state in MPS form with given quantum numbers on each site
"""
function prodmps(::Type{T}, physpaces::Vector{S}, physectors::Vector; left::S=oneunit(S), right::S=oneunit(S)) where {T <: Number, S <: GradedSpace}
	L = length(physpaces)
	(L == length(physectors)) || throw(DimensionMismatch())
	physectors = [convert(sectortype(S), item) for item in physectors]

	# the total quantum number is ignored in the Abelian case
	if (sectortype(S) <: TensorKit.AbelianIrrep)
		rightind, = ⊗(physectors...)
		right = S((rightind=>1,))
	end
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = fuse(virtualpaces[i-1], S((physectors[i-1]=>1,)) )
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(virtualpaces[i+1],  S((physectors[i]=>1,))' ))
	end
	return FiniteMPS(ones, T, physpaces, virtualpaces)
end
prodmps(::Type{T}, physpace::S, physectors::Vector; kwargs...) where {T <: Number, S <: GradedSpace} = prodmps(T, [physpace for i in 1:length(physectors)], physectors; kwargs...)
prodmps(physpaces::Vector{S}, physectors::Vector; kwargs...) where {T <: Number, S <: GradedSpace} = prodmps(Float64, physpaces, physectors; kwargs...)
prodmps(physpace::S, physectors::Vector; kwargs...) where {T <: Number, S <: GradedSpace} = prodmps(Float64, physpace, physectors; kwargs...)

function randommps(::Type{T}, physpaces::Vector{S}; sector::Sector, D::Int) where {T <: Number, S <: EuclideanSpace}
	virtualpaces = max_virtual_spaces(physpaces, sector)
	any([dim(item)==0 for item in virtualpaces]) &&  @warn "auxiliary space is empty."
	L = length(physpaces)
	mpstensors = Vector{Any}(undef, L)
	trunc = truncdim(D)
	for i in 1:L
		tmp = TensorMap(randn, T, virtualpaces[i] ⊗ physpaces[i] ← virtualpaces[i+1])
		u, s, v = stable_svd!(tmp, trunc=trunc)
		mpstensors[i] = u
		virtualpaces[i+1] = space(u, 3)'
	end
	return FiniteMPS([mpstensors...])
end

# initializer with no symmetry
function prodmps(::Type{T}, physpaces::Vector{S}, physectors::Vector{Vector{T2}}) where {T <: Number, S <: Union{CartesianSpace, ComplexSpace}, T2 <:Number}
	(length(physpaces) == length(physectors)) || throw(DimensionMismatch())
	for (a, b) in zip(physpaces, physectors)
		(dim(a) == length(b)) || throw(DimensionMismatch())
	end
	virtualspace = oneunit(S)
	return FiniteMPS([TensorMap(convert(Vector{T}, b), virtualspace ⊗ a ← virtualspace) for (a, b) in zip(physpaces, physectors)])
end
prodmps(physpaces::Vector{S}, physectors::Vector{Vector{T}}) where {S <: Union{CartesianSpace, ComplexSpace}, T <:Number} = prodmps(T, physpaces, physectors)
prodmps(::Type{T}, physectors::Vector{Vector{T2}}) where {T <: Number, T2 <:Number} = prodmps(T, [ℂ^(length(item)) for item in physectors], physectors )
prodmps(physectors::Vector{Vector{T}}) where {T <: Number} = prodmps(T, physectors)

function onehot(::Type{T}, d::Int, i::Int) where {T <: Number}
	((i >= 0) && (i < d)) || throw(BoundsError())
	r = zeros(T, d)
	r[i+1] = 1
	return r
end
function prodmps(::Type{T}, ds::Vector{Int}, physectors::Vector{Int}) where {T <: Number}
	(length(ds) == length(physectors)) || throw(DimensionMismatch())
	return prodmps([onehot(T, d, i) for (d, i) in zip(ds, physectors)])
end
prodmps(ds::Vector{Int}, physectors::Vector{Int}) = prodmps(Float64, ds, physectors)


function randommps(::Type{T}, physpaces::Vector{Int}; D::Int) where {T <: Number}
	physpaces = [ℂ^b for b in physpaces]
	virtualspace = ℂ^D
	return FiniteMPS(randn, T, physpaces, virtualspace)
end
randommps(physpaces::Vector{Int}; D::Int) = randommps(Float64, physpaces; D=D)
randommps(::Type{T}, L::Int; d::Int, D::Int) where {T <: Number} = randommps(T, [d for i in 1:L]; D=D)
randommps(L::Int; d::Int, D::Int) = randommps(Float64, L; d=d, D=D)


