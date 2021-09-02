
function _is_prop_util(x::M, a::M) where M
	scal = dot(a,x)/dot(a,a)
	diff = x-scal*a
	scal = (scal ≈ 0.0) ? 0.0 : scal #shouldn't be necessary (and I don't think it is)
	return norm(diff)<1e-14,scal
end

function isid(x::MPOTensor)
    cod = space(x,1)*space(x,2);
    dom = space(x,3)'*space(x,4)';

    #would like to have an 'isisomorphic'
    for c in union(blocksectors(cod), blocksectors(dom))
        blockdim(cod, c) == blockdim(dom, c) || return false,0.0;
    end

    id = isomorphism(Matrix{eltype(x)},cod,dom)

    return _is_prop_util(x, id)
end

isid(x::Number) = true, x

function isid(x::MPSBondTensor)
	(domain(x) == codomain(x)) || return false, 0.0
	id = isomorphism(Matrix{eltype(x)}, codomain(x), domain(m))
	return _is_prop_util(x, id)
end

abstract type AbstractSiteMPOTensor{S} end


TensorKit.spacetype(::Type{<:AbstractSiteMPOTensor{S}}) where S = S
TensorKit.spacetype(x::AbstractSiteMPOTensor) = spacetype(typeof(x))
scalar_type(x::AbstractSiteMPOTensor) = scalar_type(typeof(x))

raw_data(m::AbstractSiteMPOTensor) = m.Os

Base.size(m::AbstractSiteMPOTensor) = size(raw_data(m))
Base.size(m::AbstractSiteMPOTensor, i::Int) = size(raw_data(m), i)
odim(m::AbstractSiteMPOTensor) = size(m, 1)
function Base.getindex(m::AbstractSiteMPOTensor, j::Int, k::Int) 
	T = scalar_type(m)
	r = getindex(raw_data(m), j, k)
	if isa(r, T)
		imspace = m.imspaces[j]
		omspace = m.omspaces[k]
		pspace = m.pspace
		if r == zero(T)
			return TensorMap(zeros, T, imspace*pspace,omspace*pspace)
		else
			return r * isomorphism(Matrix{T}, imspace*pspace,omspace*pspace)
		end
	else
		return r
	end
end 

Base.keys(x::AbstractSiteMPOTensor) = Iterators.filter(a->contains(x, a[1],a[2]),Iterators.product(1:size(x, 1),1:size(x, 2)))
opkeys(x::AbstractSiteMPOTensor) = Iterators.filter(a-> !isscal(x,a[1],a[2]),keys(x))
scalkeys(x::AbstractSiteMPOTensor) = Iterators.filter(a-> isscal(x,a[1],a[2]),keys(x))



