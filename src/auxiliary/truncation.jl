# customized truncation


struct MPSTruncation <: TensorKit.TruncationScheme
	dim::Int
	ϵ::Float64
	verbosity::Int
end


MPSTruncation(D::Int, ϵ::Real; verbosity::Int=1) = MPSTruncation(D, convert(Float64, ϵ), verbosity)
MPSTruncation(;D::Int, ϵ::Real, verbosity::Int=1) = MPSTruncation(D, convert(Float64, ϵ), verbosity)

compute_size(v::AbstractVector) = length(v)
function compute_size(v::AbstractDict) 
	init = 0
	for (c, b) in v
		init += dim(c) * length(b)
	end
	return init
end

function TensorKit._truncate!(v, trunc::MPSTruncation, p::Real = 2)
	n = TensorKit._norm(v, p, 0.)

	verbosity = trunc.verbosity
	s_1 = compute_size(v)
	# truncate using relative cutof
	v, err1 = TensorKit._truncate!(v, TensorKit.truncbelow(trunc.ϵ / n), p)
	s_2 = compute_size(v)
	if s_2 <= trunc.dim
		(verbosity >= 2) && println("sum: $s_1 -> $s_2.")
		return v, norm([err1], p)
	end
	# if still larger than D, then truncate using D
	v, err2 = TensorKit._truncate!(v, TensorKit.truncdim(trunc.dim), p)
	s_3 = compute_size(v)
	err = norm((err1, err2), p)

	(verbosity > 0) && println("sum: $s_1 -> $s_2 -> $s_3, maximum $(trunc.dim), truncation error: absolute=$(err), relative=$(err/n).")

	return v, err
end

const DefaultSVDCutoff = 1.0e-8
const DefaultVerbosity = 1
const DefaultNonSymmetricD = 200
const DefaultAbelianSymmetricD = 1000
const DefaultNonAbelianSymmetricD = 10000

function default_truncation(::Type{S}) where {S <: EuclideanSpace}
	D = 0
	cutoff = DefaultSVDCutoff
	if S == ComplexSpace || S == CartesianSpace
		D = DefaultNonSymmetricD
	elseif S <: GradedSpace
		st = sectortype(S)
		if st <: TensorKit.AbelianIrrep
			D = DefaultAbelianSymmetricD
		else
			D = DefaultNonAbelianSymmetricD
		end
	else
		throw(ArgumentError("unkonwn space type $S."))
	end
	return MPSTruncation(D, cutoff; verbosity=DefaultVerbosity)
end

default_truncation() = MPSTruncation(DefaultNonSymmetricD, DefaultSVDCutoff, verbosity=DefaultVerbosity)

