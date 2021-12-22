
const DeparalleliseTol = 1.0e-11

function _istwocolumnparallel(cola, colb, tol::Real)
    (length(cola) != length(colb)) && throw(DimensionMismatch())
    (length(cola) == 0) && throw(ArgumentError("the column should not be empty."))
    n = length(cola)
    idx = findall(x->abs(x)>tol, cola)
    (length(idx) == 0) && throw(ArgumentError("the column can not be all zeros."))
    factor = colb[idx[1]]/cola[idx[1]]
    dif = colb - factor*cola
    for i in dif
        if abs(i) > tol
        	return false, 0.
        end
    end
    return true, factor
end

function _getridofzerocol(m::AbstractArray{T, 2}, tol::Real, verbosity::Int=0) where {T}
    s1, s2 = size(m)
    zerocols = Vector{Int}(undef, 0)
    for j = 1:s2
    	allzero = true
    	for i=1:s1
    	    if abs(m[i, j]) > tol
    	        allzero = false
    	        break
    	    end
    	end
    	if allzero
    	    (verbosity > 1) && println("all elements of column $j are zero.")
    	    Base.push!(zerocols, j)
    	end
    end
    ns = s2 - length(zerocols)
    (ns == 0 && verbosity > 1) && println("all the columns are zero.")
	mout = zeros(T, (s1, ns))
	j = 1
	for i=1:s2
		if !(i in zerocols)
		    mout[:, j] .= view(m, :, i)
		    j += 1
	    end
	end
	return mout, zerocols
end


function _matrixdeparallelisenozerocols(m::AbstractArray{T, 2}, tol::Real, verbosity::Int=0) where {T}
    s1, s2 = size(m)
    K = []
    Tm = zeros(T, (s2, s2))
    for j = 1:s2
    	exist = false
    	for i=1:length(K)
    	    p, factor = _istwocolumnparallel(K[i], m[:, j], tol)
    	    if p
    	       (verbosity > 1) && println("column $i is in parallel with column $j.")
    	       Tm[i, j] = factor
    	       exist = true
    	       break
    	    end
    	end
    	if !exist
    	    Base.push!(K, m[:, j])
    	    nK = length(K)
    	    Tm[nK, j] = 1
    	end
    end
    nK = length(K)
    M = zeros(T, (s1, nK))
    for j=1:nK
        M[:, j] = K[j]
    end
    return M, Tm[1:nK, :]
end

function matrixdeparlise_col(m::AbstractArray{T, 2}, tol::Real; verbosity::Int=0) where {T}
    mnew, zerocols = _getridofzerocol(m, tol, verbosity)
    M, Tm = _matrixdeparallelisenozerocols(mnew, tol, verbosity)
    # isapprox(mnew, M*Tm) || error("matrixdeparallise error.")
    if isempty(M)
        (verbosity > 1) && println("all the elements of the matrix M are 0.")
        return M, Tm
    end
    Tnew = zeros(T, (size(Tm, 1), size(m, 2)))
    j = 1
    for i = 1:size(Tnew, 2)
        if !(i in zerocols)
            Tnew[:, i] .= Tm[:, j]
            j += 1
        end
    end
    # println("dim $(size(m, 2)) -> $(size(M, 2))")
    isapprox(m, M*Tnew) || error("matrixdeparallise error.")
    return M, Tnew
end

function matrixdeparlise_row(m::AbstractMatrix, tol::Real; verbosity::Int=0)
    a, b = matrixdeparlise_col(transpose(m), tol, verbosity=verbosity)
    return transpose(b), transpose(a)
end

matrixdeparlise(m::AbstractMatrix, row::Bool, tol::Real; verbosity::Int=0) = row ? matrixdeparlise_row(m, tol, verbosity=verbosity) : matrixdeparlise_col(m, tol, verbosity=verbosity)

"""
    deparallelise(t::TensorMap{<:EuclideanSpace}; row::Bool, tol::Real=DeparalleliseTol, verbosity::Int=0)
    t = M * T, eleminate parallel rows (if row=true) or columns (if row=false)
"""
function deparallelise(t::TensorMap{<:EuclideanSpace}; row::Bool, tol::Real=DeparalleliseTol, verbosity::Int=0)
    I = sectortype(t)
    S = spacetype(t)
    A = storagetype(t)
    Qdata = TensorKit.SectorDict{I, A}()
    Rdata = TensorKit.SectorDict{I, A}()
    dims = TensorKit.SectorDict{I, Int}()
    for c in blocksectors(domain(t))
        isempty(block(t,c)) && continue
        # Q, R = _leftorth!(block(t, c), alg, atol)
        Q, R = matrixdeparlise(block(t, c), row, tol; verbosity=verbosity)
        Qdata[c] = Q
        Rdata[c] = R
        dims[c] = size(Q, 2)
    end
    V = S(dims)
    # if alg isa Polar
    #     @assert V ≅ domain(t)
    #     W = domain(t)
    # elseif length(domain(t)) == 1 && domain(t) ≅ V
    #     W = domain(t)
    # elseif length(codomain(t)) == 1 && codomain(t) ≅ V
    #     W = codomain(t)
    # else
    #     W = ProductSpace(V)
    # end
    W = ProductSpace(V)
    return TensorMap(Qdata, codomain(t)←W), TensorMap(Rdata, W←domain(t))	
end
deparallelise(t::AbstractTensorMap, left::Tuple, right::Tuple; kwargs...) = deparallelise(permute(t, left, right); kwargs...)


