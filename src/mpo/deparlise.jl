
# mpo deparallelisation


function deparallelise_left!(x::AbstractMPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	for i = 1:(length(x)-1)
		(verbosity > 2) && println("deparallelisation sweep from left to right on site: $i.")
		# M, Tm = deparallelise(x[i], (3,), tol, verbosity=verbosity)
		M, Tm = deparallelise(x[i], (1,2,4), (3,); row=false, tol=tol, verbosity=verbosity)
		if dim(M) > 0
		    x[i] = permute(M, (1,2), (4,3))
		    @tensor tmp[-1 -2; -3 -4] := Tm[-1, 1] * x[i+1][1,-2,-3,-4]
		    x[i+1] = tmp
		else
			(verbosity >= 1) && println("mpo becomes empty after deparallelisation left.")
			return nothing
		end
	end
	return x
end

function deparallelise_right!(x::AbstractMPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	for i = length(x):-1:2
	    (verbosity > 2) && println("deparallelisation sweep from right to left on site: $i.")
	    Tm, M = deparallelise(x[i], (1,), (2,3,4); row=true, tol=tol, verbosity=verbosity)
	    if dim(M) > 0
	    	# ii = isomorphism()
	        x[i] = permute(M, (1,2), (3,4))
	        @tensor tmp[-1 -2; -3 -4] := x[i-1][-1,-2,1,-4] * Tm[1,-3]
	        x[i-1] = tmp
	    else
	    	(verbosity >= 1) && println("mpo becomes empty after deparallelisation right.")
	    	return nothing
	    end
	end
	return x
end

function deparallelise!(x::AbstractMPO; kwargs...)
	r = deparallelise_left!(x; kwargs...)
	if !isnothing(r)
		r = deparallelise_right!(x; kwargs...)
	end
	return r
end

"""
	deparallelise(x::AbstractMPO)
	reduce the bond dimension of mpo using deparallelisation
"""
deparallelise(x::AbstractMPO; kwargs...) = deparallelise!(copy(x); kwargs...)