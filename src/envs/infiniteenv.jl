


mutable struct InfiniteEnv{M<:MPOHamiltonian, V<:MPSTensor, B <: MPSBondTensor, H, S}
	mpo::M
	As::PeriodicArray{V, 1}
    Bs::PeriodicArray{V, 1}
    Cs::PeriodicArray{B, 1}
    lw::PeriodicArray{Vector{H},1}
    rw::PeriodicArray{Vector{H},1}
    solver::S
end

function Base.getproperty(m::InfiniteEnv, s::Symbol)
    if s == :state
        return InfiniteMPS(m.Bs, m.Cs)
    elseif s == :h
        return m.mpo
    else
        return getfield(m, s)
    end
end

scalar_type(::Type{InfiniteEnv{M, V, B, H, S}}) where {M,V,B,H,S} = eltype(V)
scalar_type(x::InfiniteEnv) = scalar_type(typeof(x))

leftenv(m::InfiniteEnv, pos::Int) = envs.lw[pos]
rightenv(m::InfiniteEnv, pos::Int) = envs.rw[pos]


_lw_rw_tensor_type(::Type{A}) where {A <: MPSTensor} = tensormaptype(spacetype(A), 1, 2, eltype(A))


function gen_lw_rw(h::MPOHamiltonian, As::PeriodicArray{A, 1}) where {A <: MPSTensor}
	L = length(mps)
	H = _lw_rw_tensor_type(A)
	lw = PeriodicArray{Vector{H},1}(undef,L)
    rw = PeriodicArray{Vector{H},1}(undef,L)
    d = odim(h)
    T = promote_type(scalar_type(h), scalar_type(mps))

    for i in 1:L
    	lw[i] = Vector{H}(undef, d)
    	rw[i] = Vector{H}(undef, d)
    	for j in 1:d
    		lw[i][j] = TensorMap(rand,T,space(mps[i],1), h[i].imspaces[j] * space(mps[i],1))
            rw[i,j] = TensorMap(rand,T,space(mps[i],3), h[i].omspaces[j]' * space(mps[i],3))
    	end
    end
    return (lw, rw)
end


#randomly initialize envs
function environments(ham::MPOHamiltonian, state::InfiniteMPS; solver=Defaults.solver, kwargs...)
    As, Bs, Cs = _canonicalize!(state; kwargs...)
    (lw,rw) = gen_lw_rw(ham, Bs)
    state = InfiniteMPS(Bs, Cs)
    envs = InfiniteEnv(ham, state, As, Bs, Cs, lw, rw, solver)
    calculate!(envs)
end


function calculate!(envs::InfiniteEnv)
    calclw!(envs)
    calcrw!(envs)
    return envs
end

function recalculate!(envs::InfiniteEnv, As::Vector, Bs::Vector, Cs::Vector)
    (envs.lw,envs.rw) = gen_lw_rw(envs.h, Bs)
    envs.As = As
    envs.Bs = Bs
    envs.Cs = Cs
    return calculate!(envs)
end

function calclw!(envs::InfiniteEnv)
    fixpoints = envs.lw
    ham = envs.h
    As = envs.As
    Bs = envs.Bs
    Cs = envs.Cs
    st = envs.state
    solver = envs.solver

    len = length(As)

    #the start element
    leftutil = Tensor(ones,scalar_type(envs),space(ham[1,1,1],1))
    @tensor fixpoints[1][1][-1; -2 -3] = l_LL(st)[-1,-3]*conj(leftutil[-2])
    (len>1) && left_cyclethrough!(1,fixpoints,ham,As)

    for i = 2:odim(ham)
        prev = copy(fixpoints[1][i])

        rmul!(fixpoints[1][i],0);
        left_cyclethrough!(i,fixpoints,ham,As)

        # isa(ham.Os[i, i], Number) || throw(ArgumentError("diagonals should be scalars."))

        if isscal(ham, i) #identity matrices; do the hacky renormalization

            #subtract fixpoints
            @tensor tosvec[-1 -2;-3] := fixpoints[1][i][-1,-2,-3]-fixpoints[1][i][1,-2,2]*r_LL(st)[1,2]*l_LL(st)[-1,-3]

            (fixpoints[1][i],convhist) = linsolve(x->x - updateleft(x,As,As,r_LL(st),l_LL(st)), tosvec,prev,solver) 

            convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            (len>1) && left_cyclethrough!(i,fixpoints,ham,As)

            #go through the unitcell, again subtracting fixpoints
            for potato in 1:len
                @tensor fixpoints[potato][i][-1 -2;-3]-=fixpoints[potato][i][1,-2,2]*r_LL(st,potato-1)[2,1]*l_LL(st,potato)[-1,-3]
            end

        else
            if all([contains(ham,x,i,i) for x in 1:len])

                (fixpoints[1][i],convhist) = linsolve(x->x-updateleft(x, ham[:,i,i], As, As), fixpoints[1][i],prev,solver)

                convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            end
            (len>1) && left_cyclethrough!(i,fixpoints,ham,As)
        end

    end

    return fixpoints
end

function calcrw!(envs::InfiniteEnv)
    fixpoints = envs.rw
    ham = envs.h
    As = envs.As
    Bs = envs.Bs
    Cs = envs.Cs
    st = envs.state
    solver = envs.solver
    len = length(st)

    #the start element
    rightutil = Tensor(ones,scalar_type(envs),space(ham[len,1,1],3))
    @tensor fixpoints[end][end][-1 -2;-3] = r_RR(st)[-1,-3]*conj(rightutil[-2])

    (len>1) && right_cyclethrough!(ham.odim,fixpoints,ham,Bs) #populate other sites

    for i = (ham.odim-1):-1:1
        prev = copy(fixpoints[end][i])
        rmul!(fixpoints[end][i],0);
        right_cyclethrough!(i,fixpoints,ham,Bs)


        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #subtract fixpoints
            @tensor tosvec[-1 -2;-3]:=fixpoints[end][i][-1,-2,-3]-fixpoints[end][i][1,-2,2]*l_RR(st)[2,1]*r_RR(st)[-1,-3]

            (fixpoints[end][i],convhist) = linsolve(x -> x - transfer_right(x,Bs,Bs,l_RR(st),r_RR(st)), tosvec,prev,solver) 
            convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            len>1 && right_cyclethrough!(i,fixpoints,ham,Bs)

            #go through the unitcell, again subtracting fixpoints
            for potatoe in 1:len
                @tensor fixpoints[potatoe][i][-1 -2;-3]-=fixpoints[potatoe][i][1,-2,2]*l_RR(st,potatoe+1)[2,1]*r_RR(st,potatoe)[-1,-3]
            end
        else
            if all([contains(ham,x,i,i) for x in 1:len])

                (fixpoints[end][i],convhist) = linsolve(x -> x-transfer_right(x,ham[:,i,i],Bs,Bs), fixpoints[end][i],prev,solver) 

                convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            end

            (len>1) && right_cyclethrough!(i,fixpoints,ham,Bs)
        end
    end

    return fixpoints
end

function left_cyclethrough!(index::Int,fp,ham,As) #see code for explanation
    for i=1:length(fp)
        rmul!(fp[i+1][index],0);

        for j=index:-1:1
            contains(ham,i,j,index) || continue

            if isscal(ham,i,j,index)
                fp[i+1][index] += transfer_left(fp[i][j],As[i],As[i])*ham.Os[i,j,index]
            else
                fp[i+1][index] += transfer_left(fp[i][j],ham[i,j,index],As[i],As[i])
            end
        end
    end
end

function right_cyclethrough!(index,fp,ham,Bs) #see code for explanation
    for i=length(fp):(-1):1
        rmul!(fp[i-1][index],0);

        for j=index:ham.odim
            contains(ham,i,index,j) || continue

            if isscal(ham,i,index,j)
                fp[i-1][index] += transfer_right(fp[i][j], Bs[i], Bs[i]) * ham.Os[i,index,j]
            else
                fp[i-1][index] += transfer_right(fp[i][j], ham[i,index,j], Bs[i], Bs[i])
            end
        end
    end
end









