


struct InfiniteMPS{A<:MPSTensor, B<:MPSBondTensor} <: AbstractMPS
	data::PeriodicArray{A, 1}
	svectors::PeriodicArray{B, 1}
end

function Base.getproperty(psi::InfiniteMPS, s::Symbol)
	if s == :s
		return MPSBondView(psi)
	else
		return getfield(psi, s)
	end
end

raw_data(psi::InfiniteMPS) = psi.data
raw_singular_matrices(psi::InfiniteMPS) = psi.svectors
Base.size(psi::InfiniteMPS,i) = size(raw_data(psi),i)
Base.length(psi::InfiniteMPS) = size(raw_data(psi),1)
Base.eltype(psi::InfiniteMPS) = eltype(raw_data(psi))
Base.isempty(psi::InfiniteMPS) = isempty(raw_data(psi))

Base.getindex(psi::InfiniteMPS, i::Int) = getindex(raw_data(psi), i)
Base.lastindex(psi::InfiniteMPS) = lastindex(raw_data(psi))
Base.firstindex(psi::InfiniteMPS) = firstindex(raw_data(psi))

Base.getindex(psi::InfiniteMPS,r::AbstractRange{Int64}) = [psi[ri] for ri in r]


function Base.setindex!(psi::InfiniteMPS, v, i::Int)
	_check_mps_tensor_dir(v) || throw(SpaceMismatch())
	return setindex!(raw_data(psi), v, i)
end 


Base.copy(psi::InfiniteMPS) = InfiniteMPS(copy(raw_data(psi)), copy(raw_singular_matrices(psi)))

Base.repeat(m::InfiniteMPS,i::Int) = InfiniteMPS(repeat(raw_data(m),i),repeat(raw_singular_matrices(m),i))
Base.similar(m::InfiniteMPS) = InfiniteMPS(similar(raw_data(m)), similar(raw_singular_matrices(m)))

svectors_uninitialized(psi::InfiniteMPS) = _svectors_uninitialized(psi)
svectors_initialized(psi::InfiniteMPS) = !svectors_uninitialized(psi)

TensorKit.spacetype(::Type{InfiniteMPS{A, B}}) where {A, B} = spacetype(A)
TensorKit.spacetype(x::InfiniteMPS) = spacetype(typeof(x))
scalar_type(::Type{InfiniteMPS{A, B}}) where {A, B} = eltype(A)
scalar_type(x::InfiniteMPS) = scalar_type(typeof(x))

virtualspace(psi::InfiniteMPS, n::Int) = space(psi[n+1], 1)

# TensorKit.norm(st::InfiniteMPS) = norm(st.AC[1])


space_l(state::InfiniteMPS) = space(state[1], 1)
space_r(state::InfiniteMPS) = space(state[end], 3)


"""
	stores the right canonical mps tensors (Bs) together with the bond tensors (Cs)
	Bs and Cs satisfies AC = Cs[i-1] * Bs[i], notice the difference with FiniteMPS,
	where AC = Cs[i] * Bs[i], namely the index of bond matrix has been shifted.
"""
function InfiniteMPS(f, ::Type{T}, pspaces::AbstractArray{S,1}, Dspaces::AbstractArray{S,1}) where {T <:Number, S <: EuclideanSpace}
    InfiniteMPS([TensorMap(f, T, Dspaces[mod1(i-1,length(Dspaces))]*pspaces[i],Dspaces[i]) for i in 1:length(pspaces)])
end

bond_matrix_type(::Type{A}) where {S<:EuclideanSpace, A <: MPSTensor{S}} = tensormaptype(S, 1, 1, eltype(A))

function InfiniteMPS(A::AbstractArray{T,1}) where T<:MPSTensor

    #we make a copy, and are therfore garantueeing no side effects for the user
    AR = PeriodicArray(A[:])

    #initial guess for CR
    CR = PeriodicArray([isomorphism(Matrix{eltype(AR[i])},space(AR[i+1],1),space(AR[i+1],1)) for i in 1:length(A)])


    # AL = similar(AR)

    # uniform_leftorth!(AL,CR,AR;kwargs...)
    # uniform_rightorth!(AR,CR,AL;kwargs...)


    mps = InfiniteMPS{T,bond_matrix_type(T)}(AR,CR)
    (space_l(mps) == space_r(mps)') || throw(SpaceMismatch("boundary space mismatch."))
    return mps
end

function InfiniteMPS(psi::FiniteMPS{A, B}; iscanonical::Bool=false) where {A, B}
	if iscanonical
		ss = convert(Vector{B}, raw_singular_matrices(psi)[1:length(psi)]) 
		return InfiniteMPS(PeriodicArray(raw_data(psi)), PeriodicArray(ss))
	else
		psi_c = copy(psi)
	    canonicalize!(normalize!(psi_c))
		return InfiniteMPS(psi_c, iscanonical=true)
	end
end

function _canonicalize!(psi::InfiniteMPS; kwargs...)
	AR = raw_data(psi)
	CR = raw_singular_matrices(psi)
	AL = similar(AR)
	uniform_leftorth!(AL,CR,AR;kwargs...)
	uniform_rightorth!(AR,CR,AL;kwargs...)
	return AL, AR, CR
end

function canonicalize!(psi::InfiniteMPS; kwargs...)
	AL, AR, CR = _canonicalize!(psi; kwargs...)
	return InfiniteMPS(AR, CR)
end


Base.circshift(psi::InfiniteMPS,n::Int) = InfiniteMPS(circshift(raw_data(psi),n), circshift(raw_singular_matrices(psi), n))


"""
	bond_dimension(psi::InfiniteMPS, bond::Int)
	return bond dimension as an integer
"""
bond_dimension(psi::InfiniteMPS, bond::Int) = begin
	dim(space(psi[bond], 3))
end 
bond_dimensions(psi::InfiniteMPS) = PeriodicArray([bond_dimension(psi, i) for i in 1:length(psi)])
bond_dimension(psi::InfiniteMPS) = maximum(raw_data(bond_dimensions(psi)))

physical_spaces(psi::InfiniteMPS) = PeriodicArray([space(item, 2) for item in raw_data(psi)])


"""
    l_RR(state,location)
    Left dominant eigenvector of the AR-AR transfermatrix
"""
l_RR(state::InfiniteMPS,loc::Int=1) = @tensor out[-1;-2]:=state.s[loc-1][1,-2]*conj(state.s[loc-1][1,-1])
"""
    l_LL(state,location)
    Left dominant eigenvector of the AL-AL transfermatrix
"""
l_LL(state::InfiniteMPS,loc::Int=1) = isomorphism(Matrix{scalar_type(state)}, space(state[loc],1),space(state[loc],1))

"""
    r_RR(state,location)
    Right dominant eigenvector of the AR-AR transfermatrix
"""
r_RR(state::InfiniteMPS,loc::Int=length(state)) = isomorphism(Matrix{scalar_type(state)},space(state[loc],3),space(state[loc,3]))

"""
    r_LL(state,location)
    Right dominant eigenvector of the AL-AL transfermatrix
"""
r_LL(state::InfiniteMPS,loc::Int=length(state))= @tensor out[-1;-2]:=state.s[loc][-1,1]*conj(state.s[loc][-2,1])

function is_right_canonical(psi::InfiniteMPS; kwargs...)
	all([_is_right_canonical(item; kwargs...) for item in raw_data(psi)]) || return false
	# we also check whether the singular vectors are the correct Schmidt numbers
	iden = one(scalar_type(psi))
	hold = l_RR(psi)
	hnew = copy(hold)
	for i in 1:length(psi)
		isapprox(tr(hnew), iden; kwargs...) || return false
		hnew = updateleft(hnew, psi[i], psi[i])
	end
	isapprox(hold, hnew; kwargs...) || return false
	return true
end
iscanonical(psi::InfiniteMPS; kwargs...) = is_right_canonical(psi; kwargs...)
