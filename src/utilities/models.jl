


function boson_hubbard_chain_util(L::Int, p; J::Real, U::Real, μ::Real=0.)
	adag, a, n = p["+"], p["-"], p["n"]
	nn = n * n
	terms = []
	# one site terms
	for i in 1:L
		push!(terms, QTerm(i=>(U/2) * (nn-n) + μ * n , coeff=1))
	end
	# nearest-neighbour interactions
	for i in 1:L-1
		t = QTerm(i=>adag, i+1=>a, coeff=-J)
		push!(terms, t)
		push!(terms, t')
	end
	return QuantumOperator([terms...])
end

"""
	boson_hubbard_chain(L::Int; J::Real, U::Real, μ::Real=0., d::Int=5)
	boson hubbard chain with U₁ symmetry
"""
boson_hubbard_chain(L::Int; d::Int=5, kwargs...) = boson_hubbard_chain_util(L, Dict(k=>abelian_matrix_from_dense(v) for (k, v) in boson_matrices(d=d)); kwargs...)

function heisenberg_xxz_util(L::Int, p; J::Real, Jzz::Real, hz::Real)
    sp, sm, z = p["+"], p["-"], p["z"]
    terms = []
    # one site terms
    for i in 1:L
        push!(terms, QTerm(i=>z, coeff=hz))
    end
    # nearest-neighbour interactions
    for i in 1:L-1
        t = QTerm(i=>sp, i+1=>sm, coeff=2*J)
        push!(terms, t)
        push!(terms, t')
        push!(terms, QTerm(i=>z, i+1=>z, coeff=Jzz))
    end
    return QuantumOperator([terms...])
end

"""
    heisenberg xxz chain
"""
heisenberg_xxz_chain(L::Int; kwargs...) = heisenberg_xxz_util(L, Dict(k=>abelian_matrix_from_dense(v) for (k, v) in spin_half_matrices()); kwargs...)

function boundary_driven_xxz_util(L::Int, p; J::Real, Jzz::Real, hz::Real, nl::Real, Λl::Real, nr::Real, Λr::Real, Λp::Real)
	(nl >=0 && nl <=1) || error("nl must be between 0 and 1.")
	(nr >=0 && nr <=1) || error("nr must be between 0 and 1.")
	(Λl >=0 && Λr >= 0 && Λp>= 0) || error("Λ should not be negative.")
	sp, sm, sz = p["+"], p["-"], p["z"]
	lindblad = superoperator(-im * heisenberg_xxz_util(L, p; J=J, Jzz=Jzz, hz=hz), fuser=⊗)

	gammal_plus = Λl*nl
	gammal_minus = Λl*(1-nl)
	gammar_plus = Λr*nr
	gammar_minus = Λr*(1-nr)

	for i in 1:L
		add_dissipation!(lindblad, QTerm(i=>sz, coeff=sqrt(Λp)))
	end
	add_dissipation!(lindblad, QTerm(1=>sp, coeff=sqrt(gammal_plus)))
	add_dissipation!(lindblad, QTerm(1=>sm, coeff=sqrt(gammal_minus)))
	add_dissipation!(lindblad, QTerm(L=>sp, coeff=sqrt(gammar_plus)))
	add_dissipation!(lindblad, QTerm(L=>sm, coeff=sqrt(gammar_minus)))
	return lindblad
end

boundary_driven_xxz(L::Int; kwargs...) = boundary_driven_xxz_util(L, Dict(k=>abelian_matrix_from_dense(v) for (k, v) in spin_half_matrices()); kwargs...)

