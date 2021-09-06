



function expectation(psiA::FiniteMPS, h::MPOHamiltonian, psiB::FiniteMPS)
    (length(psiA) == length(psiB)) || throw(DimensionMismatch())
    (mod(length(psiA), period(h))==0) || throw(DimensionMismatch())
    hold = r_RR(psiA, h, psiB)
    for i in length(psiA):-1:1
        hold = updateright(hold, psiA[i], h[i], psiB[i])
    end
    hleft = l_LL(psiA, h, psiB)
    (length(hleft) == length(hold)) || error("something wrong.")
    r = zero(promote_type(scalar_type(psiA), scalar_type(h), scalar_type(psiB)))
    for (a, b) in zip(hleft, hold)
    	r += @tensor tmp = a[1,2,3] * b[1,2,3]
    end
    return r
end
expectation(h::MPOHamiltonian, psi::FiniteMPS) = expectation(psi, h, psi)
