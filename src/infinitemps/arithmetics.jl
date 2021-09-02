


function LinearAlgebra.dot(a::InfiniteMPS,b::InfiniteMPS;krylovdim::Int = 30)
    init = TensorMap(rand,ComplexF64,space_r(a),space_r(b))
    num = lcm(length(a),length(b))

    (vals,vecs,convhist) = eigsolve(x->updateright(x, a[1:num], b[1:num]),init,1,:LM,Arnoldi(krylovdim=krylovdim))
    convhist.converged == 0 && @info "dot mps not converged"
    return vals[1]
end


function LinearAlgebra.norm(a::InfiniteMPS; kwargs...)
	v = dot(a, a)
	(imag(v) >= 1.0e-8) && @warn "imaginary part is too large."
	return sqrt(real(v))
end

