



function boson_annihilation(;d::Int)
	(d <= 1) && error("d must be larger than 1.")
	a = zeros(Float64, d, d)
	for i = 1:(d - 1)
		a[i, i+1] = sqrt(i)
	end
	return a	
end
boson_creation(;d::Int) = boson_annihilation(d=d)'

_to_tensor_map(m::AbstractMatrix) = TensorMap(m, ℂ^(size(m, 1)) ← ℂ^(size(m, 1)) )

function boson_matrices(;d::Int)
	a = boson_annihilation(d=d)
	adag = boson_creation(d=d)
	n = adag * a
	return Dict("-"=>a, "+"=>adag, "n"=>n)
end

function spin_half_matrices()
	s_SP = Array{Float64, 2}([0 0; 1 0])
	s_SM = Array{Float64, 2}([0 1; 0 0])
	s_Z = Array{Float64, 2}([-1 0; 0 1])
	s_x = s_SP+s_SM
	s_y = -im*(s_SP-s_SM)
	return Dict("x"=>s_x, "y"=>s_y, "z"=>s_Z, "+"=>s_SP, "-"=>s_SM)
end

function spin_matrices(s::Union{Rational{Int},Int})
    N = Int(2*s)

    Sx = zeros(N+1,N+1)
    Sy = zeros(ComplexF64,N+1,N+1)
    Sz = zeros(N+1,N+1)

    for row=1:(N+1)
        for col=1:(N+1)
            term=sqrt((s+1)*(row+col-1)-row*col)/2.0

            if (row+1==col)
                Sx[row,col]+=term
                Sy[row,col]-=1im*term
            end

            if(row==col+1)
                Sx[row,col]+=term
                Sy[row,col]+=1im*term
            end

            if(row==col)
                Sz[row,col]+=s+1-row
            end

        end
    end
    return Dict("x"=>Sx, "y"=>Sy, "z"=>Sz)
end
