

function boson_annihilation(;d::Int)
	(d <= 1) && error("d must be larger than 1.")
	a = zeros(Float64, d, d)
	for i = 1:(d - 1)
		a[i, i+1] = sqrt(i)
	end
	return a	
end
boson_creation(;d::Int) = boson_annihilation(d=d)'


function boson_matrices(;d::Int)
	a = boson_annihilation(d=d)
	adag = boson_creation(d=d)
	n = adag * a
	return Dict("-"=>a, "+"=>adag, "n"=>n)
end

