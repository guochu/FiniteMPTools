


function stable_tsvd(m::AbstractTensorMap, args...; trunc)
	try
		return tsvd(m, args...; trunc=trunc, alg=TensorKit.SDD())
	catch
		return tsvd(m, args...; trunc=trunc, alg=TensorKit.SVD())
	end
end

function stable_tsvd!(m::AbstractTensorMap; trunc)
	try
		return tsvd!(copy(m), trunc=trunc, alg=TensorKit.SDD())
	catch
		return tsvd!(m, trunc=trunc, alg=TensorKit.SVD())
	end
end
