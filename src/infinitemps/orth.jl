"
solves AL * C = C * A in-place
"
function uniform_leftorth!(AL,CR, A; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
    #(_,CR[end]) = leftorth!(CR[end], alg = TensorKit.QRpos());
    normalize!(CR[end]);

    iteration = 1;
    delta = 2*tol;

    while iteration < maxiter && delta > tol

        # if iteration>10 #when qr starts to fail, start using eigs - should be kw arg

        #     alg = Arnoldi(krylovdim = 30, tol = max(delta*delta,tol/10),maxiter=maxiter)

        #     (vals,vecs) = @closure eigsolve(CR[end], 1, :LM,alg) do x
        #         transfer_left(x,A,AL)
        #     end

        #     (_,CR[end]) = leftorth!(vecs[1],alg=TensorKit.QRpos())
        # end

        cold = CR[end]

        for loc in 1:length(AL)
            AL[loc] = permute(CR[mod1(loc-1,end)] * permute(A[loc], (1,), (2,3)), (1,2), (3,))
            AL[loc], CR[loc] = leftorth!(AL[loc], alg = QRpos())
            normalize!(CR[loc])
        end

        #update delta
        if domain(cold) == domain(CR[end]) && codomain(cold) == codomain(CR[end])
            delta = norm(cold-CR[end])
        end

        iteration += 1
    end

    delta>tol && @warn "leftorth failed to converge $(delta)"

    AL,CR
end

"
solves C * AR = A * C in-place
"
function uniform_rightorth!(AR,CR,A; tol = Defaults.tolgauge, maxiter = Defaults.maxiter)
    #(CR[end],_) = rightorth!(CR[end], alg = TensorKit.LQpos());
    normalize!(CR[end])

    iteration = 1;
    delta = 2*tol;
    while iteration<maxiter && delta>tol

        # if iteration>10#when qr starts to fail, start using eigs
        #     alg = Arnoldi(krylovdim = 30, tol = max(delta*delta,tol/10),maxiter=maxiter)
        #     #Projection of the current guess onto its largest self consistent eigenvector + isolation of the unitary part

        #     (vals,vecs) = @closure eigsolve(CR[end], 1, :LM,alg) do x
        #         transfer_right(x,A,AR)
        #     end
        #     (CR[end],_) = rightorth!(vecs[1],alg=TensorKit.LQpos())
        # end

        cold = CR[end]
        for loc in length(AR):-1:1
            AR[loc] = A[loc]*CR[loc]

            CR[mod1(loc-1,end)], temp = rightorth!(permute(AR[loc], (1,), (2,3)), alg=LQpos())
            AR[loc] = permute(temp, (1,2), (3,))
            normalize!(CR[mod1(loc-1,end)])
        end

        #update counters and delta
        if domain(cold) == domain(CR[end]) && codomain(cold) == codomain(CR[end])
            delta = norm(cold-CR[end])
        end

        iteration += 1
  end

  delta>tol && @warn "rightorth failed to converge $(delta)"

  return AR, CR
end
