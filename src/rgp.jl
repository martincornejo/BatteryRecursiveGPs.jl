
function data_preprocessing(X_basis, X_test, Y_test)
    """
    Turns data into appropiate type for Stheno Package and shufles Text points
    """
    ## Shuffling data
    shuffled_index = sample(1:size(X_test, 1), size(X_test, 1); replace=false)
    X_test = X_test[shuffled_index, :]
    Y_test = Y_test[shuffled_index, :]

    ## Stheno Notation
    X_basis = RowVecs(X_basis)
    X_test = RowVecs(X_test)
    Y_test = RowVecs(Y_test)
    return X_basis, X_test, Y_test
end


function gp_rgp0(X_basis, batch_size, θ)
    """
    Initializes gaussian process Parameters
    # Arguments:
        - 'X_basis' = Matrix (N_basis, D) of basis vectors
        - 'θ' = ComponentArray GP hyperparameters
    # Returns:
        - 'p' = ComponentArray RGP Parameters
        - 'prior' = ComponentArray of Prior distributions
        - 'posterior' = ComponentArray of Posterior distributions
    """

    k = with_lengthscale(SEKernel(), θ.l)

    N_basis = size(X_basis, 1)
    N_test = batch_size

    mu_basis = fill(θ.mu, N_basis)
    mu_test = fill(θ.mu, N_test)
    cov_basis = kernelmatrix(k, X_basis, X_basis)

    cov_test = zeros(N_test, N_test)
    H = zeros(N_test, N_basis)
    inv_cov_basis = inv(cov_basis)

    ## Prior of p(g), p(gt). Covariance matrix  p(gt) defined at inference
    prior = ComponentArray((;
        basis=(;
            mu=mu_basis,
            cov=cov_basis,
        ),
        test=(;
            mu=mu_test,
            cov=cov_test,
        ),
    ))

    ## Posterior over p(g|y1:t) and  p(gt|y1:t) changes as new data arrives 
    posterior = ComponentArray((;
        basis=(;
            mu=mu_basis,
            cov=cov_basis
        ),
        test=(;
            mu=mu_test,
            cov=cov_test
        ),
    ))

    ## Parameters which are not hypertuned. H Initializes as nothing since depend X_test
    p = ComponentArray((;
        k=k,
        inv_cov_basis=inv_cov_basis,
        H=H,
        Q=θ.std^2 * I(N_test)
    ))
    return p, prior, posterior
end


function inference(p, prior, posterior, x_test, X_basis)
    # PARALELIZABLE Matrix computation of kernelmatrix computations
    """
    Performs predict step of the posterior
    # Arguments:
       - 'p' = ComponentArray RGP Parameters
       - 'prior' = ComponentArray of Prior distributions
       - 'posterior' = ComponentArray of Posterior distributions
       - 'x_test' = Matrix (N_test, D) of test points
       - 'X_basis' = Matrix (N_basis, D) of basis vectors
    
    # Returns_
        - 'posterior' = Predicted posterior of p(gt|y1:t-1)
    """
    p.H = kernelmatrix(p.k, x_test, X_basis) * p.inv_cov_basis # eq.8 
    posterior.test.mu = prior.test.mu + p.H * (posterior.basis.mu - prior.basis.mu) #eq.6 
    R = kernelmatrix(p.k, x_test) - p.H * kernelmatrix(p.k, X_basis, x_test) #eq.7 
    posterior.test.cov = R + p.H * posterior.basis.cov * p.H' #eq.9
    return posterior
end


function update(p, posterior, y_test)
    """
    Performs update step of the posterior

    # Arguments:
       - 'p' = ComponentArray RGP Parameters
       - 'prior' = ComponentArray of Prior distributions
       - 'posterior' = ComponentArray of Posterior distributions
       - 'y_test' = Matrix (N_test, D) of test points outputs
    
    # Returns_
        - 'posterior' = Updated posterior of p(g|y1:t)
    """
    Gk = posterior.basis.cov * p.H' * inv(posterior.test.cov + p.Q) #eq.12
    posterior.basis.mu = posterior.basis.mu + Gk * (y_test - posterior.test.mu) #eq.10

    posterior.basis.cov = posterior.basis.cov - Gk * p.H * posterior.basis.cov #eq.11

    return posterior
end


function rgp(X_basis, X_test, Y_test, θ, batch_size)
    """
    Does rgp without hyperparameter tuning
    # Arguments:
        - 'X_basis' = Matrix (N_basis, D) of basis vectors
        - 'X_test' = Matrix (n_batches * N_test, D) of test points
        - 'Y_test' = Matrix (n_batches * N_test, D) of test points
        - 'θ' = ComponentArray of GP hypertuned parameters
    # Returns:
        - 'posterior' = ComponentArray of p(g:t) GP
    """

    X_basis, X_test, Y_test = data_preprocessing(X_basis, X_test, Y_test)
    p, prior, posterior = gp_rgp0(X_basis, batch_size, θ)

    for (x_test, y_test) in zip(X_test, Y_test)
        inference(p, prior, posterior, x_test, X_basis)
        update(p, posterior, y_test)
    end

    return p, posterior
end


function predict(p, posterior, x_predict)
    """
    Does a prediction using a posterior at X_predict
    # Arguments:
        - 'posterior' = Posterior distribition of p(g|y_test)
        - 'x_predict' = Value to predict
    # Returns:
        - 'mu_predict' = ComponentArray of predicted mean and cov
        - 'C_predict' = Covariance prediction
    """
    x_predict = RowVecs(x_predict)
    H = kernelmatrix(p.k, x_predict, X_basis) * p.inv_cov_basis
    mu_predict = H * posterior.basis.mu
    R = kernelmatrix(p.k, x_predict) - H * kernelmatrix(p.k, X_basis, x_predict)
    C_predict = R + H * posterior.basis.cov * H'

    return mu_predict, C_predict

end
