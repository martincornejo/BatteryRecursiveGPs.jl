
function data_preprocessing(X_basis, X_test, Y_test)
    """
    Turns data into appropiate type for Stheno Package and shufles Text points
    """
    ## Shuffling data
    shuffled_index = sample(1:size(X_test, 1), size(X_test, 1); replace=false)
    X_test = X_test[shuffled_index, :]
    Y_test = Y_test[shuffled_index, :]

    return X_basis, X_test, Y_test
end


function gp_rgp0(X_basis, batch_size, θ)
    """
    Initializes gaussian process Parameters
    # Arguments:
        - 'X_basis' = Matrix (N_basis, D) of basis vectors
        - 'θ' = ComponentArray GP hyperparameters
    # Returns:
        - 'p' = ComponentArray RGP cte Parameters
        - 'prior' = ComponentArray of Prior distributions
        - 'posterior' = ComponentArray of Posterior distributions
    """

    kernel = with_lengthscale(SEKernel(), θ.l)

    N_basis = size(X_basis, 1)

    mu_basis = fill(θ.mu, N_basis)
    mu_test = fill(θ.mu, batch_size)
    cov_basis = kernelmatrix(kernel, X_basis, X_basis)

    inv_cov_basis = inv(cov_basis)
    cov_test = zeros(batch_size, batch_size)

    ## Prior of p(g), p(gt). Covariance matrix  p(gt) defined at inference
    # Set as Dict since test matrix can have different size at runtime

    prior = Dict(
        :basis => Dict(:mu => mu_basis, :cov => cov_basis),
        :test => Dict(:mu => mu_test, :cov => cov_test)
    )

    ## Posterior over p(g|y1:t) and  p(gt|y1:t) changes as new data arrives
    # Set as Dict since test matrix can have different size at runtime
    posterior = Dict(
        :basis => Dict(:mu => mu_basis, :cov => cov_basis),
        :test => Dict(:mu => mu_test, :cov => cov_test)
    )

    ## Parameters which are not hypertuned and remain constant
    p = ComponentArray((;
        kernel=kernel,
        inv_cov_basis=inv_cov_basis,
    ))
    return p, prior, posterior
end


function inference(p, prior, posterior, X_batch, X_basis)
    ## All kernelmatrix computation can be precomputed or paralellizable
    """
    Performs predict step of the posterior
    # Arguments:
       - 'p' = ComponentArray RGP Parameters
       - 'prior' = ComponentArray of Prior distributions
       - 'posterior' = ComponentArray of Posterior distributions
       - 'X_batch' = Matrix (N_batch, D) of batch test points
       - 'X_basis' = Matrix (N_basis, D) of basis vectors
    
    # Returns_
        - 'posterior' = Predicted posterior of p(g|y1:t-1)
    """
    H = kernelmatrix(p.kernel, X_batch, X_basis) * p.inv_cov_basis # eq.8

    posterior[:test][:mu] = prior[:test][:mu] + H * (posterior[:basis][:mu] - prior[:basis][:mu]) #eq.6 
    R = kernelmatrix(p.kernel, X_batch) - H * kernelmatrix(p.kernel, X_basis, X_batch) #eq.7 
    println(posterior[:test][:mu])
    posterior[:test][:cov] = R + H * posterior[:basis][:cov] * H' #eq.9
    return posterior, H
end


function update(θ, posterior, H, Y_batch)
    """
    Performs update step of the posterior

    # Arguments:
       - 'p' = ComponentArray RGP Parameters
       - 'prior' =  of Prior distributions
       - 'posterior' = ComponentArray of Posterior distributions
       - 'y_test' = Matrix (N_batch, D) of batch test points
    
    # Returns:
        - 'posterior' = Updated posterior of p(g|y1:t)
        - 'H' = Observe-state matrix
    """
    Gk = posterior[:basis][:cov] * H' * inv(posterior[:test][:cov] + θ.std^2 * I(size(Y_batch, 1))) #eq.12

    posterior[:basis][:mu] = posterior[:basis][:mu] + Gk * (Y_batch - posterior[:test][:mu]) #eq.10
    posterior[:basis][:cov] = posterior[:basis][:cov] - Gk * H * posterior[:basis][:cov] #eq.11

    return posterior
end

function learning_step(p, prior, posterior, X_basis, X_batch, Y_batch, θ)
    """
    Performs one update step
    """

    posterior, H = inference(p, prior, posterior, X_batch, X_basis)
    posterior = update(θ, posterior, H, Y_batch)

    return posterior
end

function rgp(X_basis, X_test, Y_test, θ, batch_size, save_data=false)
    """
    Does rgp without hyperparameter tuning
    # Arguments:
        - 'X_basis' = Matrix (N_basis, D) of basis vectors
        - 'X_test' = Matrix (N_test, D) of test points
        - 'Y_test' = Matrix (N_test, D) of test points
        - 'θ' = ComponentArray of GP hypertuned parameters
        - 'batch_size' = Number of samples per update
        - 'save_data' = Set to true if save posterior of all iterations
    # Returns:
        - 'posterior' = ComponentArray of p(g:t) GP
    """

    X_basis, X_test, Y_test = data_preprocessing(X_basis, X_test, Y_test)
    p, prior, posterior = gp_rgp0(X_basis, batch_size, θ)
    N_test = size(X_test, 1)
    batch_rem = rem(N_test, batch_size)

    for batch_start in 1:batch_size:N_test-batch_rem

        batch_end = batch_start + batch_size - 1

        X_batch = X_test[batch_start:batch_end]
        Y_batch = Y_test[batch_start:batch_end]

        learning_step(p, prior, posterior, X_basis, X_batch, Y_batch, θ)
    end

    # Last iteration if batch_size non divisible by N_test
    batch_start = N_test - batch_rem + 1
    batch_end = N_test
    prior[:test][:mu] = fill(θ.mu, batch_rem)
    X_batch = X_test[batch_start:batch_end]
    Y_batch = Y_test[batch_start:batch_end]
    learning_step(p, prior, posterior, X_basis, X_batch, Y_batch, θ)


    return p, posterior
end


function predict(p, posterior, x_predict)
    ## Paralelizable if needed
    """
    Does a prediction using a posterior at X_predict assuming a prior mean 0
    Note: If N_predict high (Ej.: 3000-5000 points), do by batches
    # Arguments:
        - 'posterior' = Posterior distribition with p(g|y_test)
        - 'x_predict' = Value to predict
    # Returns:
        - 'mu_predict' = Mean prediction
        - 'C_predict' = Covariance prediction
    """

    H = kernelmatrix(p.kernel, x_predict, X_basis) * p.inv_cov_basis
    mu_predict = H * posterior[:basis][:mu]
    R = kernelmatrix(p.kernel, x_predict) - H * kernelmatrix(p.kernel, X_basis, x_predict)
    C_predict = R + H * posterior[:basis][:cov] * H'

    return mu_predict, C_predict

end
