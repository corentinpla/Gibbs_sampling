#%% 
import numpy as np
from scipy.stats import invgamma
from sklearn.linear_model import Lasso
from scipy.linalg import sqrtm
from scipy.special import gamma
#from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numba
from numba import float32, njit

n_iter = 5000
burnout = 1000
n_datasets = 100

RHO = 0.75
T=200
l=0
k=100

a = 1
b = 1
A = 1
B = 1

grid_R2 = np.hstack([[0.001*x for x in range(1,101)],[0.1+0.01*x for x in range(1,81)],[0.9 + 0.001*x for x in range(1,100)]])
grid_q = np.hstack([[0.001*x for x in range(1,101)],[0.1+0.01*x for x in range(1,81)],[0.9 + 0.001*x for x in range(1,100)]])
interlaced_grid = np.array([[(R2,q) for q in grid_q] for R2 in grid_R2])

#@numba.jit(nopython=True)
def X_generator():
    X = (np.random.randn(T,k)).astype(np.float32)
    toeplitz_corr = np.ones((k,k), dtype=np.float32)
    for i in range(k):
        for j in range(i+1,k):
            toeplitz_corr[i,j] = RHO**abs(np.float32(i)-np.float32(j))
            toeplitz_corr[j,i] = RHO**abs(np.float32(i)-np.float32(j))
    L = np.linalg.cholesky(toeplitz_corr)
    for t in range(T):
        X[t] = np.dot(L, X[t])
        X[t] = (X[t] - np.mean(X[t]))/np.std(X[t]) # Standardize the data
    return X

#@numba.jit(nopython=True)
def beta_generator(s):
    beta = np.zeros(k, dtype=np.float32)
    indexes = np.random.choice(k,s,replace=False)
    for i in indexes:
        beta[i] = np.random.randn()
    return beta

#@numba.jit(nopython=True)
def epsilon_generator(R_y, beta, X):
    T = X.shape[0]
    sigma2 = (1/R_y - 1)*np.mean(np.dot(X,beta,)**2)
    return np.sqrt(sigma2)*np.random.randn(T)

#@numba.jit(nopython=True)
def Y_generator(X,beta,epsilon):
    return np.dot(X,beta) + epsilon

#@numba.jit(nopython=True)
def empirical_var(x):
    return np.mean(x**2)-np.mean(x)**2

#@numba.jit(nopython=True)
def get_v_bar(X):
    return np.mean(np.array([empirical_var(X[:,j]) for j in range(X.shape[1])]))

#@numba.jit(nopython=True)
def fct_in_exp(sigma2, k, v_bar, q, R2, beta_tilde):
    matrix_product = np.dot(beta_tilde, beta_tilde) if beta_tilde is not None else 0
    fraction = k*v_bar*q*(1-R2)/R2
    return -fraction*matrix_product/(2*sigma2)

#@numba.jit(nopython=True)
def discrete_prior_grid_R2_q(grid_R2,grid_q,v_bar,sigma2, beta_tilde ,s_z):

    grid_q = grid_q[:,np.newaxis]
    grid_R2 = grid_R2[np.newaxis,:]

    grid = np.exp(fct_in_exp(sigma2,k,v_bar,grid_q,grid_R2,beta_tilde))*(
        grid_q**(s_z+s_z/2+a-1)
        )*((1-grid_q)**(k-s_z+b-1)
        )*(grid_R2**(A-1-s_z/2)
        )*((1-grid_R2)**(s_z/2+B-1))
    return grid/np.sum(grid) # Normalization to have a distribution

#@numba.jit(nopython=True)
def sample_R2_q(v_bar, sigma2, beta_tilde, s_z):
    prior_R2_q = discrete_prior_grid_R2_q(grid_R2,grid_q,v_bar,sigma2,beta_tilde,s_z)
    flattened_prior = prior_R2_q.flatten()
    index = np.random.choice(np.arange(len(flattened_prior)),size=1,p=flattened_prior)
    row_index = index//len(grid_q)
    col_index = index%len(grid_q)
    return interlaced_grid[row_index, col_index][0]

#@numba.jit(nopython=True)
def get_gamma2_from_R2(q,k,v_bar,R2):
    return (1-R2)/(R2*q*k*v_bar)

#@numba.jit(nopython=True)
def sample_sigma2(Y,X,s_z, z_idx, gamma2):
    X_tilde = X[:,z_idx]
    W_tilde = np.dot(X_tilde.T,X_tilde) + np.eye(s_z.item())/gamma2
    beta_tilde_hat = np.dot(np.linalg.inv(W_tilde),np.dot(X_tilde.T,Y))
    beta = 0.5*(np.dot(Y.T,Y) - np.dot(beta_tilde_hat.T,np.dot(W_tilde,beta_tilde_hat)))
    return invgamma.rvs(T/2, beta.item())

#@numba.jit(nopython=True)
def sample_beta_tilde(Y,X,sigma2,s_z, z_idx,gamma2):
    # print("z!=0", np.sum(z!=0))
    if s_z > 0:
        X_tilde = X[:,z_idx]
        W_tilde = (np.dot(X_tilde.T,X_tilde) + np.eye(s_z.item())/gamma2).astype(np.float32)
        mean = np.dot(np.linalg.inv(W_tilde),np.dot(X_tilde.T,Y))
        variance = sigma2 * np.linalg.inv(W_tilde)
        L = np.linalg.cholesky(variance).astype(np.float32)
        beta_tilde = np.array([np.random.randn() for _ in range(len(mean))], dtype=np.float32)
        return mean + np.dot(L, beta_tilde)
    else:
        return None
    
#@numba.jit(nopython=True)
def soft_threshold(rho, alpha):
    """Soft-thresholding operator"""
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0

#@numba.jit(nopython=True)
def get_lasso_beta(X, Y, beta=np.zeros(k, dtype=np.float32), alpha=1, max_iter=1000, tol=1e-4):
    n, d = X.shape

    for _ in range(max_iter):
        beta_old = beta.copy()

        # Update each coefficient independently
        for j in range(d):
            rho = 0.0
            for i in range(n):
                rho += X[i, j] * (Y[i] - np.dot(X[i, :], beta) + beta[j] * X[i, j])

            beta[j] = soft_threshold(rho, alpha) / (X[:, j]**2).sum()

        # Check for convergence
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta

#@numba.jit(nopython=True)
def Gibbs_sampling_step(X, Y, q, z, gamma2):
    for i in range(len(z)):
        # We are going to compute the ratio P(z_i=1)/(P(z_i=0) + P(z_i=1)) = 1 / (1 + ratio) with ration = P(z_i=0) / P(z_i=1)
        z_0 = z.copy()
        z_1 = z.copy()
        z_0[i] = 0
        z_1[i] = 1

        s_0 = np.sum(z_0)
        s_1 = np.sum(z_1)

        X_tilde_0 = X[:,z_0!=0]
        X_tilde_1 = X[:,z_1!=0]

        W_tilde_0 = (np.dot(X_tilde_0.T,X_tilde_0) + np.eye(s_0.item())/gamma2).astype(np.float32)
        W_tilde_1 = (np.dot(X_tilde_1.T,X_tilde_1) + np.eye(s_1.item())/gamma2).astype(np.float32)

        sqrt_det_0 = np.sqrt(np.linalg.det(W_tilde_0))
        sqrt_det_1 = np.sqrt(np.linalg.det(W_tilde_1))

        term_0 = np.dot(Y.T,np.dot(X_tilde_0,np.dot(np.linalg.inv(W_tilde_0).T,np.dot(X_tilde_0.T,Y))))
        term_1 = np.dot(Y.T,np.dot(X_tilde_1,np.dot(np.linalg.inv(W_tilde_1).T,np.dot(X_tilde_1.T,Y))))
        
        matrix_prods_ratio = (np.dot(Y.T,Y) - term_1)/(
            np.dot(Y.T,Y) - term_0)
        denominator = 1 + q / ((1-q)*np.sqrt(gamma2)) * ((sqrt_det_0)/(sqrt_det_1)) * matrix_prods_ratio**(-T/2)
        ratio = 1-1/denominator
        # print(ratio)
        if np.random.uniform() < ratio:
            z[i] = 1
        else:
            z[i] = 0
    return z

#@numba.jit(nopython=True)
def get_beta_from_z_and_beta_tilde(z,beta_tilde):
    if beta_tilde is not None:
        indices = np.nonzero(z)
        beta = np.zeros_like(z, dtype=np.float32)
        beta[indices] = beta_tilde
        return beta
    else:
        # print("beta_tilde est vide")
        # raise RuntimeError
        return np.zeros(k, dtype=np.float32)

#@numba.jit(nopython=True)
def all_iterations(n_datasets, n_iter, burnout, s_values = [5,10,100], R_y_values=[0.02,0.25,0.5]):
    all_median_results = []
    # all_results = []
    for idx_dataset in range(n_datasets):
        median_res = []
        X = X_generator()
        v_bar = get_v_bar(X)
        for s in s_values:
            for R_y in R_y_values:
                # Dataset generation
                beta = beta_generator(s)
                epsilon = epsilon_generator(R_y, beta, X)
                Y = Y_generator(X, beta, epsilon).astype(np.float32)
                # Initialization
                beta = get_lasso_beta(X, Y) # Lasso to retrieve beta, z and sigma2
                z = (beta!=0).astype(np.int64)
                s_z = np.sum(z)
                beta_tilde = beta[z!=0]
                epsilon = Y - np.dot(X,beta)
                sigma2 = np.var(epsilon)
                R2, q = sample_R2_q(v_bar, sigma2, beta_tilde, s_z)

                # List initialization
                R2_res = np.zeros(n_iter+1)
                R2_res[0] = R2
                q_res = np.zeros(n_iter+1)
                q_res[0] = q
                z_res = np.zeros((n_iter+1,k))
                z_res[0] = z
                beta_res = np.zeros((n_iter+1,k))
                beta_res[0] = beta
                sigma2_res = np.zeros(n_iter+1)
                sigma2_res[0] = sigma2

                for idx_iter in range(n_iter):
                    print(str(idx_iter)+"/"+str(n_iter))
                    # block 1
                    R2, q = sample_R2_q(v_bar, sigma2, beta_tilde, s_z)
                    gamma2 = get_gamma2_from_R2(q, k, v_bar, R2)
                    # block 2
                    z = Gibbs_sampling_step(X, Y, q, z, gamma2)
                    s_z = np.sum(z)
                    z_idx = (z!=0)
                    sigma2 = sample_sigma2(Y, X, s_z, z_idx, gamma2)
                    beta_tilde = sample_beta_tilde(Y, X, sigma2, s_z, z_idx, gamma2)
                    beta = get_beta_from_z_and_beta_tilde(z, beta_tilde)

                    R2_res[idx_iter+1] = R2
                    q_res[idx_iter+1] = q
                    z_res[idx_iter+1] = np.array(z)
                    beta_res[idx_iter+1] = np.array(beta)
                    sigma2_res[idx_iter+1] = sigma2

                R2_res = R2_res[burnout:].tolist()
                q_res_list = q_res[burnout:].tolist()
                z_res = z_res[burnout:].tolist()
                beta_res = beta_res[burnout:].tolist()
                sigma2_res = sigma2_res[burnout:].tolist()
                dataset_results = [X.tolist(), R2_res, q_res_list, z_res, beta_res, sigma2_res]

                median_q = np.median(q_res)
                median_res.append(median_q.item())
                save_results(idx_dataset, s, R_y, dataset_results)

        all_median_results.append(median_res)
        # dataset_results=[R2_res, q_res, z_res, beta_res, sigma2_res]
        # all_results.append(dataset_results)

    # return all_results
    return all_median_results

#@numba.jit(nopython=True)
def save_results(idx_dataset, s, R_y, dataset_results):
    [X, R2_res, q_res, z_res, beta_res, sigma2_res] = dataset_results
    results = {"X" : X, "R2" : R2_res, "q" : q_res, "z" : z_res, "beta" : beta_res, "sigma2" : sigma2_res }
    file_name = "/Users/corentinpla/Documents/results_stats" + str(idx_dataset) + "_s" + str(s) + "_Ry" + str(int(R_y*100)) + ".json"
    with open(file_name, 'w') as json_file:
        json.dump(results, json_file)

#@numba.jit(nopython=True)
def plot_median_q(results, save_results=True, display=True):
    transposed_median = list(map(list, zip(*results)))
    for i, values_for_dataset in enumerate(transposed_median):
        plt.figure(figsize=(6, 4))
        plt.hist(values_for_dataset, bins=20, alpha=0.5)
        plt.title(f'Histogram of median q for (s,R_y) = {i + 1}')
        if save_results:
            plt.savefig(f'/Users/corentinpla/Documents/results_stats/histograms/s_R_y_{i + 1}_histogram.png')
        if display:
            plt.show()

all_median_results = all_iterations(100,n_iter,burnout, [5], [0.02])
# plot_median_q(all_median_results)


#final assigment
np.random.seed(42)
production_index = np.random.rand(100)
features = np.random.rand(100, 10)


# Function to predict using the selected variables
def make_predictions(features, coefficients):
    return np.dot(features, coefficients)

# Function for mean imputation
def mean_imputation(data):
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(data)

# Plotting function for the posterior density of q
def plot_posterior_density(q_posterior):
    plt.figure(figsize=(8, 6))
    plt.hist(q_posterior, bins=20, density=True, alpha=0.5, color='blue')
    plt.title('Posterior Density of $q$')
    plt.xlabel('$q$')
    plt.ylabel('Density')
    plt.show()

# Plotting function for prediction comparison
def plot_prediction_comparison(actual, predicted_all, predicted_subset):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(predicted_all, label='All Variables', linestyle='--', marker='o')
    plt.plot(predicted_subset, label='Selected Variables', linestyle='--', marker='o')
    plt.title('Prediction Comparison')
    plt.xlabel('Period')
    plt.ylabel('Production Index')
    plt.legend()
    plt.show()

# Perform Gibbs sampling
num_iterations = 100000
q_posterior = gibbs_sampler(features, num_iterations)

# Plot the posterior density of q
plot_posterior_density(q_posterior)

# Select variables based on q
selected_variables = features[:, q_posterior.mean(axis=0) > 0.55]

# Perform mean imputation
features_imputed = mean_imputation(selected_variables)

# Perform Lasso regression for the entire dataset
coefficients_all = perform_lasso(features_imputed[:-10], production_index[:-10], alpha=0.1)

# Perform Lasso regression for the subset
coefficients_subset = perform_lasso(features_imputed[:-10], production_index[:-10], alpha=0.1)

# Make predictions
predictions_all = make_predictions(features_imputed[-10:], coefficients_all)
predictions_subset = make_predictions(features_imputed[-10:], coefficients_subset)

# Plot prediction comparison
plot_prediction_comparison(production_index[-10:], predictions_all, predictions_subset)
