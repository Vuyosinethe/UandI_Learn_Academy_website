import numpy as np
from scipy.optimize import minimize

def augmented_lagrangian(x, y, z, lambda1, lambda2, rho):
    # Objective function
    f = x * y + x * z + y * z
    
    # Constraints
    g1 = x**2 + y**2 + z**2 - 2
    g2 = z - 1
    L = f + lambda1 * g1 + lambda2 * g2 + (rho / 2) * (g1**2 + g2**2)
    return L

def objective(X, lambda1, lambda2, rho):
    x, y, z = X
    return augmented_lagrangian(x, y, z, lambda1, lambda2, rho)

def augmented_lagrangian_method(initial_guess, lambda1=0, lambda2=0, rho=10, tol=1e-6, max_iter=100):
    X = np.array(initial_guess)
    
    for _ in range(max_iter):
        res = minimize(objective, X, args=(lambda1, lambda2, rho), method='BFGS')
        X = res.x

        g1 = X[0]**2 + X[1]**2 + X[2]**2 - 2
        g2 = X[2] - 1
        lambda1 += rho * g1
        lambda2 += rho * g2
        
        # Convergence check
        if abs(g1) < tol and abs(g2) < tol:
            break
    
    return X, lambda1, lambda2

if __name__ == "__main__":
    initial_guess = [1.0, 1.0, 1.0]
    solution, lambda1, lambda2 = augmented_lagrangian_method(initial_guess)

    print(f"Optimal solution: x = {solution[0]}, y = {solution[1]}, z = {solution[2]}")
    print(f"Lagrange multipliers: lambda1 = {lambda1}, lambda2 = {lambda2}")
