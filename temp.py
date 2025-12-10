import cvxpy as cp
import numpy as np

# 1. Setup Data (Example with 1693 stocks)
# In reality, 'mu' is your expected returns, 'Sigma' is your covariance matrix
n_assets = 1693
k_max = 20  # Max stocks allowed

# Generating a dummy covariance matrix for the example (Use Numba/PyCUDA here if needed!)
np.random.seed(42)
returns = np.random.randn(n_assets, 1000)
Sigma = np.cov(returns)

# 2. Define Variables
w = cp.Variable(n_assets)          # Continuous weights
z = cp.Variable(n_assets, boolean=True) # Binary variables (0 or 1)

# 3. Define Objective (Minimize Variance)
# Note: You can also maximize Return - Lambda * Risk
risk = cp.quad_form(w, Sigma)
objective = cp.Minimize(risk)

# 4. Define Constraints
constraints = [
    cp.sum(w) == 1,         # Fully invested
    w >= 0,                 # Long only
    cp.sum(z) <= k_max,     # Max 20 stocks
    w <= z                  # Big-M constraint: if z=0, w must be 0
]

# 5. Define and Solve Problem
prob = cp.Problem(objective, constraints)

print(f"Solving MIQP for {n_assets} assets...")

# SOLVER CHOICE IS CRITICAL HERE:
# For 1693 integer variables, you need a strong solver.
# Options:
# 1. cp.GUROBI (Best, Commercial, requires license)
# 2. cp.CPLEX (Excellent, Commercial, requires license)
# 3. cp.SCIP (Good, Open Source, requires 'pyscipopt' installed)
# 4. cp.CBC (Good, Open Source, requires 'cylp' installed)

try:
    # Trying with SCIP or a generic solver if available
    prob.solve(solver=cp.SCIP, verbose=True) 
except:
    print("SCIP not found. Trying default (may be slow or fail)...")
    prob.solve(verbose=True)

# 6. Output Results
print("\nOptimization Status:", prob.status)
print("Optimal Portfolio Risk (Variance):", prob.value)

# Extract selected stocks
selected_indices = np.where(z.value > 0.5)[0]
weights = w.value[selected_indices]

print(f"\nNumber of stocks selected: {len(selected_indices)}")
print("Selected Indices:", selected_indices)
print("Weights:", np.round(weights, 4))