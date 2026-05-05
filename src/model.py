import numpy as np
import cvxpy as cp


# Compute pairwise distances between members and ATM sites
def compute_distance_matrix(members, atm_sites):
    n_members = members.shape[0]
    n_sites = atm_sites.shape[0]

    D = np.zeros((n_members, n_sites))

    # Loop through each member and ATM
    for i in range(n_members):
        for j in range(n_sites):
            D[i, j] = np.linalg.norm(members[i] - atm_sites[j])

    return D


# Demand-weighted distance minimization model
def solve_atm_placement(members, atm_sites, weights, K, lambda_fairness=0.0):
    n_members = members.shape[0]
    n_sites = atm_sites.shape[0]

    # Distance matrix
    D = compute_distance_matrix(members, atm_sites)

    # Decision variables
    x = cp.Variable(n_sites, boolean=True) # ATM selection
    y = cp.Variable((n_members, n_sites), boolean=True) # member assignment
    z = cp.Variable() # max distance (fairness)

    # Weighted travel distance
    weighted_distance = cp.sum(
        cp.multiply(weights[:, None], cp.multiply(D, y))
    )

    # minimize weighted distance + fairness penalty
    objective = cp.Minimize(
        weighted_distance + lambda_fairness * z
    )

    constraints = []

    # Each member assigned to exactly 1 ATM
    constraints += [
        cp.sum(y[i, :]) == 1 for i in range(n_members)
    ]

    # Assignment only allowed if ATM is selected
    constraints += [
        y[i, j] <= x[j] for i in range(n_members) for j in range(n_sites)
    ]

    # Limit number of ATMs
    constraints += [
        cp.sum(x) == K
    ]

    # Define max distance for fairness
    constraints += [
        cp.sum(cp.multiply(D[i, :], y[i, :])) <= z
        for i in range(n_members)
    ]

    # Solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCIPY)

    return x.value, y.value, problem.value, weighted_distance.value, z.value


# Coverage model: maximize members within radius
def solve_coverage_model(members, atm_sites, weights, K, coverage_radius=0.08):
    n_members = members.shape[0]
    n_sites = atm_sites.shape[0]

    # Distance matrix
    D = compute_distance_matrix(members, atm_sites)

    # Coverage matrix
    coverage_matrix = (D <= coverage_radius).astype(int)

    # Decision variables
    x = cp.Variable(n_sites, boolean=True)
    y = cp.Variable((n_members, n_sites), boolean=True)

    # maximize weighted covered assignments
    objective = cp.Maximize(
        cp.sum(cp.multiply(weights[:, None], y))
    )

    constraints = []

    # Limit number of ATMs
    constraints += [
        cp.sum(x) == K
    ]

    # Each member can only be covered once
    constraints += [
        cp.sum(y[i, :]) <= 1 for i in range(n_members)
    ]

    # Assignment only if ATM is selected and within radius
    constraints += [
        y[i, j] <= x[j] * coverage_matrix[i, j]
        for i in range(n_members)
        for j in range(n_sites)
    ]

    # Solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCIPY)

    covered = np.sum(y.value, axis=1)

    return x.value, covered, problem.value


# Calculate summary metrics for weighted model
def calculate_solution_metrics(members, atm_sites, weights, y):
    D = compute_distance_matrix(members, atm_sites)

    # Assigned distances per member
    assigned_distances = np.sum(D * y, axis=1)

    # Metrics
    avg_distance = np.mean(assigned_distances)
    max_distance = np.max(assigned_distances)
    weighted_distance = np.sum(weights * assigned_distances)

    return {
        "avg_distance": avg_distance,
        "max_distance": max_distance,
        "weighted_distance": weighted_distance
    }
