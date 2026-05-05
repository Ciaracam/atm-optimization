import numpy as np
from data_generation import generate_data
from model import solve_atm_placement, solve_coverage_model, calculate_solution_metrics


# Clean labels
def clean_label(text):
    return text.replace("_", " ").title()


# Run simulation
def run_simulations(K=3, coverage_radius=0.08):

    # Scenarios
    scenarios = [
        "baseline",
        "urban_demand_increase",
        "suburban_demand_increase",
        "demand_variability"
    ]

    # Model types
    models = [
        "weighted_distance",
        "coverage_model"
    ]

    results = {}

    # Loop through each scenario
    for scenario in scenarios:
        results[scenario] = {}

        members, atm_sites, weights, labels = generate_data(scenario=scenario)

        # Loop through each model
        for model_name in models:

            
            # Weighted Distance Model
            
            if model_name == "weighted_distance":

                x, y, obj, weighted_distance, max_distance = solve_atm_placement(
                    members,
                    atm_sites,
                    weights,
                    K
                )

                selected_atms = np.where(x > 0.5)[0]

                metrics = calculate_solution_metrics(
                    members,
                    atm_sites,
                    weights,
                    y
                )

                # Save results
                results[scenario][model_name] = {
                    "members": members,
                    "atm_sites": atm_sites,
                    "weights": weights,
                    "labels": labels,
                    "x": x,
                    "y": y,
                    "objective_value": obj,
                    "weighted_distance": weighted_distance,
                    "max_distance": max_distance,
                    "selected_atms": selected_atms,
                    "avg_distance": metrics["avg_distance"],
                    "metric_max_distance": metrics["max_distance"],
                    "metric_weighted_distance": metrics["weighted_distance"]
                }

                # Print results
                print(f"Scenario: {clean_label(scenario)} | Model: Demand-Weighted Distance")
                print(f"Objective Value: {obj:.3f}")
                print(f"Weighted Distance: {weighted_distance:.3f}")
                print(f"Average Distance: {metrics['avg_distance']:.3f}")
                print(f"Max Distance: {metrics['max_distance']:.3f}")
                print(f"Selected ATMs: {selected_atms}")
                print("-" * 50)

            
            # Coverage Model
            
            elif model_name == "coverage_model":

                x, covered, obj = solve_coverage_model(
                    members,
                    atm_sites,
                    weights,
                    K,
                    coverage_radius=coverage_radius
                )

                selected_atms = np.where(x > 0.5)[0]
                coverage_rate = np.mean(covered > 0.5)

                # Save results
                results[scenario][model_name] = {
                    "members": members,
                    "atm_sites": atm_sites,
                    "weights": weights,
                    "labels": labels,
                    "x": x,
                    "covered": covered,
                    "objective_value": obj,
                    "selected_atms": selected_atms,
                    "coverage_rate": coverage_rate,
                    "weighted_coverage": obj
                }

                # Print results
                print(f"Scenario: {clean_label(scenario)} | Model: Coverage Maximization Model")
                print(f"Weighted Coverage: {obj:.3f}")
                print(f"Coverage Rate: {coverage_rate:.3f}")
                print(f"Selected ATMs: {selected_atms}")
                print("-" * 50)

    return results
