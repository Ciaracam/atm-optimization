import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from simulation import run_simulations
from model import compute_distance_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# Clean labels for titles and print output
def clean_label(text):
    return text.replace("_", " ").title()


# Figure 1: Before optimization
def plot_member_distribution(result, scenario_name, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    members = result["members"]
    atm_sites = result["atm_sites"]
    labels = result["labels"]

    plt.figure(figsize=(10, 8))

    # Plot member clusters before assignment
    for group in np.unique(labels):
        group_points = members[labels == group]

        plt.scatter(
            group_points[:, 1],
            group_points[:, 0],
            s=25,
            alpha=0.7,
            label=group
        )

    # Plot candidate ATM locations
    plt.scatter(
        atm_sites[:, 1],
        atm_sites[:, 0],
        marker="x",
        s=90,
        label="Candidate ATM Locations"
    )

    plt.title(f"Member Distribution and Candidate ATM Locations - {clean_label(scenario_name)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{scenario_name}_member_distribution.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


# Plot weighted distance assignments
def plot_weighted_distance_scenario(result, scenario_name, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    members = result["members"]
    atm_sites = result["atm_sites"]
    labels = result["labels"]
    selected_atms = result["selected_atms"]
    y = result["y"]

    plt.figure(figsize=(10, 8))

    # Draw assignment lines
    for i in range(len(members)):
        assigned_j = np.argmax(y[i])

        plt.plot(
            [members[i, 1], atm_sites[assigned_j, 1]],
            [members[i, 0], atm_sites[assigned_j, 0]],
            linewidth=0.5,
            alpha=0.18
        )

    # Plot member clusters
    for group in np.unique(labels):
        group_points = members[labels == group]

        plt.scatter(
            group_points[:, 1],
            group_points[:, 0],
            s=25,
            alpha=0.7,
            label=group
        )

    # Plot candidate ATM locations
    plt.scatter(
        atm_sites[:, 1],
        atm_sites[:, 0],
        marker="x",
        s=90,
        label="Candidate ATM Locations"
    )

    # Highlight selected ATM locations
    plt.scatter(
        atm_sites[selected_atms, 1],
        atm_sites[selected_atms, 0],
        marker="*",
        s=350,
        label="Selected ATM Locations"
    )

    plt.title(f"Member Assignment to Selected ATM Locations (Nearest ATM Assignment) - {clean_label(scenario_name)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{scenario_name}_weighted_distance_map.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


# Plot coverage model results
def plot_coverage_scenario(result, scenario_name, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    members = result["members"]
    atm_sites = result["atm_sites"]
    selected_atms = result["selected_atms"]
    covered = result["covered"]

    covered_mask = covered > 0.5
    uncovered_mask = covered <= 0.5

    plt.figure(figsize=(10, 8))

    # Plot members within service radius
    plt.scatter(
        members[covered_mask, 1],
        members[covered_mask, 0],
        s=25,
        alpha=0.7,
        label="Assigned Members (distance ≤ 0.08)"
    )

    # Plot members outside service radius
    plt.scatter(
        members[uncovered_mask, 1],
        members[uncovered_mask, 0],
        s=25,
        alpha=0.7,
        label="Unassigned Members (distance > 0.08)"
    )

    # Plot candidate ATM locations
    plt.scatter(
        atm_sites[:, 1],
        atm_sites[:, 0],
        marker="x",
        s=90,
        label="Candidate ATM Locations"
    )

    # Highlight selected ATM locations
    plt.scatter(
        atm_sites[selected_atms, 1],
        atm_sites[selected_atms, 0],
        marker="*",
        s=350,
        label="Selected ATM Locations"
    )


    # Draw radius
    for atm in selected_atms:
        circle = patches.Circle(
            (atm_sites[atm, 1], atm_sites[atm, 0]),  # (lon, lat)
            0.08,
            fill=False,
            linestyle='--',
            linewidth=1.5,
            alpha=0.3
        )
        plt.gca().add_patch(circle)

    

    

    plt.title(f"Coverage Model (Assignment Based on Service Radius r = 0.08) - {clean_label(scenario_name)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.plot([], [], linestyle="--", color="gray", label="Service Radius (r = 0.08)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{scenario_name}_coverage_model_map.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


# Calculate regional metrics for weighted distance model
def calculate_group_metrics(result):
    members = result["members"]
    atm_sites = result["atm_sites"]
    weights = result["weights"]
    labels = result["labels"]
    y = result["y"]

    D = compute_distance_matrix(members, atm_sites)

    assigned_distances = np.sum(D * y, axis=1)
    weighted_burden = weights * assigned_distances

    groups = np.unique(labels)

    avg_distance_by_group = []
    weighted_burden_by_group = []

    for group in groups:
        mask = labels == group

        avg_distance_by_group.append(
            np.mean(assigned_distances[mask])
        )

        weighted_burden_by_group.append(
            np.sum(weighted_burden[mask])
        )

    return groups, avg_distance_by_group, weighted_burden_by_group


# Compare baseline and suburban demand shift by region
def plot_regional_tradeoff(results, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    width = 0.35

    groups, baseline_avg, baseline_burden = calculate_group_metrics(
        results["baseline"]["weighted_distance"]
    )

    _, suburban_avg, suburban_burden = calculate_group_metrics(
        results["suburban_demand_increase"]["weighted_distance"]
    )

    x = np.arange(len(groups))

    plt.figure(figsize=(18, 6))

    # Average distance comparison
    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, baseline_avg, width, label="Baseline")
    plt.bar(x + width / 2, suburban_avg, width, label="Suburban Demand Increase")
    plt.title("Average Travel Distance by Region")
    plt.xlabel("Region")
    plt.ylabel("Average Travel Distance")
    plt.xticks(x, groups, rotation=25, ha="right")
    plt.legend()
    plt.grid(axis="y")

    # Weighted burden comparison
    plt.subplot(1, 2, 2)
    plt.bar(x - width / 2, baseline_burden, width, label="Baseline")
    plt.bar(x + width / 2, suburban_burden, width, label="Suburban Demand Increase")
    plt.title("Demand-Weighted Travel Burden by Region")
    plt.xlabel("Region")
    plt.ylabel("Demand-Weighted Travel Burden")
    plt.xticks(x, groups, rotation=25, ha="right")
    plt.legend()
    plt.grid(axis="y")

    plt.suptitle("Regional Access and Demand Burden Comparison", fontsize=16)
    plt.tight_layout()

    filename = os.path.join(output_dir, "regional_tradeoff_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


# Plot ATM location selection frequency for weighted distance model
def plot_atm_selection_frequency(results, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    atm_sites = results["baseline"]["weighted_distance"]["atm_sites"]
    selection_counts = np.zeros(len(atm_sites))

    scenarios = list(results.keys())

    for scenario in scenarios:
        selected_atms = results[scenario]["weighted_distance"]["selected_atms"]

        for atm in selected_atms:
            selection_counts[atm] += 1

    selected_mask = selection_counts > 0

    plt.figure(figsize=(10, 8))

    # Plot candidate ATM locations
    plt.scatter(
        atm_sites[:, 1],
        atm_sites[:, 0],
        marker="x",
        s=90,
        label="Candidate ATM Locations"
    )

    # Plot selection frequency
    plt.scatter(
        atm_sites[selected_mask, 1],
        atm_sites[selected_mask, 0],
        s=selection_counts[selected_mask] * 160,
        alpha=0.65,
        label="Selection Frequency Across Scenarios"
    )

    # Add frequency labels
    for idx in np.where(selected_mask)[0]:
        plt.text(
            atm_sites[idx, 1],
            atm_sites[idx, 0],
            f"{int(selection_counts[idx])}/4",
            fontsize=11,
            ha="center",
            va="center"
        )

    plt.title("Consistent ATM Location Selections Across Scenarios", pad=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(output_dir, "atm_selection_frequency.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


# Compare K = 3 and K = 2
def plot_k_sensitivity(results_k3, results_k2, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    scenarios = list(results_k3.keys())

    k3_values = [
        results_k3[scenario]["weighted_distance"]["objective_value"]
        for scenario in scenarios
    ]

    k2_values = [
        results_k2[scenario]["weighted_distance"]["objective_value"]
        for scenario in scenarios
    ]

    x = np.arange(len(scenarios))
    width = 0.35

    scenario_labels = [clean_label(scenario) for scenario in scenarios]

    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, k3_values, width, label="K = 3 ATM Locations")
    plt.bar(x + width / 2, k2_values, width, label="K = 2 ATM Locations")

    plt.title("Impact of Reducing ATM Availability")
    plt.xlabel("Scenario")
    plt.ylabel("Weighted Travel Distance")
    plt.xticks(x, scenario_labels, rotation=25, ha="right")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    filename = os.path.join(output_dir, "k_sensitivity_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


if __name__ == "__main__":
    results_k3 = run_simulations(K=3, coverage_radius=0.08)
    results_k2 = run_simulations(K=2, coverage_radius=0.08)

    # Figure 1: Member distribution before optimization
    plot_member_distribution(
        results_k3["baseline"]["weighted_distance"],
        "baseline"
    )

    # Figure 2: Baseline weighted distance assignment model
    plot_weighted_distance_scenario(
        results_k3["baseline"]["weighted_distance"],
        "baseline"
    )

    # Figure 3: Baseline service radius coverage
    plot_coverage_scenario(
        results_k3["baseline"]["coverage_model"],
        "baseline"
    )

    # Figure 4: Suburban demand shift weighted distance model
    plot_weighted_distance_scenario(
        results_k3["suburban_demand_increase"]["weighted_distance"],
        "suburban_demand_increase"
    )

    # Figure 5: Regional access and burden comparison
    plot_regional_tradeoff(results_k3)

    # Figure 6: K sensitivity comparison
    plot_k_sensitivity(results_k3, results_k2)

    # Figure 7: ATM location selection frequency
    plot_atm_selection_frequency(results_k3)

    print("Done.")
