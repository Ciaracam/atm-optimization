import numpy as np


def generate_members(seed=42):
    np.random.seed(seed)

    # Real Maryland/DC-area anchor points
    # Format: [latitude, longitude]

    anchors = {
        "DC Metro/Prince Georges": [38.9072, -77.0369],
        "Silver Spring": [38.9907, -77.0261],
        "Bowie": [39.0068, -76.7791],
        "Annapolis": [38.9784, -76.4922],
        "Waldorf": [38.6265, -76.9105],
    }

    # Simulate members around each real anchor point
    dc_pg = np.random.normal(loc=anchors["DC Metro/Prince Georges"], scale=[0.025, 0.025], size=(70, 2))
    silver_spring = np.random.normal(loc=anchors["Silver Spring"], scale=[0.025, 0.025], size=(50, 2))
    bowie = np.random.normal(loc=anchors["Bowie"], scale=[0.035, 0.035], size=(40, 2))
    annapolis = np.random.normal(loc=anchors["Annapolis"], scale=[0.04, 0.04], size=(30, 2))
    waldorf = np.random.normal(loc=anchors["Waldorf"], scale=[0.05, 0.05], size=(25, 2))

    members = np.vstack([dc_pg, silver_spring, bowie, annapolis, waldorf])

    labels = (
        ["DC Metro/Prince George's"] * len(dc_pg)
        + ["Silver Spring"] * len(silver_spring)
        + ["Bowie"] * len(bowie)
        + ["Annapolis"] * len(annapolis)
        + ["Waldorf"] * len(waldorf))

    return members, np.array(labels)


def generate_atm_sites(seed=42):
    np.random.seed(seed + 1)

    # Candidate ATM sites based on realistic Maryland/DC-area locations
    atm_sites = np.array([
        [38.9072, -77.0369],  # DC
        [38.9560, -76.9455],  # Hyattsville
        [38.7849, -76.8721],  # Clinton
        [38.9784, -76.4922],  # Annapolis
        [39.0068, -76.7791],  # Bowie
        [38.6265, -76.9105],  # Waldorf
        [39.0458, -76.6413],  # Crofton
        [38.9897, -76.9378],  # College Park
        [38.8462, -76.9239],  # Suitland
        [38.9421, -76.7303],  # Largo
        [38.9907, -77.0261],  # Silver Spring
        [39.0839, -77.1528],  # Rockville
        [39.1434, -77.2014],  # Gaithersburg
        [39.2904, -76.6122],  # Baltimore
        [39.4143, -77.4105],  # Frederick
    ])

    return atm_sites


def generate_demand_weights(labels, scenario="baseline", seed=42):
    np.random.seed(seed + 2)

    weights = np.ones(len(labels))

    # Baseline demand assumptions
    weights[labels == "DC Metro/Prince Georges"] = np.random.uniform(1.8, 2.6, sum(labels == "DC Metro / Prince George's"))
    weights[labels == "Silver Spring"] = np.random.uniform(1.6, 2.4, sum(labels == "Silver Spring"))
    weights[labels == "Bowie"] = np.random.uniform(1.2, 2.0, sum(labels == "Bowie"))
    weights[labels == "Annapolis"] = np.random.uniform(1.0, 1.7, sum(labels == "Annapolis"))
    weights[labels == "Waldorf"] = np.random.uniform(0.8, 1.5, sum(labels == "Waldorf"))

    # Simulation scenarios
    if scenario == "urban_demand_increase":
        weights[labels == "DC Metro/Prince Georges"] *= 3.0

    elif scenario == "suburban_demand_increase":
        weights[labels == "Bowie"] *= 3.0
        weights[labels == "Waldorf"] *= 3.0

    elif scenario == "demand_variability":
        weights *= np.random.uniform(0.7, 1.6, len(weights))

    return weights


def generate_data(scenario="baseline", seed=42):
    members, labels = generate_members(seed=seed)
    atm_sites = generate_atm_sites(seed=seed)
    weights = generate_demand_weights(labels, scenario=scenario, seed=seed)

    return members, atm_sites, weights, labels


if __name__ == "__main__":
    members, atm_sites, weights, labels = generate_data()

    print("Members shape:", members.shape)
    print("ATM sites shape:", atm_sites.shape)
    print("Weights shape:", weights.shape)
    print("Labels shape:", labels.shape)
    print("First 5 members:")
    print(members[:5])
    print("First 5 weights:")
    print(weights[:5])
