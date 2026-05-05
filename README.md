#ATM Optimization

##Project Overview

This project implements an optimization-based approach to ATM placement for a credit union network. The goal is to analyze how ATM locations change under different demand patterns and constraints.

Two models are used:

1) Demand-Weighted Distance Model – minimizes total weighted travel distance  
2) Coverage Maximization Model – maximizes the number of members within a service radius  


##Scenarios

The analysis considers four demand scenarios:

1) Baseline  
2) Urban Demand Increase  
3) Suburban Demand Increase  
4) Demand Variability  


### Repository Layout

All source code is located in the `src/` directory.

1) data_generation.py – generates member locations and demand weights  
2) model.py – contains optimization models  
3) simulation.py – runs scenarios and prints results  
4) visualization.py – generates visuals  



##Outputs

The visualization script produces the figures used in the report, including:

1) ATM placement maps  
2) Regional comparison charts  
3) Sensitivity analysis (K = 2 vs K = 3)  
4) ATM selection patterns across scenarios  


##Setup
1) Clone repository: 

git clone https://github.com/Ciaracam/atm-optimization.git

2) From the project folder, install dependencies:

pip install -r requirements.txt

How to Run:

1) Run the full simulation pipeline:

python src/simulation.py

2) Generate visualizations:

python src/visualization.py 


-All data is generated directly within the simulation
-A fixed random seed is used so results stay consistent
-All outputs are automatically saved to the outputs/ folder