#NWChem-SOC Wrapper
Automated Spin-Orbit Coupling Calculations for NWChem
This repository provides a standalone Python tool to compute Spin-Orbit Coupling (SOC) strengths between singlet and triplet states. It leverages Linear-Response Time-Dependent Density Functional Theory (LR-TDDFT) to provide perturbative SOC corrections to non-relativistic electronic structure calculations.

#Overview
Spin-orbit coupling is essential for understanding excited-state processes like intersystem crossing and phosphorescence. While full relativistic treatments are computationally expensive, this tool treats SOC as a perturbative correction, allowing researchers to utilize existing non-relativistic packages.

Key Feature: This code extends NWChem functionality, providing an interface to compute SOC matrix elements—a feature not natively available in the standard NWChem suite.

#Technical Methodology
Theoretical Basis: Based on the Breit-Pauli Hamiltonian, treating spin as a natural consequence of relativistic quantum mechanics.

Approach: Uses LR-TDDFT to compute SOC strengths between states of different multiplicities.

Implementation: A flexible Python wrapper that parses NWChem outputs and performs the necessary matrix element integrations.

Features
Stand-alone Utility: Can be adapted to various electronic structure outputs.

Computational Efficiency: Avoids the overhead of full relativistic four-component calculations.

NWChem Integration: Specifically tailored to work with NWChem's transition density and orbital data.
