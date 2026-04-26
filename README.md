# NWChem-SOC Wrapper
### Automated Spin-Orbit Coupling (SOC) Matrix Element Calculations

This repository provides a standalone Python tool to compute **Spin-Orbit Coupling (SOC)** strengths between singlet and triplet states. It leverages Linear-Response Time-Dependent Density Functional Theory (LR-TDDFT) to provide perturbative SOC corrections to non-relativistic electronic structure calculations.

## Background
Spin-orbit coupling arises from the interaction of the magnetic moment of the electronic spin angular momentum with the magnetic field induced by the orbiting motion of other charges in an atom. 

In non-relativistic quantum theory, spin is an ad-hoc quantity. However, when principles of special relativity are integrated with quantum mechanics, spin arises naturally. While full relativistic treatments of molecular systems are computationally expensive, the energy contribution due to SOC is typically small enough to be treated as a **perturbative correction**. This allows us to take advantage of existing non-relativistic electronic structure packages while still capturing essential excited-state physics.

## Key Features
* **NWChem Integration:** Extends NWChem functionality by providing an SOC calculation interface not natively available in the standard suite.
* **Excited-State Analysis:** Enables the study of spin-forbidden processes such as **intersystem crossing** and **phosphorescence**.
* **Methodology:** Uses an accurate and flexible implementation based on LR-TDDFT to compute SOC strengths between states of different multiplicities.
* **Stand-alone Utility:** Designed to be flexible and adaptable to various electronic structure outputs.

## Implementation Details
The code is written in **Python**, designed for ease of use and integration into broader molecular dynamics workflows. For detailed setup and advanced configuration, please refer to Manual.pdf
