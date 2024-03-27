<h1 align="center">
Quantum Reaction-Limited Reaction-Diffusion Dynamics of Noninteracting Bose Gases
</h1>

This repository contains figure data and code implementation of the paper 'quantum reaction-limited reaction-diffusion dynamics of noninteracting Bose gases' (https://arxiv.org/abs/2311.04018).

The data to create figures can be found in the results folder. The code to create the figures can be found in the source-code (src) folder. There are two python scripts (phasediagram.py, qrd_dynamics.py), one containing the code used to create the phase diagram (Fig. 6 in the paper) and the other containing the code to create the dynamics and finding the effective exponent (Figs. 2-5 in the paper).

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Abstract

We investigate quantum reaction-diffusion systems in one-dimension with bosonic particles that coherently hop in a lattice, and when brought in range react dissipatively. Such reactions involve binary annihilation ($A + A \to \emptyset$) and coagulation ($A + A \to A$) of particles at distance $d$. 
We consider the reaction-limited regime, where dissipative reactions take place at a rate that is small compared to that of coherent hopping. In classical reaction-diffusion systems, this regime is correctly captured by the mean-field approximation. In quantum reaction-diffusion systems, for noninteracting fermionic systems, the reaction-limited regime recently attracted considerable attention because it has been shown to give universal power law decay beyond mean field for the density of particles as a function of time. Here, we address the question whether such universal behavior is present also in the case of the noninteracting Bose gas. 
We show that beyond mean-field density decay for bosons is possible only for reactions that allow for destructive interference of different decay channels. 
Furthermore, we study an absorbing-state phase transition induced by the competition between branching $A\to A+A$, decay $A\to \emptyset$ and coagulation $A+A\to A$. We find a stationary phase-diagram, where a first and a second-order transition line meet at a bicritical point which is described by tricritical directed percolation. 
These results show that quantum statistics significantly impact on both the stationary and the dynamical universal behavior of quantum reaction-diffusion systems.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt`
```

## Usage

- `git clone https://github.com/AG-Lesanovsky/2023-quantum-reaction-diffusion-dynamics-of-bose-gases`

To run the code:
```setup
python <path-to-code-file>
```
To work with the code and see examples, refer to the doc folder.

## License

Distributed under the CC-BY 4.0 License. See `LICENSE.txt` for more information.

