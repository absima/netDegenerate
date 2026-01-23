# Network Degeneration Pipeline

This repository implements a **generic, modular pipeline for network generation, ordering, degeneration, and analysis**, designed with neuroscience applications in mind but mostly generic. 


---

## Pipeline overview


[ Generate / Load Network ]
        |
        v
[ Edge Ordering (MaxMatch Decomposition) ]
        |
        v
[ Degeneration (Synapse / Neuron, staged) ]
        |
        v
[ Weighting (E/I 4-block scaling) ]
        |
        v
[ Simulation + Analysis + Store ]




Each stage produces well-defined artifacts that can be reused independently or orchestrated later via workflow managers (e.g. Snakemake).

---

## What this repository is about

This project focuses on **Stage I: structural network processing**, with the following goals:

- Provide **multiple network families** (random, small-world, spatial, scale-free-like).
- Define a **canonical edge and matrix convention** used consistently throughout.
- Introduce an **ordering of edges via repeated maximum matching**, enabling controlled degeneration strategies.
- Implement **degeneration protocols** that operate on either synapses or neurons.
- Keep the pipeline **agnostic to the simulator**, while demonstrating one concrete end-to-end use case.

The simulation step included here is **illustrative**.

---

## Conventions (used throughout)

- **Edge list:** `(src, tgt)`
- **Adjacency matrix:** `M[tgt, src] = 1` means `src → tgt`

---

## What is implemented

### 1. Network generation
Multiple directed network families are supported, including:
- Erdős–Rényi–type random graphs
- Degree-controlled variants
- Spatial / distance-dependent networks
- Small-world–like directed graphs
- Prototype “empirical-like” templates

Generation produces **binary sparse adjacency matrices** with no self-loops and duplicates.

---

### 2. Edge ordering (maximum matching decomposition)
Edges are ordered by **iteratively extracting maximum matchings**:
- Each layer is a matching (no shared sources or targets).
- Layers are concatenated to produce a full edge ordering.
- This ordering enables structured degeneration (e.g. early vs late removal).

---

### 3. Structural degeneration
Two degeneration modes are implemented:

- **Synapse trimming**
  - Random
  - In-degree–based
  - Out-degree–based
  - Ordered (using the matching decomposition)
  - Reverse-ordered 

- **Neuron trimming**
  - Degree-based strategies
  - Separate control of inhibitory and excitatory populations

---

### 4. Weight allocation
Binary adjacency matrices are converted to weighted networks using **block-structured I/E weights**:

- II, IE, EI, EE blocks (here XY means Y to X)
- Explicit weight dictionaries

---

### 5. Simulation & analysis (example)
A single wrapper (`simulateAndStore`) demonstrates how the structural pipeline can be coupled to a simulator:

- Builds the weighted network
- Runs a short simulation
- Computes basic structural and dynamical observables
- Stores results to disk

---

### 6. Tests and demos

The `tests_and_demo/` directory contains:

- **Unit tests** for presimulation components  
  (generation, ordering, degeneration, weighting)
- **End-to-end cascade demo**  
  (generation → ordering → trimming → simulation)
- **Visualization utilities** for sanity checks and inspection

These scripts are meant for **verification and development**.

---


## Outlook

Planned next steps include:
- Formal Snakemake workflows
- Simulator-independent dynamical backends
- Larger-scale parameter sweeps
- Extended structural observables
- more visualizing helpers
- feature correlation and prediction analysis

---


