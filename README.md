# Dandelion vs. Spy: Network Privacy Simulator

This project is a dynamic, multi-process network simulator designed to evaluate the privacy guarantees of the **Dandelion protocol** against an adversary controlling a subset of "spy" (bribed) nodes. The simulator evaluates source-identification attacks across pure flooding and Dandelion diffusion networks under realistic latency and jitter constraints.

## 🏗 System Architecture

The simulation environment is built to mimic a real distributed network rather than relying on sequential, centralized loop steps:
*   **Independent Processes:** Every node in the simulated network runs as an independent OS process.
*   **Pure UDP Data Plane:** Nodes communicate exclusively over UDP using `asyncio`. The only bytes transmitted over the wire are the packet ID and its state (Stem or Fluff); creation times and true originators are never transmitted.
*   **Control Plane:** A lightweight TCP server is used strictly by the simulator controller to inject packets and orchestrate graceful shutdowns.
*   **Realistic Link Delays:** Base delay is calculated via Euclidean distance (1 ms per unit), and each transmission experiences an independent $U(-20\%, +20\%)$ jitter applied dynamically without blocking the event loop.

## ⚙️ Installation

The simulator requires Python 3.8+ and uses minimal external dependencies.

1. Clone the repository and navigate to the `Code` directory:
   ```bash
   cd Code
   ```
2. Install the required dependencies (`numpy>=1.24`, `matplotlib>=3.7`):
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## 🚀 Usage & CLI

The project provides several modules to run the simulation, verify structural integrity, and generate analytical reports. All scripts should be run as modules from the root `Code` directory.

*   **Run the Full Campaign:** 
    Generates the topology, runs calibration, executes all 5 simulation phases, and performs offline analysis.
    ```bash
    python3 -m src.experiments all
    ```
    *Note: Each run writes its output to `results/runs/` and will be skipped if it already exists, allowing the campaign to be safely interrupted and resumed.*

*   **Generate Plots & Tables:** 
    Parses `results.json` to generate performance visualizations and Markdown tables.
    ```bash
    python3 -m src.plots
    ```

*   **Verify Simulation Integrity:** 
    Checks all run directories to ensure topology invariants (no bridges, correct degrees), validates the Dandelion state machine (e.g., stem paths are valid graph walks), and confirms delay jitters match the specified bounds.
    ```bash
    python3 -m src.verify
    ```

*   **Build Final Report:** 
    Generates the comprehensive RTL Persian report (`Report.html` and `Report.pdf`) directly from the simulation data to ensure zero discrepancies.
    ```bash
    python3 -m src.report
    ```

*   **Interactive Demo:** 
    Runs a small, verbose simulation perfect for video demonstrations, logging the stem paths, spy receptions, and estimator guesses.
    ```bash
    python3 -m src.demo --mode dandelion --p 0.5 --packets 20
    ```

## 🕵️‍♂️ Adversary Estimators

The adversary has complete knowledge of the network topology and base delays, but only sees packets that reach its bribed spy nodes. We implement three source-estimation algorithms:

1.  **Baseline (`first_spy`)**: Guesses the neighbor that delivered the packet to the first spy node to observe it.
2.  **Phase 2 Attack (`timing_ml`)**: A maximum-likelihood estimator utilizing first-arrival times at all spies, assuming a Gaussian model of the jitter. It incorporates a strict consistency filter on the observed last hop to massively improve accuracy.
3.  **Phase 4 Attack (`dandelion_ml`)**: A highly advanced estimator aware of the Dandelion protocol. It combines a reverse random-walk likelihood for the stem phase with the timing ML likelihood of the fluff phase (mapped back through the stem kernel).

## 🧪 Simulation Phases

*   **Phase 1:** Baseline diffusion (flooding).
*   **Phase 2:** Source-identification attacks on flooding, evaluating the `timing_ml` estimator against the baseline.
*   **Phase 3:** Dandelion protocol implementation with varying stem probabilities ($p \in \{0.1, 0.5, 0.9\}$).
*   **Phase 4:** Advanced attacks on Dandelion using the `dandelion_ml` estimator.
*   **Phase 5:** Intentional spy delays. Bribed nodes hold packets for up to 1 base delay before relaying to disrupt timing analysis. Evaluates both an uncompensated adversary model and a compensated model that adapts its shortest-path graph to account for the intentional delays.
