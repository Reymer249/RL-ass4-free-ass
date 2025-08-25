# Reinforcement Learning – n-step TD in Cliff Walking

This repository contains an academic project on **n-step Temporal Difference (TD) learning** methods, completed as part of the Reinforcement Learning course at Leiden University (2024).  
We investigate **Q-learning, SARSA, and Expected SARSA** in a 12×12 Cliff Walking environment, inspired by Sutton & Barto’s *Reinforcement Learning: An Introduction*.

---

## 📖 Project Overview
- **Environment**: 12×12 grid with cliffs, two random start positions, deterministic transitions, and a terminal goal.  
- **Algorithms studied**:
  - Q-learning (off-policy, greedy target policy)
  - SARSA (on-policy)
  - Expected SARSA (expectation over actions)
- **Experiments**:
  - Varying number of TD steps (*n*) and learning rates (*α*).
  - Comparing agent performance in deterministic vs. stochastic ("windy") environments.
  - Evaluation using **Area Under the Learning Curve (AUC)** and learning curve plots.

---

## 🔑 Key Findings
- Smaller **n** values (1–2) generally perform best in both environments.
- Larger **n** increases uncertainty in stochastic settings.
- Optimal learning rate **α decreases as n increases**.
- Suggested link between the **distance to the nearest cliff** and the optimal number of TD steps.

---

## 📂 Repository Contents
- `Report.pdf` – Full academic report with methodology, results, and discussion.
- `Appendix.pdf` - An appendix for the `Report.pdf`
- `IntroRL_Assignment_4.pdf` - An assignment statement specifying the tasks to complete.
- `requirements.txt` - Project requirements
- `results/` – Files with the results 
- `plots/` – Visualizations of agent performance.  
- `src/` – Python implementation of the environment and agents.  

---

## 🚀 Getting Started
Clone the repository and install dependencies:

```bash
git clone https://github.com/Reymer249/RL-ass4-free-ass
cd RL-ass4-free-ass
pip install -r requirements.txt
