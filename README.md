# SDN fault-tolerance simulation (paper experiments)

Flow-level, Gymnasium-compatible SDN routing simulator for the paper
«Improving Fault Tolerance in Software-Defined Networks with GNN and RL».
All software used is free / open-source.

## Layout
- `sdn_sim/topologies.py` — NSFNET-like (14n/21l) and fat-tree k=4
- `sdn_sim/env.py` — dynamic routing env (arrivals/departures, mice+elephants,
  hotspots, heterogeneous capacities, M/M/1-style delay)
- `sdn_sim/baselines.py` — SP and ECMP
- `sdn_sim/qlearning.py` — tabular Q-learning baseline
- `run_e1.py` — experiment E1 runner (see `EXP_E1_instruction.md`, in Russian)

## Quickstart
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python run_e1.py --selftest
python run_e1.py --fast
python run_e1.py --outdir ./results_e1_nsfnet
python run_e1.py --topology fattree --outdir ./results_e1_fattree
```
