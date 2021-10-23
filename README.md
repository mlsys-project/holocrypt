# HoloCrypt: Understanding and Optimizing the Performance Bottlenecks of End-to-End Private Inference

This repository contains the code for HoloCrypt. 
1. `client_garbler`: code for simulating the Client Garbler protocol
2. `server_garbler`: code for simulating the Server Garbler protocol
3. `data`: latency, communcation, and storage costs on a per-layer level for all networks and datasets (obtained using the [Delphi codebase](https://github.com/mc2-project/delphi))
4. `rust`: Networks configured in Rust

## Installation 
Download this repo:
```bash
https://github.com/mlsys-project/holocrypt.git
```
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```
## Running a simulation
Navigate to the correct protocol directory (for example, `client_garbler`). Run 
```bash
python simulate_client_garbler.py -h
``` 
to learn about the simulation parameters. Each run takes as input:
1. Storage Capacity (for Garbled Circuits)
2. Total Simulation Time 
3. Bandwidth 
4. Number of Simulation Runs
5. Neural Network (ResNet-18, ResNet-32, VGG-16)
6. Dataset (CIFAR-10, TinyImageNet)
7. Start Arrival Rate (requests / second)
8. End Arrival Rate (requests / second)
9. Arrival Rate step size

## Helpful Links
1. [Simpy Docs](https://simpy.readthedocs.io/en/latest/)
2. [Working with Simpy](https://medium.com/@malith.jayasinghe/understanding-the-performance-characteristics-of-systems-via-simulations-db7af8ba0ef1)
