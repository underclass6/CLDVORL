# CLDVORL

### Data Valuation based Batch-Constraint Q Learning (in PyTorch)



#### Links:
Data Valuation using Reinforcement Learning Pytorch Version
- https://github.com/DerrickXuNu/dvrl_pytorch

Author's PyTorch implementation of BCQ for continuous and discrete actions
- https://github.com/sfujim/BCQ

- https://github.com/Hyeyoung-WakeUp/Project-Robust-RL/tree/main/BCQ-master

  
  
### Modified Gym Environments  
  https://github.com/StanfordASL/ALPaCA
  
#### Papers:
- [ ] Data Valuation using Reinforcement Learning (DVRL)
    https://arxiv.org/pdf/1909.11671.pdf

- [ ] Off-Policy Deep Reinforcement Learning without Exploration (BCQ)
    https://arxiv.org/pdf/1812.02900.pdf

- [ ] Robust Adversarial Reinforcement Learning
    http://proceedings.mlr.press/v70/pinto17a/pinto17a.pdf
    
- [ ] Batch Value-function Approximation with Only Realizability
    https://arxiv.org/pdf/2008.04990.pdf
    
## Related Work

#### 1. Data valuation based on shapley value

- [A Distributional Framework for Data Valuation](https://arxiv.org/pdf/2002.12334.pdf) (ICML-2020)
- [Neuron Shapley: Discovering the Responsible Neurons](https://arxiv.org/pdf/2002.09815.pdf) (Nurips-2020)
- [Asymmetric Shapley values: incorporating causal knowledge into model-agnostic explainability](https://arxiv.org/pdf/1910.06358.pdf) (Nurips-2020)
- [The Explanation Game: Explaining Machine Learning Models Using Shapley Values](https://arxiv.org/pdf/1909.08128.pdf) (ARXIV-2020)
- [Towards Efficient Data Valuation Based on Shapley Value](https://arxiv.org/pdf/1902.10275.pdf) (AISTATS-2019)
- [Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms](https://arxiv.org/pdf/1908.08619.pdf) (VLDB-2019)
- [What is your data worth? Equitable Valuation of Data](https://arxiv.org/pdf/1904.02868.pdf) (ICML-2019)
- [L-shapley and C-shapley: Efficient Model Interpretation for Structured Data](https://arxiv.org/pdf/1808.02610.pdf) (ICLR-2018)

#### 2. Data valuation based on Reinforcement Learning

- [Data Valuation using Reinforcement Learning](https://arxiv.org/pdf/1909.11671.pdf) (ICML-2020)
- [A Minimax Game for Instanced based Selective Transfer Learning](https://dl.acm.org/doi/pdf/10.1145/3292500.3330841) (KDD-2019)


# DGX



### Step 1 — Creating the Key Pair
    ssh-keygen

### Step 2 — Copying the Public Key to Server5
    ssh-copy-id abolfazli@server5.l3s.uni-hannover.de

### Step 3 — Editting the local .ssh/config file
    Host l3srelay
    Hostname server5.l3s.uni-hannover.de
    User abolfazli
    
    Host dgx<br/>
    HostName 130.75.87.175
    User abolfazli
    ProxyCommand ssh -q l3srelay nc 130.75.87.175 22


### Step 4 — Copying the Public Key to the DGX Server
    ssh-copy-id -f dgx


### Step 5 — Building the Container

To build the container:

    docker build -t mujoco-container -f ${PWD}/Docker/Dockerfile ${PWD}/Docker

### Step 6 — Running the Container

To mount a volume to the container:
    `docker run -v ${PWD}/DVBCQ:/DVBCQ -it mujoco-container /bin/bash`


To run the container:

    docker run --gpus 'device=0' -v ${PWD}/notebooks:/notebooks -p 8989:8888 mujoco-container

For interactive mode:

    docker run --gpus 'device=0' -v ${PWD}/notebooks:/notebooks -p 8989:8888 -it mujoco-container /bin/bash/

Then start up jupyter via

    sh run_servers.sh
### Step 7 — Accesing the Jupyter Notebook Remotely

To remotely access jupyter notebook ssh in using tunelling:

    ssh -L 8989:localhost:8989 dgx

Then in your browser open 

    http://localhost:8989/

Jupyter notebook password: cloudywithachanceofamonia

### Step 8 — Setup OfflineRL-Lib

    python setup.py

# Usage

### Run Experiments

    python reproduce/run_cldvorl.py [--option value]
    
    Description
    option: device value: cpu, cuda
    option: seed value: integer, e.g. 111
    option: baseline value: awac, edac, inac, iql, sacn, td3bc, td7, xql
    option: exp_type value: vanilla, withoutCL, trajValuation
    option: dataset value: hopper-random-v2_hopper-medium-v2[,...], e.g. hopper-random-v2_hopper-medium-v2,hopper-random-v2_hopper-expert-v2
    option: target_max_epoch value: integer, e.g. 50

### Run Ablation Study

    python reproduce/run_cldvorl_ablation.py [--option value]
    
    Description
    option: device value: cpu, cuda
    option: seed value: integer, e.g. 111
    option: baseline value: awac, edac, inac, iql, sacn, td3bc, td7, xql
    option: ablation_types value: dcla_ratios[,...], e.g. dcla_ratios,modify_ratios
    option: dcla_ratios value: list of integer, e.g. 0.0,0.25,0.5,0.75,1.0
    option: modify_ratios value: list of integer, e.g. 0.0,0.25,0.5,0.75,1.0
    option: std_scales value: list of integer, e.g. 0.01,0.1,1,10,100
    option: temperatures value: list of integer, e.g. 0.01,0.1,1,10,100
