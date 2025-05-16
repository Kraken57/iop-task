# IOP Project Repository

#### [List of task for reference](https://github.com/Kraken57/iop-task/tree/main/tasks)

### Hbridge Simualtion

- #### [Simualtion of H-Bridge in Matlab-Simulink ](https://github.com/Kraken57/iop-task/tree/main/matlab_simulation/hbridge_simulink)

### Simuation and Codes for PV system

- #### [Simulation of PV System in Matlab-Simulink](https://github.com/Kraken57/iop-task/tree/main/matlab_simulation/pvsystem_simulink)

- #### [Matlab code to generate the dataset for the PV System](https://github.com/Kraken57/iop-task/blob/main/ml_pvsystem/generatepvdataset.m)

- #### [Generated Dataset for the PV Sytem](https://github.com/Kraken57/iop-task/blob/main/ml_pvsystem/pvdataset.xlsx)

- #### [ML code to Compare the model and original PV system to predict the Duty cycle and MI](https://github.com/Kraken57/iop-task/blob/main/ml_pvsystem/pvmppt.ipynb)

### ML Model To C

- #### [Converting the Model the RL one to C file so that we can run it into DSP ](https://github.com/Kraken57/iop-task/tree/main/rltoc)
  > **NOTE**: You have to prune the model to reduce the size of this weights.h file

> **Also this weights.h file is for the ML model that is generating the SPWM wave**

### SPWM using ML

**1. [Documentation](https://github.com/Kraken57/iop-task/blob/main/solutions/task04/spwm_ml/documentation/spwm_ml.md)**

**2. [Generating Dataset from MATLAB](https://github.com/Kraken57/iop-task/blob/main/solutions/task04/spwm_ml/spwm.m)**

**3. [DATASET](https://github.com/Kraken57/iop-task/blob/main/solutions/task04/spwm_ml/spwm_dataset_60kHz.csv)**

**4. [RL code to train the model](https://github.com/Kraken57/iop-task/blob/main/solutions/task04/spwm_ml/spwm_ml.ipynb)**

### RL-Based Power Optimization in Series Circuit with Temperature-Dependent Resistance (Rs = αT)

- #### [Power Efficiency Estimation Using Neural Networks](https://github.com/Kraken57/iop-task/blob/main/solutions/task02/documentation/mldoc.md)(without temp as variable)

**1. [Documentation](https://github.com/Kraken57/iop-task/blob/main/solutions/task04/temp_ml/documentation/ralphaT_circuit.md)**

**2. [RL model code](https://github.com/Kraken57/iop-task/blob/main/solutions/task04/temp_ml/rsalphaT.ipynb)**

**3. [Trained RL model zip file](https://github.com/Kraken57/iop-task/blob/main/solutions/task04/temp_ml/circuit_rl_model.zip)**

**4. [Proof for the above model](https://github.com/Kraken57/iop-task/blob/main/solutions/task04/temp_ml/RL-proof.ipynb)**

### RL model to predict the SPWM

**1. [Code in google collab](https://github.com/Kraken57/iop-task/blob/main/solutions/task05/rl_spwm/src/spwm-rl.ipynb)**

**2. [RL model into matlab](https://github.com/Kraken57/iop-task/blob/main/solutions/task05/rl_spwm/matlabmodel/spwm_rl_function.m)**

**3. [Pruning docs](https://github.com/Kraken57/iop-task/blob/main/solutions/task05/documentation/pruning.md)**

**4. [Trained Model zip file](https://github.com/Kraken57/iop-task/tree/main/solutions/task05/rl_spwm/model)**

### CCS Codes

1. [SPWM by cpu_timer in CCS](https://github.com/Kraken57/iop-task/blob/main/ccscodes/spwmcputimer.c)

2. [ML based SPWM with cpu_timer in CCS](https://github.com/Kraken57/iop-task/blob/main/ccscodes/mlspwmwithcputimer.c)

3. [epwm code in CCS](https://github.com/Kraken57/iop-task/blob/main/ccscodes/epwm.c)

4. [SPWM with epwm in CCS](https://github.com/Kraken57/iop-task/blob/main/ccscodes/spwmwithepwm.c)
