# 📖 Reinforcement Learning for Circuit Optimization

## **Introduction**
This documentation explains how we use **Reinforcement Learning (RL)** to optimize power in a simple circuit using **Stable-Baselines3's PPO algorithm**. We'll cover everything from defining the circuit environment, training an RL agent, validating its performance against theoretical calculations, and understanding the theory behind RL and PPO.

---
## **📌 Step 1: Understanding Reinforcement Learning and PPO**

### **What is Reinforcement Learning (RL)?**
Reinforcement Learning (RL) is a branch of machine learning where an agent learns by interacting with an environment. The agent takes actions, receives rewards, and improves its strategy to maximize the cumulative reward.

- **Agent**: The decision-maker (RL model)
- **Environment**: The system the agent interacts with (our circuit model)
- **State**: The current condition of the environment (temperature, resistance, etc.)
- **Action**: The decision taken by the agent (choosing load resistance $R_l$)
- **Reward**: A numerical value given as feedback (power transferred to the load)
- **Policy**: The strategy that the agent uses to decide actions

### **What is PPO (Proximal Policy Optimization)?**
PPO is a policy optimization algorithm that:
- Uses a neural network to approximate the best policy.
- Updates the policy gradually to prevent large, unstable updates.
- Balances exploration (trying new actions) and exploitation (using known good actions).

PPO is widely used due to its stability, efficiency, and ability to work in **continuous action spaces** like our circuit problem.

---
## **📌 Step 2: Import Required Libraries**
```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
```
### 🔹 **Why These Libraries?**
- **`gymnasium`**: Creates a custom RL environment.
- **`numpy`**: Handles numerical operations.
- **`matplotlib.pyplot`**: Used for plotting and visualizing results.
- **`stable_baselines3`**: Contains pre-built RL algorithms like PPO.

---
## **📌 Step 3: Define the RL Environment**
### **Circuit Model**
We consider a simple **series circuit** where we aim to **maximize the power transferred to the load**.

#### **Circuit Components:**
- **Supply voltage** $V = 10V$
- **Series resistance** $R_s = \alpha T$, where $\alpha = 0.05$ (depends on temperature $T$)
- **Load resistance** $R_l$ (action chosen by RL agent)

### **Power Calculation**
Using the standard power formula:
$P = \frac{V^2 R_l}{(R_s + R_l)^2}$
where:
- $P$ = power transferred to the load
- $V$ = supply voltage (constant at 10V)
- $R_s$ = series resistance
- $R_l$ = load resistance (chosen by RL agent)

### **State Space**
The environment state consists of:
- **Temperature (T):** Ranges from **20°C to 100°C**.
- **Series Resistance (Rs):** Computed as $R_s = 0.05T$, so it ranges from **1Ω to 5Ω**.
```python
self.observation_space = spaces.Box(low=np.array([20, 1]), high=np.array([100, 5]), dtype=np.float32)
```
### **Action Space**
The RL agent selects **load resistance** $R_l$ in the range **0.1Ω to 5Ω**:
```python
self.action_space = spaces.Box(low=np.array([0.1]), high=np.array([5.0]), dtype=np.float32)
```
### **Reset Function**
The environment starts by setting a **random temperature** $T$ and computing $R_s$:
```python
def reset(self, seed=None, options=None):
    T = np.random.uniform(20, 100)
    Rs = self.alpha * T
    self.state = np.array([T, Rs], dtype=np.float32)
    return self.state, {}
```
### **Step Function**
The agent selects $R_l$, and we calculate the power:
```python
def step(self, action):
    T, Rs = self.state
    Rl = np.clip(action[0], 0.1, 5)
    P = (self.V**2 * Rl) / ((Rs + Rl) ** 2)
    reward = P
    done = True  # Single-step task
    return self.state, reward, done, False, {}
```

---
## **📌 Step 4: Train RL Model**
```python
env = CircuitEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("circuit_rl_model.zip")
```
### **How Training Works?**
- The agent **selects different values of $R_l$** and learns from the power obtained.
- **The reward** is the power transferred to the load.
- Over **50,000 iterations**, PPO optimizes its policy.
- The trained model is saved as **circuit_rl_model.zip**.

---
## **📌 Step 5: Validate RL Model**
We compare the RL model’s **predictions** against the **theoretical maximum power transfer theorem**, which states:
$R_l$ = $R_s$
$P_{max}$ = $\frac{V^2}{4R_s}$

### **Testing the Model**
```python
temperatures = np.linspace(20, 100, 20)
predicted_rl = []
true_rl = []
predicted_power = []
true_power = []

for T in temperatures:
    Rs = env.alpha * T
    obs = np.array([T, Rs], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    Rl_rl = np.clip(action[0], 0.1, 5)
    P_rl = (env.V**2 * Rl_rl) / ((Rs + Rl_rl) ** 2)
    Rl_opt, P_max = theoretical_max_power(env.V, Rs)
    predicted_rl.append(Rl_rl)
    true_rl.append(Rl_opt)
    predicted_power.append(P_rl)
    true_power.append(P_max)
```

---
## **📌 Step 6: Plot Results**
```python
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(temperatures, true_rl, label="Theoretical Optimal Rl", marker="o")
plt.plot(temperatures, predicted_rl, label="RL Predicted Rl", marker="x")
plt.xlabel("Temperature (T)")
plt.ylabel("Rl")
plt.title("RL vs Theoretical Rl")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(temperatures, true_power, label="Theoretical Max Power", marker="o")
plt.plot(temperatures, predicted_power, label="RL Predicted Power", marker="x")
plt.xlabel("Temperature (T)")
plt.ylabel("Power")
plt.title("RL vs Theoretical Power")
plt.legend()
plt.tight_layout()
plt.show()
```

---
## **Conclusion**
- The RL agent learns **optimal resistance $R_l$** that closely matches the theoretical value.
- The **power values** predicted by RL align with theoretical predictions.
- This demonstrates that **RL can effectively learn circuit optimization problems**! 🚀

---
## **📌 Optional: Download the Model**
```python
from google.colab import files
files.download("circuit_rl_model.zip")
print("✅ Model downloaded successfully!")
```

