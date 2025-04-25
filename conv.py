import numpy as np
import matplotlib.pyplot as plt
import pickle

index = '100_10_1_1_32'
np.random.seed(42)
# Define a simple MDP
num_states = 100
num_actions = 10
gamma = 0.9  # Discount factor

# Transition probabilities and rewards
P = np.random.dirichlet(np.ones(num_states), size=(num_states, num_actions))  # P(s'|s,a)
R = np.random.rand(num_states, num_actions)  # R(s, a)

# Actor and Critic parameters
theta = np.random.rand(num_states, num_actions)
Q = np.random.rand(num_states, num_actions)

# Step size functions
eta_k_funcs = {
    "0": lambda k: 0.01,
    "1/3": lambda k: 1*(k ** -0.333),
    "1/2": lambda k: 1*(k ** -0.5),
    "2/3": lambda k: 1*(k ** -0.667),
    "1": lambda k:   1*(k ** -1)
}
beta_k_funcs = {
    "0": lambda k: 0.01,
    "1/3": lambda k: 1*(k ** -0.333),
    "1/2": lambda k: 1*(k ** -0.5),
    "2/3": lambda k: 1*(k ** -0.667),
    "1": lambda k: 1*(k ** -1)
}

# Helper functions
def policy(s, theta):
    """Softmax policy."""
    exp_values = np.exp(theta[s])
    return exp_values / np.sum(exp_values)

def sample_action(s, theta):
    """Sample an action according to the current policy."""
    probs = policy(s, theta)
    return np.random.choice(len(probs), p=probs)

# Simulation parameters
num_iterations = 5000000
returns_logs_dict = {"0": [],"1/3": [], "1/2": [], "2/3": [], "1": []}

# Main simulation loop
def run_simulation(a_key):
    global theta, Q
    theta = np.random.rand(num_states, num_actions)
    Q = np.random.rand(num_states, num_actions)

    eta_k_func = eta_k_funcs[a_key]
    beta_k_func = beta_k_funcs[a_key]

    cumulative_returns = 0

    for k in range(1, num_iterations + 1):
        # Sample state, action, next state, and next action
        s = np.random.choice(num_states)
        a = sample_action(s, theta)
        s_prime = np.random.choice(num_states, p=P[s, a])
        a_prime = sample_action(s_prime, theta)

        # Immediate reward
        reward = R[s, a]
        cumulative_returns += reward

        # Calculate Advantage A(s, a)
        v_s = sum(policy(s, theta)[a] * Q[s, a] for a in range(num_actions))
        A_s_a = Q[s, a] - v_s

        # Policy update
        theta[s, a] += eta_k_func(k) * (1 - gamma) ** -1 * A_s_a

        # Q-value update
        td_target = reward + gamma * Q[s_prime, a_prime]
        Q[s, a] += beta_k_func(k) * (td_target - Q[s, a])

        # Log returns for analysis
        if k % 100 == 0:
            returns_logs_dict[a_key].append(cumulative_returns / k)
            print(k/num_iterations, a_key)

# Run simulations for different learning rates
for a_key in ["0", "1/3", "1/2", "2/3", "1"]:
    run_simulation(a_key)


# # Save the results to a file
# with open('returns_logs_dict{}.pkl'.format(index), 'wb') as f:
#     pickle.dump(returns_logs_dict, f)

# Plot the results
A = []
plt.figure(figsize=(12, 8))
for a_key, returns_logs in returns_logs_dict.items():
    plt.plot(returns_logs, label=f"a = {a_key}")
    A.append(returns_logs)
A = np.array(A)
np.savetxt('conv{}.txt'.format(index),A)

plt.title("Return of the Current Policy for Different Learning Rates")
plt.xlabel("Iteration (x100)")
plt.ylabel("Cumulative Average Return")
plt.legend()
plt.savefig("conv{}".format(index))
plt.show()
