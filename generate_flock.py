import numpy as np

# Simulation parameters
efficiency = 0.55  # Adjust based on specific paper (e.g., 0.55 for 2001 PRL)
num_points = 100000

# Generate mixed distribution: eta*Fock(1) + (1-eta)*Vacuum
# Vacuum component
vac = np.random.normal(0, 1/np.sqrt(2), num_points)
# Fock 1 component (can be sampled as x*vac or via specific distribution)
# For simplicity, using a distribution-specific sampler:
x = np.linspace(-5, 5, 1000)
p = efficiency * (2/np.sqrt(np.pi) * x**2 * np.exp(-x**2)) + \
    (1-efficiency) * (1/np.sqrt(np.pi) * np.exp(-x**2))
p /= p.sum() # Normalize

simulated_data = np.random.choice(x, size=num_points, p=p)

# Save to file
np.savetxt('ONEPHOTONFOCK_qnoise.txt', simulated_data, fmt='%.6f')