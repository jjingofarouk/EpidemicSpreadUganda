import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Set up parameters
N = 45.74e6  # population of Uganda (2020 estimate)
I0, R0 = 1000, 0  # initial number of infected and recovered individuals
S0 = N - I0 - R0  # initial number of susceptible individuals
beta = 0.3  # infection rate
gamma = 1/14  # recovery rate (1/average duration of infection)

# Time grid
t = np.linspace(0, 365, 365)  # 365 days

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t
ret = odeint(sir_model, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data
fig = plt.figure(figsize=(10,6))
plt.plot(t, S/1e6, 'b', alpha=0.5, lw=2, label='Susceptible')
plt.plot(t, I/1e6, 'r', alpha=0.5, lw=2, label='Infected')
plt.plot(t, R/1e6, 'g', alpha=0.5, lw=2, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number (millions)')
plt.title('SIR Model for Epidemic Spread in Uganda')
plt.legend()
plt.grid(True)

# Find and plot the peak of the infection
peak_day = np.argmax(I)
peak_infected = max(I)
plt.plot(peak_day, peak_infected/1e6, 'o', color='red')
plt.annotate(f'Peak: {peak_infected/1e6:.1f}M on day {peak_day}',
             xy=(peak_day, peak_infected/1e6), xytext=(peak_day+10, peak_infected/1e6),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

print(f"Peak infection occurs on day {peak_day} with {peak_infected/1e6:.1f} million infected.")
