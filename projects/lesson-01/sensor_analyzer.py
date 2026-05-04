import numpy as np
import matplotlib.pyplot as plt

# Reproducible random numbers — same "noise" every run
np.random.seed(42)

# 1440 minutes in a day, sample temperature each minute
hours = np.linspace(0, 24, 1440)

# Base 22°C, +/- 3°C daily swing (sin wave), plus 0.5°C of sensor noise
temperature = 22 + 3 * np.sin(hours * np.pi / 12) + np.random.normal(0, 0.5, 1440)

print(f"Mean temperature: {temperature.mean():.2f} °C")
print(f"Max temperature:  {temperature.max():.2f} °C")
print(f"Min temperature:  {temperature.min():.2f} °C")
print(f"Std deviation:    {temperature.std():.2f} °C")

overheating_mask = temperature > 24
print(f"Minutes spent overheating: {overheating_mask.sum()}")

plt.figure(figsize=(10, 4))
plt.plot(hours, temperature, label="Temperature")
plt.axhline(y=24, color='r', linestyle='--', label="Overheat threshold")
plt.xlabel("Hour of day")
plt.ylabel("Temperature (°C)")
plt.title("Joey's Robot — 24h Sensor Data")
plt.legend()
plt.savefig("sensor_plot.png")
plt.show()