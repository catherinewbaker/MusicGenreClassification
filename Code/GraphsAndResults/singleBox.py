import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Model names
models = ['NB', 'SGD', 'KNN']

# Define your data
simple_data = np.array([
    [0.43, 0.45, 0.44, 0.25],  # NB
    [0.44, 0.44, 0.44, 0.5],    # SGD
    [0.46, 0.46, 0.46, 0.44]    # KNN
]).T  # Transpose the data to match models

echonest_data = np.array([
    [0.47, 0.55, 0.48, 0.47, 0.47, 0.5, 0.47, 0.47],   # NB
    [0.47, 0.48, 0.48, 0.47, 0.47, 0.49, 0.47, 0.47],   # SGD
    [0.46, 0.55, 0.48, 0.47, 0.47, 0.51, 0.52, 0.47]  # KNN
]).T  # Transpose the data to match models

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Transform the data
simple_scaled = scaler.fit_transform(simple_data)
echonest_scaled = scaler.fit_transform(echonest_data)

# Set up the figure and axes for plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Creating box plots with scaled data
axs[0].boxplot(simple_scaled, labels=models, notch=True, patch_artist=True)
axs[0].set_title('Simple Features (Scaled)')
axs[0].set_ylabel('Scaled Values')
axs[0].set_ylim(-0.5, 1.25)  # Set y-axis limits

axs[1].boxplot(echonest_scaled, labels=models, notch=True, patch_artist=True)
axs[1].set_title('Echonest Features (Scaled)')
axs[1].set_ylim(-0.5, 1.25)  # Set y-axis limits

plt.suptitle('Box Plot of Model Metrics for Scaled Simple and Echonest Features')
plt.show()
