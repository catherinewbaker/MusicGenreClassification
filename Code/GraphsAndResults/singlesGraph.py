import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['NB', 'SGD', 'KNN']

# Average values for Simple and Echonest features
simple_features_avg = [0.3925, 0.455, 0.455]
echonest_features_avg = [0.485, 0.475, 0.49125]

# Date recorded feature values
simple_features_date = [0.69, 0.7, 0.8]
echonest_features_date = [0.63, 0.65, 0.76]

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# X positions of the bars and width
ind = np.arange(len(models))  # the x locations for the groups
width = 0.35  # the width of the bars

# Plotting the bars for Simple and Echonest features
bars1 = ax.bar(ind - width/2, simple_features_avg, width, label='Simple Features Average')
bars2 = ax.bar(ind + width/2, echonest_features_avg, width, label='Echonest Features Average')

# Plotting the points for date recorded values
ax.scatter(ind - width/2, simple_features_date, color='blue', s=100, label='Date Recorded Value Simple', zorder=5)
ax.scatter(ind + width/2, echonest_features_date, color='orange', s=100, label='Date Recorded Value Echonest', zorder=5)

# Adding labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Values')
ax.set_title('Comparison of Model Performance by Feature Type for Single Feature Models')
ax.set_xticks(ind)
ax.set_xticklabels(models)
ax.legend()

plt.show()