# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data.csv')

# Create a new column called 'total' that is the sum of the 'a' and 'b' columns
data['total'] = data['a'] + data['b']

# Create a scatter plot of the 'a' and 'b' columns
plt.scatter(data['a'], data['b'])

# Add a line of best fit to the scatter plot
plt.plot(np.unique(data['a']), np.poly1d(np.polyfit(data['a'], data['b'], 1))(np.unique(data['a'])), color='red')

# Show the plot
plt.show()