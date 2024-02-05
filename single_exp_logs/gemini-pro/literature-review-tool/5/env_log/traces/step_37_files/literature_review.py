import os

# Create a new directory for the project.
os.makedirs('literature_review_tool', exist_ok=True)

# Add the project directory to the system path.
sys.path.append(os.path.join(os.getcwd(), 'literature_review_tool'))

# Import the necessary modules.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data.
data = pd.read_csv('data/literature_review_data.csv')

# Clean the data.
data = data.dropna()
data = data[data['year'] >= 2000]

# Create a pivot table to summarize the data.
pivot_table = data.pivot_table(index='year', columns='category', values='count', aggfunc='sum')

# Plot the pivot table.
sns.heatmap(pivot_table, annot=True, fmt='d')
plt.title('Number of Publications by Year and Category')
plt.xlabel('Year')
plt.ylabel('Category')
plt.show()