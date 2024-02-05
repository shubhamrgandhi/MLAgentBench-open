# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('literature_review.csv')

# Clean the data
df = df.dropna()
df['Year'] = df['Year'].astype(int)

# Create a new column for the decade
df['Decade'] = df['Year'].apply(lambda x: x // 10 * 10)

# Group the data by decade and count the number of studies
df_decades = df.groupby('Decade').size().reset_index(name='Count')

# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(x='Decade', y='Count', data=df_decades)
plt.title('Number of Studies Published by Decade')
plt.xlabel('Decade')
plt.ylabel('Count')
plt.show()

# Calculate the average number of studies per year
avg_studies_per_year = df['Year'].count() / df['Decade'].nunique()

# Print the average number of studies per year
print('Average number of studies published per year:', avg_studies_per_year)