try:
    df = pd.read_csv('literature_review.csv')
except FileNotFoundError:
    df = pd.DataFrame()
df.to_csv('literature_review.csv', index=False)

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('literature_review.csv')

# Clean the data
df.dropna(inplace=True)
df['Year'] = df['Year'].astype(int)

# Create a new column for the number of citations
df['Citations'] = df['Citations'].astype(int)

# Group the data by year and count the number of citations
df_grouped = df.groupby('Year').agg({'Citations': 'sum'})

# Plot the data
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_grouped, x='Year', y='Citations')
plt.title('Number of Citations by Year')
plt.xlabel('Year')
plt.ylabel('Number of Citations')
plt.show()

# Calculate the average number of citations per year
avg_citations = df['Citations'].mean()

# Print the average number of citations per year
print('Average number of citations per year:', avg_citations)

# Calculate the median number of citations per year
median_citations = df['Citations'].median()

# Print the median number of citations per year
print('Median number of citations per year:', median_citations)

# Calculate the mode number of citations per year
mode_citations = df['Citations'].mode()[0]

# Print the mode number of citations per year
print('Mode number of citations per year:', mode_citations)

# Calculate the standard deviation of the number of citations per year
std_citations = df['Citations'].std()

# Print the standard deviation of the number of citations per year
print('Standard deviation of the number of citations per year:', std_citations)

# Calculate the variance of the number of citations per year
var_citations = df['Citations'].var()

# Print the variance of the number of citations per year
print('Variance of the number of citations per year:', var_citations)

# Calculate the range of the number of citations per year
range_citations = df['Citations'].max() - df['Citations'].min()

# Print the range of the number of citations per year
print('Range of the number of citations per year:', range_citations)

# Calculate the interquartile range of the number of citations per year
iqr_citations = df['Citations'].quantile(0.75) - df['Citations'].quantile(0.25)

# Print the interquartile range of the number of citations per year
print('Interquartile range of the number of citations per year:', iqr_citations)

# Calculate the coefficient of variation of the number of citations per year
cv_citations = std_citations / avg_citations

# Print the coefficient of variation of the number of citations per year
print('Coefficient of variation of the number of citations per year:', cv_citations)

# Add the following code to the end of the file:

import requests
import json

# Set the API key
api_key = "YOUR_API_KEY"

# Set the query
query = "Language model hallucination detection"

# Set the URL
url = "https://api.arxiv.org/api/query?search_query={}".format(query)

# Make the request
response = requests.get(url)

# Parse the response
data = json.loads(response.text)

# Print the results
print(data)