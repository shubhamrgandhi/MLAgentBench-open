# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arxiv import Arxiv
import transformers

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

# Define the function to call the arXiv API
def get_arxiv_data(query, start_date, end_date):
    # Create an Arxiv object
    arxiv = Arxiv()

    # Search for articles matching the query
    results = arxiv.search(query=query, start_date=start_date, end_date=end_date)

    # Convert the results to a pandas DataFrame
    df_arxiv = pd.DataFrame(results)

    # Return the DataFrame
    return df_arxiv

# Call the arXiv API to get data on articles published in the last 10 years
df_arxiv = get_arxiv_data(query='machine learning', start_date='2012-01-01', end_date='2022-12-31')

# Print the number of articles returned by the API
print('Number of articles returned by the API:', len(df_arxiv))

# Define the function to use LLM to summarize papers
def summarize_paper(paper_text):
    # Load the LLM model
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

    # Tokenize the paper text
    input_ids = model.tokenizer(paper_text, return_tensors="pt").input_ids

    # Generate the summary
    output = model.generate(input_ids, max_length=128)

    # Decode the summary
    summary = model.tokenizer.batch_decode(output, skip_special_tokens=True)

    # Return the summary
    return summary[0]

# Summarize the first 10 papers in the arXiv DataFrame
for paper in df_arxiv.head(10)['summary']:
    print(summarize_paper(paper))