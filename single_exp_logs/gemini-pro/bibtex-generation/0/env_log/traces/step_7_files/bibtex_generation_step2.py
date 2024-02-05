import spacy
from google_scholar_search import search_publications

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the text to be analyzed
text = """
According to a recent study, the average American consumes approximately 3,500 calories per day.
This is significantly higher than the recommended daily intake of 2,000 calories.
As a result, many Americans are overweight or obese.
In fact, the Centers for Disease Control and Prevention (CDC) reports that more than one-third of American adults are obese.
This is a serious problem, as obesity can lead to a number of health problems, including heart disease, stroke, type 2 diabetes, and cancer.
"""

# Process the text with spaCy
doc = nlp(text)

# Identify phrases and claims requiring citations
phrases_requiring_citations = []
claims_requiring_citations = []
for ent in doc.ents:
    if ent.label_ == "ORG" or ent.label_ == "PERSON":
        phrases_requiring_citations.append(ent.text)
    elif ent.label_ == "DATE" or ent.label_ == "TIME" or ent.label_ == "PERCENT" or ent.label_ == "MONEY" or ent.label_ == "QUANTITY":
        claims_requiring_citations.append(ent.text)

# Generate queries for Google Scholar API
queries = []
for phrase in phrases_requiring_citations:
    queries.append(f'"{phrase}"')
for claim in claims_requiring_citations:
    queries.append(f"{claim}")

# Print the queries
print("Queries for Google Scholar API:")
for query in queries:
    print(query)