# This file contains an LLM based AI system for the task of bibtex generation.

import re
import requests
from bs4 import BeautifulSoup

def generate_bibtex(title, authors, journal, year, volume, pages):
  """Generates a BibTeX entry for a journal article.

  Args:
    title: The title of the article.
    authors: A list of the authors of the article.
    journal: The name of the journal in which the article was published.
    year: The year in which the article was published.
    volume: The volume of the journal in which the article was published.
    pages: The pages on which the article was published.

  Returns:
    A BibTeX entry for the article.
  """

  # Convert the authors to a string in the format "Last, First".
  authors = ", ".join([f"{author.split()[-1]}, {author.split()[0]}" for author in authors])

  # Generate the BibTeX entry.
  bibtex = f"""
    @article{{
      title = {{'{title}'}},
      author = {{'{authors}'}},
      journal = {{'{journal}'}},
      year = {{'{year}'}},
      volume = {{'{volume}'}},
      pages = {{'{pages}'}}
    }}
  """

  # Return the BibTeX entry.
  return bibtex


def get_article_info(url):
  """Gets the article information from a URL.

  Args:
    url: The URL of the article.

  Returns:
    A tuple containing the title, authors, journal, year, volume, and pages of the article.
  """

  # Get the HTML of the article.
  html = requests.get(url).text

  # Parse the HTML.
  soup = BeautifulSoup(html, "html.parser")

  # Get the title of the article.
  title = soup.find("title").text

  # Get the authors of the article.
  authors = [author.text for author in soup.find_all("span", class_="authors__author")]

  # Get the journal of the article.
  journal = soup.find("span", class_="journal-title").text

  # Get the year of the article.
  year = soup.find("span", class_="year").text

  # Get the volume of the article.
  volume = soup.find("span", class_="volume").text

  # Get the pages of the article.
  pages = soup.find("span", class_="pages").text

  # Return the article information.
  return title, authors, journal, year, volume, pages


def main():
  """Gets the article information from a URL and generates a BibTeX entry for the article."""

  # Get the URL of the article.
  url = input("Enter the URL of the article: ")

  # Get the article information.
  title, authors, journal, year, volume, pages = get_article_info(url)

  # Generate the BibTeX entry.
  bibtex = generate_bibtex(title, authors, journal, year, volume, pages)

  # Print the BibTeX entry.
  print(bibtex)


if __name__ == "__main__":
  main()