import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Tuple, Optional
import pandas as pd


def reduce_newlines(text: str, max_newlines: int = 1) -> str:
  lines = text.split('\n')
  reduced_lines = []
  prev_line_empty = False
  for line in lines:
    if line.strip():
      reduced_lines.append(line)
      prev_line_empty = False
    elif not prev_line_empty:
      reduced_lines.extend([''] * min(max_newlines, 1))
      prev_line_empty = True

  reduced_text = ('\n' * max_newlines).join(reduced_lines)
  chunks = reduced_text.split('\n')
  return reduced_text


def scrape_link(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  return reduce_newlines(soup.get_text())


def extract_links(url: str) -> List[str]:
  content = requests.get(url).content
  soup = BeautifulSoup(content, 'html.parser')

  links = []
  for a in soup.find_all('a', href=True):
    link = a['href']
    absolute_link = urljoin(url, link)
    links.append(absolute_link)
  return links


def crawl_docs(url: str,
               depth: int = 3,
               visited_urls: Optional[set] = None,
               df=pd.DataFrame()) -> None:
  """Crawl a URL recursively to extract text content and links.

    Args:
        url: The URL to crawl.
        depth: The recursion depth limit.
        visited_urls: A set of visited URLs to avoid duplicate processing.
    """
  if depth < 0:
    return
  if visited_urls is None:
    visited_urls = set()

  if url in visited_urls:
    return

  print(f"Processing URL {url}")
  visited_urls.add(url)

  try:
    content = scrape_link(url)
    valid_links = extract_links(url)
    # with open(f'nodes.txt', 'a', encoding='utf-8') as nodes_file:
    new_df = chunk_and_write(content, nodes_file)
    df = pd.concat([df, new_df], ignore_index=True)
    for link in valid_links:
      crawl_docs(link, depth=depth - 1, visited_urls=visited_urls)
      # with open(f'edges.txt', 'a', encoding='utf-8') as edges_file:
      # edges_file.write(link + '\n')
  except Exception as e:
    print(f"Failed to process URL {url}: {e}")
  # df.columns = ['text']
  return df


def chunk_and_write(content: str, file) -> None:
  chunk_size = 1000
  start_index = 0
  df = pd.DataFrame()
  while start_index < len(content):
    chunk = content[start_index:start_index + chunk_size]
    # file.write(chunk + '\n')
    start_index += chunk_size
    new_row = pd.DataFrame({'text': [chunk]})
    df = pd.concat([df, new_row], ignore_index=True)
  return df


def merge_rows(df, column='text', target_length=1000):
  merged_text = ''
  merged_rows = []
  for text in df[column]:
    if len(merged_text) + len(text) > target_length:
      merged_rows.append(merged_text)
      merged_text = text
    else:
      merged_text += ' ' + text
  if merged_text:
    merged_rows.append(merged_text)

  return pd.DataFrame(merged_rows, columns=[column])
