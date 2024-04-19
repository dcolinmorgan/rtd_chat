from flask import Flask, request, render_template
import os, logging
import pandas as pd
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from helper import crawl_docs, merge_rows
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

app = Flask(__name__)
app.secret_key = os.environ['OPENAI_API_KEY']


def most_common_phrase(df, phrase):
  words = phrase.split()
  df['count'] = 0
  for word in words:
    df['count'] += df['text'].str.count(word)
  index = df['count'].idxmax()
  row_with_most_occurrences = df.loc[index]
  return row_with_most_occurrences


def string_prompt(df, query: str):
  source_knowledge = most_common_phrase(df, query).tolist()
  augmented_prompt = f"""Using the contexts below, answer the query.

  Contexts:
  {source_knowledge}

  Query: {query}"""
  return augmented_prompt


@app.route('/', methods=['GET', 'POST'])
def main():
  if request.method == 'POST':
    library_name = request.form.get('library_name')
    user_query = request.form.get('query')
    language = 'en'
    version = 'latest'

    url = f'https://{library_name}.readthedocs.io/{language}/{version}/'
    df = crawl_docs(url, depth=1)
    df.columns = ['text']
    # with open('nodes.txt', 'r') as f:
    # lines = f.readlines()
    # df = pd.DataFrame(lines, columns=['text'])
    df = df.drop(df.loc[df['text'] == '\n'].index)
    df['text'] = df['text'].apply(lambda row: ''.join(row))
    df['text'] = df['text'].apply(lambda x: ' '.join(x.split()))
    df = df[df['text'].str.len() >= 50]
    merged_df = merge_rows(df)
    merged_df = merged_df[~merged_df['text'].str.contains('�', na=False)]
    dd = pd.Series(pd.unique(merged_df['text']))
    merged_df = merged_df[~merged_df['text'].str.contains('�', na=False)]
    dd = pd.Series(pd.unique(merged_df['text']))

    chat = ChatOpenAI(openai_api_key=app.secret_key, model='gpt-3.5-turbo')
    prompt = HumanMessage(content=string_prompt(
        df=pd.DataFrame(dd, columns=['text']), query=user_query))
    # prompt = HumanMessage(content=string_prompt(user_query))

    # res = chat(messages + [prompt])
    # print(res.content)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi AI, how are you today?"),
        AIMessage(content="I'm great thank you. How can I help you?"),
        HumanMessage(content="I'd like to understand string theory.")
    ]

    response = chat(messages + [prompt])

    # Render the results.html template with the user's query and the response from the AI model
    return render_template('results.html',
                           query=user_query,
                           answer=response.content)

  # Render the index.html template if the request method is GET
  return render_template('index.html')


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=80)
