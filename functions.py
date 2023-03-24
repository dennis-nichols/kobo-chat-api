from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import pandas as pd
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


class Message(BaseModel):
    role: str
    content: str


def create_context(
    question, messages: List[Message], max_len=1800, size="ada", df=df
):
    """
    Create a context for a question by finding the most similar context from the dataframe and incorporating chat history
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(
        q_embeddings, df['embeddings'].values, distance_metric='cosine')

    scraped_texts = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        scraped_texts.append(row["text"])

    # Combine scraped text and chat history
    chat_history = "\n".join([message.content if hasattr(
        message, "content") else message["content"] for message in messages])



    # Return the combined context
    return f"###\n\n{chat_history}\n\n###\n\n" + "\n\n###\n\n".join(scraped_texts)


def answer_question(
    messages,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=2000,
    stop_sequence=None
):
    context = create_context(
        question,
        messages,
        max_len=max_len,
        size=size,
    )
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n If answers would be best in list form, please format them as lists. Context: {context}"},
                {"role": "user", "content": question}
            ],
            temperature=0.8,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence
        )

        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""
