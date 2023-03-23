from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import pandas as pd
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
from functions import create_context, answer_question

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatInput(BaseModel):
    messages: List[Message]
    max_len: Optional[int] = 1800
    size: Optional[str] = "ada"
    debug: Optional[bool] = False
    max_tokens: Optional[int] = 150
    stop_sequence: Optional[str] = None


@app.post("/chat")
async def chat_with_bot(chat_input: ChatInput):
    if not chat_input.messages or chat_input.messages[-1].role != "user":
        raise HTTPException(
            status_code=400, detail="The last message should be from the user.")

    context = create_context(
        chat_input.messages[-1].content,
        chat_input.messages,
        max_len=chat_input.max_len,
        size=chat_input.size,
    )

    # Add context message to the chat history
    chat_input.messages.insert(-1, {"role": "system",
                               "content": f"Context: {context}"})

    answer = answer_question(
        messages=chat_input.messages,
        question=chat_input.messages[-1].content,
        max_len=chat_input.max_len,
        size=chat_input.size,
        debug=chat_input.debug,
        max_tokens=chat_input.max_tokens,
        stop_sequence=chat_input.stop_sequence
    )

    chat_input.messages.append({"role": "assistant", "content": answer})
    return {"messages": chat_input.messages}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
