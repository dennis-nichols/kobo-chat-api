# Kobo Toolbox Chatbot API

This repository contains the backend API for the Kobo Toolbox support bot project. The API is built using the FastAPI framework in Python and leverages the OpenAI API with the GPT-3.5-turbo model to provide question-answering capabilities. This project is inspired by and based on the "Web Crawl Q&A" example from the OpenAI Cookbook repository.

## Overview

This API allows users to engage in a chat-based interaction with the GPT-3.5-turbo model using scraped content from the Kobo Toolbox support pages as context. The API uses FastAPI to create a chat endpoint where users can send a series of messages and receive responses generated by the language model. The chat history is used as context to provide meaningful and relevant answers to user queries.

## Features

- Chat-based question-answering capabilities using the GPT-3.5-turbo model
- FastAPI framework for building and deploying the API
- Integration with the OpenAI API for language model inference

## Getting Started

### Prerequisites

- Python 3.7+
- An OpenAI API key (obtainable from the [OpenAI platform](https://beta.openai.com/signup/))

### Installation

1. Clone this repository:

```
git clone https://github.com/your-repo-url/web-crawl-q-and-a.git
cd web-crawl-q-and-a
```

2. Install the required Python packages:

```
pip install -r requirements.txt
```

3. Make a .env file and set your OpenAI API key as an environment variable


### Running the API

1. Start the FastAPI server:

```
uvicorn main:app --reload
```

2. Access the API documentation at `http://localhost:8000/docs` to explore the available endpoints and interact with the API.


## Reference

This project is based on the "Web Crawl Q&A" example from the OpenAI Cookbook repository. You can find the original example
[here](https://github.com/openai/openai-cookbook/tree/main/apps/web-crawl-q-and-a).

