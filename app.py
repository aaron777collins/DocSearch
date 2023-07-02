import json
import random
import string
import sys
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
import os
from PyPDF2 import PdfReader
import faiss
from pymongo import MongoClient, database
import io
import numpy as np
from dotenv import load_dotenv
import openai
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, load_chain
import requests
from typing import Any, Dict, List
from langchain.schema import BaseMessage, get_buffer_string, SystemMessage, HumanMessage, AIMessage
from langchain import LLMChain
from langchain.schema import messages_from_dict, messages_to_dict


load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

MAX_CHAT_INTERPRET_RETRIES=3

# Initialize Flask application
app = Flask(__name__)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}
swagger = Swagger(app, config=swagger_config)

# Establish a connection to MongoDB
client = MongoClient('mongodb://127.0.0.1:27017/')

def split_text(text, length, overlap):
    return [text[i : i + length] for i in range(0, len(text), length - overlap)]

def get_sentences_from_pdf(file):
    # Load PDF into PyPDF2 reader
    pdf = PdfReader(file)

    # Concatenate text from all pages
    text = ''.join([page.extractText() for page in pdf.pages])

    # Split text into chunks
    return split_text(text, 1000, 200)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    embedding = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    return np.array(embedding)

@app.route('/embeddings', methods=['POST'])
@swag_from('docs/embeddings.yml')  # YAML file describing the endpoint
def create_embeddings():
    # Get userID from the request
    userID = request.form['userID']

    # Get the PDF file from the request
    file = request.files['file']

    # Read sentences from PDF
    sentences = get_sentences_from_pdf(io.BytesIO(file.read()))

    # Get database and collection
    db = client[userID]  # Use userID as database name
    collection = db['sentences']
    counters = db['counters']

    embeddings = []
    for i, sentence in enumerate(sentences):
        embedding = get_embedding(sentence)
        embeddings.append(embedding)

        # Retrieve the current highest index for the user
        counter = counters.find_one({'userID': userID})
        if counter is None:
            # If the counter does not exist, create it
            counters.insert_one({'userID': userID, 'index': 0})
            highest_index = 0
        else:
            # If the counter does exist, increment it
            highest_index = counter['index'] + 1
            counters.update_one({'userID': userID}, {'$set': {'index': highest_index}})

        # Use the incremented index as the index for the sentence
        sentence_id = f"{userID}_{file.filename}_{highest_index}"

        # Store sentence along with its unique _id in MongoDB
        collection.insert_one({'_id': highest_index, 'sentence': sentence, "file": file.filename, "sentence_id": sentence_id, "timestamp": datetime.now()})

    # Convert list of arrays to 2D array
    embeddings = np.vstack(embeddings)

    # Create and store FAISS index
    if len(embeddings) > 0:

        print(embeddings.shape)
        index = None
        if os.path.exists(f'Indexes/{userID}.index'):
            index = faiss.read_index(f'Indexes/{userID}.index')
        else:
            # Create a new index if one does not exist
            index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # Create Indexes directory if it does not exist
        if not os.path.exists('Indexes'):
            os.makedirs('Indexes')

        faiss.write_index(index, f'Indexes/{userID}.index')

        return {"status": "success"}, 200
    else:
        return {"status": "error", "message": "No sentences were extracted from the PDF or no embeddings were created."}, 400


@app.route('/search', methods=['POST'])
@swag_from('docs/search.yml')  # YAML file describing the endpoint
def search():
    # Get userID and query from the request
    userID = request.form['userID']
    query = request.form['query']

    print(userID, query)

    # Get database
    db = client[userID]

    # Get the query embedding
    query_embedding = get_embedding(query)

    # Search the FAISS index and MongoDB collection for each file uploaded by the user
    results = []

    # Create Indexes directory if it does not exist
    if not os.path.exists('Indexes'):
        os.makedirs('Indexes')

    for file in os.listdir('Indexes/.'):
        if file.startswith(f'{userID}') and file.endswith('.index'):
            # Load the user's FAISS index
            index = faiss.read_index(f'Indexes/{file}')

            # Search the FAISS index
            D, I = index.search(np.array([query_embedding]), k=5)

            print(I[0].tolist())

            # Get the top 4 sentences associated with the nearest embedding
            collection = db["sentences"]
            for index in I[0]:
                if (index == -1):
                    continue
                result = collection.find_one({'_id': int(index)})  # Convert numpy integer to Python native integer
                if (result is None):
                    print("Could not find sentence with index", index, file=sys.stderr)
                    continue
                result['distance'] = float(D[0][I[0].tolist().index(index)])
                print(index, result['distance'])
                results.append(result)

    return jsonify(results)

@app.route('/getChatID', methods=['POST'])
@swag_from('docs/getChatID.yml')  # YAML file describing the endpoint
def getChatID():

    userID = request.form['userID']

    if client[userID] is None:
        return {"status": "error", "message": "User does not exist"}, 400

    db = client[userID]

    chatID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
    # ensure that the chatID is unique
    while db["chats"].find_one({"_id": chatID}) is not None:
        chatID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))

    return {"chatID": chatID}

@app.route('/chat', methods=['POST'])
@swag_from('docs/chat.yml')  # YAML file describing the endpoint
def chat():
    # Uses ConversationSummaryBufferMemory to store the conversation history
    # uses the gpt-3.5-turbo for chats
    # uses the gpt-3.5-turbo for summaries
    # Calls the search API with the query as the last sentence in the conversation
    # uses the 10 results from the search API, looping through each result and summarizing
    # the result with the query as the context
    # the best answer is iteratively refined as the AI's answer
    # the AI's answer is then added to the conversation history

    userID = request.form['userID']
    chatID = request.form['chatID']
    db = client[userID]

    if (userID is None or chatID is None):
        return {"status": "error", "message": "userID or chatID is not provided."}, 400


    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # search database for chain
    # if chain exists, load it
    # otherwise make one

    conversation_with_summaries = None

    # make sure chainBuffers folder exists
    if not os.path.exists('chainBuffers'):
        os.makedirs('chainBuffers')

    messagesFromDB = db["chats"].find_one({"_id": chatID})
    if (messagesFromDB is None):
        # create a new chain
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=3000,
            return_messages=True,
        )
    else:
        messages = messages_from_dict(messagesFromDB["messages"])
        summary = messagesFromDB["summary"]
        history = ChatMessageHistory(messages=messages)
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=3000,
            return_messages=True,
            chat_memory=history,
        )
        memory.moving_summary_buffer = summary


    conversation_with_summaries = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
    )


    # # check if the chatID for the respective userID exists in the database
    # # if it does, load the conversation history from the database
    # # otherwise create a new conversation history
    # chatsFromDB = db["chats"].find_one({"_id": chatID})
    # if (chatsFromDB is None):
    #     db["chats"].insert_one({"_id": chatID, "conversation": memory.chat_memory.messages})
    # else:
    #     memory.chat_memory.messages = chatsFromDB["conversation"]

    # get the query from the request
    query = request.form['query']

    if (query is None):
        return {"status": "error", "message": "query is not provided."}, 400

    # call the search API with the query as the last sentence in the conversation
    # get the top 10 results from the search API
    # loop through each result and summarize the result with the query as the context

    # calling the search API
    searchResults = requests.post("http://localhost:5000/search", data={"userID": userID, "query": query})

    # Example response
    #     [
    #   {
    #     "_id": 0,
    #     "distance": 0.4732133746147156,
    #     "file": "NewInfo (3).pdf",
    #     "sentence": "S e c r e t\np a s s w o r d\ni s\nI L i k e A p p l e s 3\nT h e\np u r p l e\ne l e p h a n t\ni s\nn a m e d\nJ o e\nS m i t h\na n d\nt h e\nb l u e\ne l e p h a n t\ni s\nn a m e d\nJ o h n\nC e n a .",
    #     "sentence_id": "Test123_NewInfo (3).pdf_0",
    #     "timestamp": "Sat, 01 Jul 2023 19:53:53 GMT"
    #   },
    #   ...
    # ]

    # extract the json objects out of the response array
    searchResults = searchResults.json()

    # loop through each result and summarize the result with the query as the context
    # the best answer is iteratively refined as the AI's answer
    answerSummary = None
    # for result in list(searchResults):
    #     predictionStr = f"Use the previous info, query: |{query}|, the current answer so far (this may change): |{answerSummary}| and possibly this new info to formulate a better answer: " + result["sentence"].replace("\n", " ")
    #     print(predictionStr)
    #     answerSummary = conversation_with_summaries.predict(input=predictionStr)
    #     print(answerSummary)

    infoStr = "-----".join([f"Snippet index from a file: <{i+1}> (note the index is hidden from answers), where the filename: '{result['file']}' (public) has the contents (this file may be unrelated to the other parts): " + result["sentence"].replace("\n", " ") + f"## END OF File snippet from {result['file']} (Note that when one asks which file to find the info in, these can be used) ##"for i, result in enumerate(searchResults)])

    # finalize an answer giving ONLY the response given the query
    predictionStr = f"Please provide clear and concise info related to this: {query} given the following info (Do not include the question in your response.. just the info. If no info is found, provide your best guess based on the info. Try not to mix people/things up between files. If the response is that you don't have any, include '[NO ANSWER]' within your response.):" + infoStr

    # copy the conversation but use the 16k model instead
    conversation_with_summaries_big = ConversationChain(
        llm=ChatOpenAI(model="gpt-3.5-turbo-16k"),
        memory=memory.copy(deep=True),
        verbose=True,
    )

    # attempting to answer with the 16k model with the summaries
    answer = "[NO ANSWER]"
    count = 0
    max_count = MAX_CHAT_INTERPRET_RETRIES
    while "[NO ANSWER]" in answer and count < max_count:
        answer = conversation_with_summaries_big.predict(input=predictionStr)
        count += 1

    # attempting to answer with the 16k model without the summaries but still with conversation context
    if "[NO ANSWER]" in answer:
        # attempt to answer with the 16k model without the summaries, just using your knowledge
        alreadyTrainedQueryStr = f"Please provide clear and concise info related to this: {query}. Use your abilities as a super-intelligent AI in all fields (language, math, reasoning, legal, etc.) to guess the answer. Say '[NO ANSWER]' if you can't guess the answer or related info. Note that the query is more important than previous info and may be unrelated to previous info."
        answer = conversation_with_summaries_big.predict(input=alreadyTrainedQueryStr)

    if "[NO ANSWER]" in answer:
        # remove the [NO ANSWER] from the answer
        answer = answer.replace("[NO ANSWER]", "")
        # if the answer is still empty, then just say that you don't know
        if answer.strip() == "":
            answer = "I apologize, but I don't know the answer to that with the information I have."

    print(query, answer)

    # add the AI's answer to the conversation history
    # memory.save_context(inputs={"input": query}, outputs={"output": answer})
    # memory.prune()

    memory.save_context(inputs={"input": query}, outputs={"output": answer})
    conversationHistory: List[BaseMessage]= memory.load_memory_variables(inputs=None)["history"]

    # save the conversation history to the database
    # db["chats"].update_one({"_id": chatID}, {"$set": {"conversation": memory.chat_memory.messages}})
    # db["summaries"].update_one({"_id": chatID}, {"$set": {"summary": memory.moving_summary_buffer}})

    # save the conversation history to the database
    nativeMsgObjects = messages_to_dict(conversationHistory)

    # get summary
    memorySummary = memory.moving_summary_buffer

    # insert or update the chat history
    db["chats"].update_one({"_id": chatID}, {"$set": {"messages": nativeMsgObjects, "summary": memorySummary}}, upsert=True)

    return {"answer": answer, "summary": memory.moving_summary_buffer, "history": nativeMsgObjects}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
