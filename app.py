import hashlib
import json
import logging
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
from flask_cors import CORS
import copy

logname = 'APILog.txt'

logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

MAX_CHAT_INTERPRET_RETRIES=3
MAX_CHAT_REQ_COUNT = 20
TEMPURATURE=0.7

# Initialize Flask application
app = Flask(__name__)
CORS(app)
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

def checkFlagged(input: str) -> bool:
    # hit the openAI moderation endpoint
    # if the response is flagged, return true
    # otherwise return false

    #  EXAMPLE RESPONSE    {
    #   "id": "modr-XXXXX",
    #   "model": "text-moderation-005",
    #   "results": [
    #     {
    #       "flagged": true,
    #       "categories": {
    #         "sexual": false,
    #         "hate": false,
    #         "harassment": false,
    #         "self-harm": false,
    #         "sexual/minors": false,
    #         "hate/threatening": false,
    #         "violence/graphic": false,
    #         "self-harm/intent": false,
    #         "self-harm/instructions": false,
    #         "harassment/threatening": true,
    #         "violence": true,
    #       },
    #       "category_scores": {
    #         "sexual": 1.2282071e-06,
    #         "hate": 0.010696256,
    #         "harassment": 0.29842457,
    #         "self-harm": 1.5236925e-08,
    #         "sexual/minors": 5.7246268e-08,
    #         "hate/threatening": 0.0060676364,
    #         "violence/graphic": 4.435014e-06,
    #         "self-harm/intent": 8.098441e-10,
    #         "self-harm/instructions": 2.8498655e-11,
    #         "harassment/threatening": 0.63055265,
    #         "violence": 0.99011886,
    #       }
    #     }
    #   ]
    # }
    response = openai.Moderation.create(
        input=input
    )
    output = response["results"][0]["flagged"]
    return output

@app.route('/logs', methods=['GET'])
def get_logs():
    with open("APILog.txt", "r") as f:
        lines = f.readlines()
        last_lines = lines[-100:]
    return jsonify({"logs": last_lines})

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

    logger = logging.getLogger('embeddings')

    # Get userID from the request
    userID = request.form['userID']

    # Get the PDF file from the request
    file = request.files['file']
    date = datetime.now()


    # Read sentences from PDF
    sentences = get_sentences_from_pdf(io.BytesIO(file.read()))


    originalFileNameWithoutExtension = file.filename.split(".")[0]
    fileExtension = file.filename.split(".")[1]
    filename = originalFileNameWithoutExtension.replace(" ", "") + "_" + hashlib.sha256("--- JOINED SNIPPET ---".join(sentences).encode()).hexdigest() + "." + fileExtension

    # Get database and collection
    db = client[userID]  # Use userID as database name
    collection = db['sentences']
    counters = db['counters']

    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=TEMPURATURE)

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=10000,
        return_messages=True,
        chat_memory=ChatMessageHistory(messages=[]),
    )

    conversation_with_summaries_big = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
    )

    embeddings = []
    lowest_index = -1
    highest_index = -1
    fileIDsAdded = []

    for i, sentence in enumerate(sentences):
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


        if lowest_index == -1:
            lowest_index = highest_index

        # make summary of the sentence as the current snippet
        # using the previous convo to make sense
        predictionStr = f"You're a super intelligent AI with amazing linguistic and summary skills and keeps track of where info comes from (for future summarizing). Using the previous information (if any), please summarize the following information:" + sentence.replace("\n", " ") + "-- METADATA: From File: {filename} at snippet ID: {highest_index},  (Only respond with the summary related info and do not say anything else.)"

        # get the summary
        if checkFlagged(predictionStr):
            # if the summary is flagged, then the content is inappropriate
            # so we should stop summarizing and return an error
            return {"status": "error", "message": "The content is inappropriate."}, 400
        summary = conversation_with_summaries_big.predict(input=predictionStr)
        conversation_with_summaries_big.memory.save_context(inputs={"input": predictionStr}, outputs={"output": summary})

        sentence_with_summary = f"{sentence} [SUMMARY of snippet from File: {filename} with snippet index: {highest_index} saved at date {date}: {summary} ENDOFSUMMARY]"

        embedding = get_embedding(sentence_with_summary)
        embeddings.append(embedding)


        # Use the incremented index as the index for the sentence
        sentence_id = f"{userID}_{filename}_{highest_index}"

        # Store sentence along with its unique _id in MongoDB
        collection.insert_one({'_id': highest_index, 'sentence': sentence_with_summary, "file": filename, "sentence_id": sentence_id, "timestamp": date})
        fileIDsAdded.append(sentence_id)

    # preparing a final summary of the entire file using the conversation info.
    fileSummaryPrompt = f"You're a super intelligent AI with amazing linguistic and summary skills. Please summarize the entire file: {filename} using the previous info from our chat (Note that this summary will be used for future searches. Only respond with the summary related info and do not say anything else)."

    # check flagged
    if checkFlagged(fileSummaryPrompt):
        # if the summary is flagged, then the content is inappropriate
        # so we should stop summarizing and return an error
        # we also need to delete the sentences that were added to the database

        # delete the sentences that were added to the database
        for fileID in fileIDsAdded:
            collection.delete_one({"sentence_id": fileID})

        return {"status": "error", "message": "The content is inappropriate."}, 400
    fileSummary = conversation_with_summaries_big.predict(input=fileSummaryPrompt)

    finalSummaryStr = f"[SUMMARY of File: {filename} with snippet indexes ranging from {lowest_index} to {highest_index} saved at the datetime {date}: {fileSummary} ENDOFSUMMARY]"

    # save the summary to the database
    db["summaries"].update_one({"_id": filename}, {"$set": {"summary": finalSummaryStr}}, upsert=True)

    # Convert list of arrays to 2D array
    embeddings = np.vstack(embeddings)

    # Create and store FAISS index
    if len(embeddings) > 0:

        logger.info(str(embeddings.shape))
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

        return {"status": "success", "fileID": filename}, 200
    else:
        return {"status": "error", "message": "No sentences were extracted from the PDF or no embeddings were created."}, 400


@app.route('/search', methods=['POST'])
@swag_from('docs/search.yml')  # YAML file describing the endpoint
def search():

    logger = logging.getLogger('search')

    # Get userID and query from the request
    userID = request.form['userID']
    query = request.form['query']

    logger.info(userID + " " + query)

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

            logger.info(str(I[0].tolist()))

            # Get the top 4 sentences associated with the nearest embedding
            collection = db["sentences"]
            for index in I[0]:
                if (index == -1):
                    continue
                result = collection.find_one({'_id': int(index)})  # Convert numpy integer to Python native integer
                if (result is None):
                    logger.error("Could not find sentence with index" + str(index))
                    continue
                result['distance'] = float(D[0][I[0].tolist().index(index)])
                logger.info(str(index) + " " + str(result['distance']))
                results.append(result)

    return jsonify(results)

@app.route('/getChatID', methods=['POST'])
@swag_from('docs/getChatID.yml')  # YAML file describing the endpoint
def getChatID():

    logger = logging.getLogger('getChatID')

    userID = request.form['userID']

    if client[userID] is None:
        return {"status": "error", "message": "User does not exist"}, 400

    db = client[userID]

    chatID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
    # ensure that the chatID is unique
    while db["chats"].find_one({"_id": chatID}) is not None:
        chatID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))

    return {"chatID": chatID}

@app.route('/chatWithAIQuery', methods=['POST'])
@swag_from('docs/chatWithAIQuery.yml')  # YAML file describing the endpoint
def chatWithAIQuery():
    # Rewords the query given the context of the conversation
    # to give better results
    # Uses the gpt-3.5-turbo-16k for rewording
    userID = request.form['userID']
    chatID = request.form['chatID']
    query = request.form['query']
    db = client[userID]

    logger = logging.getLogger('chatWithAIQuery')

    if (userID is None or chatID is None or query is None):
        return {"status": "error", "message": "userID, chatID, or query is not provided."}, 400

    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=TEMPURATURE)

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

    conversation_with_summaries_big = ConversationChain(
        llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=TEMPURATURE),
        memory=memory,
        verbose=True,
    )

    # Save memory before using it
    OLD_MEMORY = memory.copy(deep=True)

    # get the reworded query
    rewordQueryStr = f"You're a super intelligent AI with the ability to reword prompts to search a vector database easier. \
        Please rewrite this query to make sense and for finding the best answer to the question (if the prompt is good as is, or is just continuing the conversation, \
                just respond with the original prompt. Note that you may need to use the context of this conversation to create a \
                    new query for the info. Don't say 'reworded prompt:', just say the new prompt.): {query} (Only respond with the \
                        reworded prompt and do not say anything else.)"

    # check flagged
    if checkFlagged(rewordQueryStr):
        # if the query is flagged, then the content is inappropriate
        # so we should stop answering and return an error
        return {"status": "error", "message": "The content is inappropriate."}, 400
    reworded_query = conversation_with_summaries_big.predict(input=rewordQueryStr)


    # now calling the /chat endpoint with the reworded query
    res = requests.post("http://localhost:5000/chat", data={"userID": userID, "chatID": chatID, "query": reworded_query})
    return res.json()

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

    logger = logging.getLogger('chat')

    userID = request.form['userID']
    chatID = request.form['chatID']
    query = request.form['query']
    db = client[userID]

    if (userID is None or chatID is None or query is None):
        return {"status": "error", "message": "userID, chatID, or query is not provided."}, 400


    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=TEMPURATURE)

    # search database for chain
    # if chain exists, load it
    # otherwise make one

    # conversation_with_summaries = None

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
            max_token_limit=10000,
            return_messages=True,
            chat_memory=history,
        )
        memory.moving_summary_buffer = summary


    # conversation_with_summaries = ConversationChain(
    #     llm=llm,
    #     memory=memory,
    #     verbose=True,
    # )

    # copy the conversation but use the 16k model instead
    conversation_with_summaries_big = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
    )

    # Save memory before using it
    OLD_MEMORY = memory.copy(deep=True)


    # # check if the chatID for the respective userID exists in the database
    # # if it does, load the conversation history from the database
    # # otherwise create a new conversation history
    # chatsFromDB = db["chats"].find_one({"_id": chatID})
    # if (chatsFromDB is None):
    #     db["chats"].insert_one({"_id": chatID, "conversation": memory.chat_memory.messages})
    # else:
    #     memory.chat_memory.messages = chatsFromDB["conversation"]

    copy_of_conversation_with_summaries_big = ConversationChain(
        llm=llm,
        memory=memory.copy(deep=True),
        verbose=True,
    )

    shouldQueryStr = f"You're a super intelligent AI with the ability to do anything. Please determine whether more info \
        is needed to answer this question or if we can just respond with the answer. If more info is needed, please respond \
            with '[QUERY]' and if we can just respond with the answer, please respond with the answer, followed by '[CONTINUE CONVO]'. The query is: {query} (Only respond with '[QUERY]' or the answer with '[CONTINUE CONVO]' and do not say anything else.)"
    # check flagged
    if checkFlagged(shouldQueryStr):
        # if the query is flagged, then the content is inappropriate
        # so we should stop answering and return an error
        return {"status": "error", "message": "The content is inappropriate."}, 400
    shouldQueryResponse = copy_of_conversation_with_summaries_big.predict(input=shouldQueryStr)

    if "[CONTINUE CONVO]" in shouldQueryResponse:
        # Respond with the answer without [CONTINUE CONVO]
        shouldQueryResponse = shouldQueryResponse.replace("[CONTINUE CONVO]", "")

        memory.save_context(inputs={"input": query}, outputs={"output": shouldQueryResponse})
        conversationHistory: List[BaseMessage]= memory.load_memory_variables(inputs=None)["history"]

        # combine the conversation history from OLD_MEMORY and the new conversation history
        OLD_CONVERSATION_HISTORY = OLD_MEMORY.load_memory_variables(inputs=None)["history"]

        oldConversationPlusOnlyLatestMessages = OLD_CONVERSATION_HISTORY + conversationHistory[-2:]

        # removing system messages
        oldConversationPlusOnlyLatestMessages = [msg for msg in oldConversationPlusOnlyLatestMessages if msg.type != "system"]

        # save the conversation history to the database
        # db["chats"].update_one({"_id": chatID}, {"$set": {"conversation": memory.chat_memory.messages}})
        # db["summaries"].update_one({"_id": chatID}, {"$set": {"summary": memory.moving_summary_buffer}})

        # save the conversation history to the database
        nativeMsgObjects = messages_to_dict(oldConversationPlusOnlyLatestMessages)

        # get summary
        memorySummary = memory.moving_summary_buffer

        # insert or update the chat history
        db["chats"].update_one({"_id": chatID}, {"$set": {"messages": nativeMsgObjects, "summary": memorySummary}}, upsert=True)

        return {"answer": shouldQueryResponse, "summary": memory.moving_summary_buffer, "history": nativeMsgObjects}
    else:

        # call the search API with the query as the last sentence in the conversation
        # get the top 10 results from the search API
        # loop through each result and summarize the result with the query as the context

        # predict whether we should query or just respond
        # if the query is a question that we wouldn't know how to answer, then we should query and include [QUERY]
        # otherwise we should just respond and include [CONTINUE CONVO]

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


        infoStr = "-----".join([f"Snippet index from a file: <{i+1}> (note the index is hidden from answers), where the filename: '{result['file']}' (public) and timestamp: {result['timestamp']} (public) has the contents (this file may be unrelated to the other parts): " + result["sentence"].replace("\n", " ") + f"## END OF File snippet from {result['file']} (Note that when one asks which file to find the info in, these can be used) ##"for i, result in enumerate(searchResults)])

        # finalize an answer giving ONLY the response given the query
        predictionStr = f"Please gather info to make a decision. The question to be answered is: {query}. IMPORTANT: You will (and have been) be given info from a file snippet with <i> as the snippet index and a filename as well. If the information from a file snippet at <i> is being used (and is relevent) then append '[REQUEST <FILENAME> <i+1>]' to retrieve the next snippet. ALWAYS request the next snippet if you're using the current snippet from that file to answer the query. Regarding the decision, either provide clear and concise info related to this: {query} given the following info (plus what you requested) (Do not include the question in your response.. just the info. If no info is found, provide your best guess based on the info. Try not to mix people/things up between files.) OR if the response would be that you don't have any related info, request more info (as '[REQUEST <FILENAME> <i+1>]' )and say '[NO ANSWER]' within your response. Searched Info:" + infoStr + "IMPORTANT REMINDER: You will be given info from a file snippet with <i> as the snippet index and a filename as well. If the information from a file snippet at <i> is being used then append '[REQUEST <FILENAME> <i+1>]' to retrieve the next snippet. ALWAYS request the next snippet if you're using the current snippet from that file. If you are told [FAILEDREQUEST <FILENAME> <[Your wanted ID]>] then the retrieval failed. Notify the user that you couldn't retrieve more info in this one request because it doesn't exist or you reached the max retrieval limit. Do not tell the user that they can request more info using [Request <FILENAME> <i+1>] or any other method since that's hidden. Simply tell them that they can ask for more info if they'd like but would need to specify the topic. Do not mention the snippets. Note that files, their upload dates, and their contents are public and important to remember. Info such as [SUMMARY ...] is the generated summary of the snippet. Do not mention the [SUMMARY ...] part in your response. (Only respond with the info )"


        # attempting to answer with the 16k model with the summaries
        answer = "[NO ANSWER]"
        count = 0
        max_count = MAX_CHAT_INTERPRET_RETRIES
        reqCount = 0
        max_req_count = MAX_CHAT_REQ_COUNT

        newInfoToAddToContext = set()
        while ("[NO ANSWER]" in answer and count < max_count) or ("[REQUEST" in answer and reqCount < max_req_count):
            while "[REQUEST" in answer and reqCount < max_req_count:
                try:
                    logger.info("Request found:" + " " + str(answer))
                    # search for the related snippet and predict again
                    # parse out the [REQUEST ...] string
                    reqStr = answer[answer.index("[REQUEST") + len("[REQUEST"):answer.index("]")].strip()
                    # originalReqStr = reqStr
                    originalFullReqStr = answer[answer.index("[REQUEST"):answer.index("]") + 1].strip()
                    reqStr = reqStr.replace("[REQUEST", "").replace("]", "").strip()
                    args = reqStr.split(" ")
                    filename = "".join(args[:-1])
                    newID = args[-1]
                    if newID.isdigit():
                        newID = int(newID)
                    else:
                        if "+" in newID:
                            addArgs = newID.split("+")
                            newID = int(addArgs[0]) + int(addArgs[1])
                    res = db["sentences"].find_one({"sentence_id": f"{userID}_{filename}_{newID}"})
                    if res is not None:
                        # new snippet found, append it to the infoStr
                        logger.debug(f"Requested {newID} from {filename} and found {res['sentence']}")
                        foundInfo = "-----".join([f"Snippet index from a file: <{i+1}> (note the index is hidden from answers), where the filename: '{result['file']}' (public) and timestamp: {result['timestamp']} (public) has the contents (this file may be unrelated to the other parts): " + result["sentence"].replace("\n", " ") + f"## END OF File snippet from {result['file']} (Note that when one asks which file to find the info in, these can be used) ##"for i, result in enumerate([res])])
                        # answer = answer.replace(originalReqStr, foundInfo)
                        answer = answer.replace(originalFullReqStr, "") # make string empty
                        # add the found info to the conversation context
                        # conversation_with_summaries_big.memory.save_context(inputs={"input": "Context: " + answer + " REQUESTED INFO: " + originalReqStr}, outputs={"output": foundInfo})
                        logger.debug("New info to be added to context: " + foundInfo)
                        newInfoToAddToContext.add((originalFullReqStr,  f"Request: {originalFullReqStr} returned {foundInfo}"))
                    else:
                        # no snippet found, mention the failed request
                        # [FAILEDREQUEST <FILENAME> <i>]
                        answer = answer.replace(originalFullReqStr, "")
                        logger.debug(f"Requested {newID} from {filename} and failed to find it.")
                        # conversation_with_summaries_big.memory.save_context(inputs={"input": "Context: " + answer + " REQUESTED: " + originalReqStr}, outputs={"output": f"[FAILEDREQUEST {filename} {newID} - Try fixing the file name or index. it should match exactly without spaces.]"})
                        logger.debug("Failed request added to context: " + f"[FAILEDREQUEST {filename} {newID} - Try fixing the file name or index. it should match exactly without spaces.]")
                        newInfoToAddToContext.add((originalFullReqStr, f"[FAILEDREQUEST {filename} {newID} - Try fixing the file name or index. it should match exactly without spaces.]"))

                except:
                    # exception occured replacing snippet
                    # just remove the [REQUEST ...] string
                    logger.debug("ISSUE WITH REQUESTING DATA: " + answer + " asking bot to fix it.")
                    # requesting BOT to fix request syntax
                    # answer = answer.replace(originalReqStr, "")

                    # answer = answer + "ISSUE WITH REQUESTING DATA: " + originalReqStr + " Please fix request syntax to match [REQUEST <FILENAME> <i+1>]"
                    # conversation_with_summaries_big.memory.save_context(inputs={"input": "Context: " + answer + " REQUESTED: " + originalReqStr}, outputs={"output": f"ISSUE WITH REQUESTING DATA: " + originalReqStr + " Please fix request syntax to match [REQUEST <FILENAME> <i+1>]"})
                    newInfoToAddToContext.add((originalFullReqStr, f"ISSUE WITH REQUESTING DATA: " + originalFullReqStr + " Please fix request syntax to match [REQUEST <FILENAME> <i+1>]"))

                reqCount += 1

            # looping through newInfoToAddToContext and creating a context string and saving it
            requestsStr = ""
            responsesStr = ""
            for req, resp in newInfoToAddToContext:
                requestsStr += req + " "
                responsesStr += resp + " "

            # add to context
            conversation_with_summaries_big.memory.save_context(inputs={"input": "Context: " + answer + " REQUESTED: " + requestsStr}, outputs={"output": responsesStr})

            logger.debug("Inputs: " + "Context: " + answer + " \nREQUESTED: " + requestsStr + " \nOutputs: " + responsesStr)

            debugMemVars = conversation_with_summaries_big.memory.load_memory_variables(inputs=None)
            logger.debug(jsonify(messages_to_dict(debugMemVars["history"])))
            logger.debug(jsonify(conversation_with_summaries_big.memory.moving_summary_buffer))

            # check flagged
            if checkFlagged(predictionStr):
                # if the query is flagged, then the content is inappropriate
                # so we should stop answering and return an error
                return {"status": "error", "message": "The content is inappropriate."}, 400
            answer = conversation_with_summaries_big.predict(input=predictionStr)
            count += 1



        # attempting to answer with the 16k model without the summaries but still with conversation context
        if "[NO ANSWER]" in answer:
            # attempt to answer with the 16k model without the summaries, just using your knowledge
            alreadyTrainedQueryStr = f"Please provide clear and concise info related to this: {query}. Use your abilities as a super-intelligent AI in all fields (language, math, reasoning, legal, etc.) to guess the answer. Say '[NO ANSWER]' if you can't guess the answer or related info. Note that the query is more important than previous info or conversation and may be unrelated to previous info/conversation."
            # check flagged
            if checkFlagged(alreadyTrainedQueryStr):
                # if the query is flagged, then the content is inappropriate
                # so we should stop answering and return an error
                return {"status": "error", "message": "The content is inappropriate."}, 400
            answer = conversation_with_summaries_big.predict(input=alreadyTrainedQueryStr)

        if "[NO ANSWER]" in answer:
            # remove the [NO ANSWER] from the answer
            answer = answer.replace("[NO ANSWER]", "")
            # if the answer is still empty, then just say that you don't know
            if answer.strip() == "":
                answer = "I apologize, but I don't know the answer to that with the information I have."

        logger.info(query + ": " + answer)

        # add the AI's answer to the conversation history
        # memory.save_context(inputs={"input": query}, outputs={"output": answer})
        # memory.prune()

        memory.save_context(inputs={"input": "[SHOWN-IN-CHAT]" + query}, outputs={"output": "[SHOWN-IN-CHAT]" + answer})
        conversationHistory: List[BaseMessage]= memory.load_memory_variables(inputs=None)["history"]

        # combine the conversation history from OLD_MEMORY and the new conversation history
        # OLD_CONVERSATION_HISTORY = OLD_MEMORY.load_memory_variables(inputs=None)["history"]


        # oldConversationPlusOnlyLatestMessages = OLD_CONVERSATION_HISTORY + conversationHistory[-2:]


        # removing system messages
        # oldConversationPlusOnlyLatestMessages = [msg for msg in oldConversationPlusOnlyLatestMessages if msg.type != "system"]

        conversationWithoutSystemMessages = [msg for msg in conversationHistory if msg.type != "system"]
        conversationConvertedToNativeMsgObjects = messages_to_dict(conversationWithoutSystemMessages)

        # for i, msg in enumerate(conversationConvertedToNativeMsgObjects[:-2]):
        #     if "[HIDDEN-IN-CHAT]" not in msg["data"]["content"]:
        #         conversationConvertedToNativeMsgObjects[i]["data"]["content"] = "[HIDDEN-IN-CHAT]" + msg["data"]["content"]

        # save the conversation history to the database
        # db["chats"].update_one({"_id": chatID}, {"$set": {"conversation": memory.chat_memory.messages}})
        # db["summaries"].update_one({"_id": chatID}, {"$set": {"summary": memory.moving_summary_buffer}})

        # save the conversation history to the database
        # nativeMsgObjectsForLatestMessages = messages_to_dict(oldConversationPlusOnlyLatestMessages)

        # get summary
        memorySummary = memory.moving_summary_buffer

        visibleMessages = [copy.deepcopy(msg) for msg in conversationConvertedToNativeMsgObjects if "[SHOWN-IN-CHAT]" in msg["data"]["content"]]
        for i, msg in enumerate(visibleMessages):
            visibleMessages[i]["data"]["content"] = msg["data"]["content"].replace("[SHOWN-IN-CHAT]", "")


        # insert or update the chat history
        db["chats"].update_one({"_id": chatID}, {"$set": {"messages": conversationConvertedToNativeMsgObjects, "summary": memorySummary}}, upsert=True)

        return {"answer": answer, "summary": memory.moving_summary_buffer, "history": visibleMessages}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
