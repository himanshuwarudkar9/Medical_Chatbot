from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input = msg
        print(f"🔍 User Question: {input}")
        
        # Get the full response including retrieved context
        response = rag_chain.invoke({"input": msg})
        
        # Print what was retrieved (for debugging)
        retrieved_docs = response.get("context", [])
        print(f"📚 Retrieved {len(retrieved_docs)} docs from book")
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:100] + "..."
            print(f"   Doc {i}: {content_preview}")
        
        answer = response["answer"]
        
        # Add source information to the response for user
        print("Response:", answer)
        return str(answer)

    except Exception as e:
        print(f"❌ Error in chat route: {str(e)}")
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)