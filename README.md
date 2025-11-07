🩺 Medical Chatbot using LLMs, LangChain, Pinecone, and Flask
This project is a local implementation of a Medical Chatbot powered by Large Language Models (LLMs). It uses LangChain for orchestration, Pinecone for vector storage, and Flask for serving the application through a local web interface.

🚀 How to Run the Project (Locally)
STEP 1: Clone the Repository
git clone https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
cd Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

STEP 2: Create a Conda Environment
conda create -n medibot python=3.10 -y
conda activate medibot

STEP 3: Install Dependencies
pip install -r requirements.txt

STEP 4: Set Up Environment Variables
Create a .env file in the project’s root directory and add your API credentials:
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key


⚠️ These keys are required for vector storage and GPT model access.


STEP 5: Generate and Store Embeddings in Pinecone
Run the following command to index your knowledge base:
python store_index.py

STEP 6: Run the Flask Application
python app.py


🌐 Access the Chatbot
Once the server starts, open your browser and visit:
http://127.0.0.1:5000

You should see your Medical Chatbot interface running locally.

🧠 Tech Stack


Python 3.10


LangChain


Flask


OpenAI GPT Models


Pinecone (Vector Database)



🧩 Project Overview
This chatbot allows users to ask medical-related questions and receive context-aware, AI-driven responses.
The knowledge base is stored as vector embeddings in Pinecone, and LangChain manages the query-response workflow between the model and stored context.

🧪 Example Use Cases


Get explanations for medical terms or symptoms


Retrieve information from custom medical documents


Use as a foundation for clinical assistant applications



⚙️ Future Enhancements


Integrate with AWS Lambda / EC2 for cloud deployment


Add user authentication


Extend to voice-based chat or multi-language support



Would you like me to rename the project (e.g., Local-Medical-Chatbot-with-LLMs) and make it sound fully original instead of referencing the Entbappy repo?
I can also tailor the “Project Overview” and “Future Enhancements” sections based on your customizations.
