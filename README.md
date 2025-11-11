# 🩺 Medical Chatbot using LLMs, RAG, LangChain, Pinecone, and Flask

This project is a **locally hosted Medical Chatbot** powered by **Large Language Models (LLMs)**.  
It uses **LangChain** for workflow orchestration, **Pinecone** for vector storage and retrieval, and **Flask** for creating a simple web interface.

---
<img width="2486" height="2254" alt="image" src="https://github.com/user-attachments/assets/4ed84596-ef91-49f6-b947-be6c57d29967" />


## How to Run the Project (Locally)

### Step 1: Clone the Repository
- https://github.com/himanshuwarudkar9/Medical_Chatbot.git
- cd Medical-Chatbot

### Step 2: Create a Conda Environment
- conda create -n medibot python=3.10 -y
- conda activate medibot

### Step 3: Install the Dependencies
- pip install -r requirements.txt

### Step 4: Configure Environment Variables
- Create a .env file in the root directory and add your API keys:
  - PINECONE_API_KEY=your_pinecone_api_key
  - GEMINI_API_KEY=your_gemini_api_key
- These credentials are required for connecting to Pinecone and accessing OpenAI’s GPT models.

### Step 5: Generate and Store Embeddings
- Run the following command to create and upload embeddings to your Pinecone index:
  - python store_index.py

### Step 6: Run the Flask Application
- python app.py












