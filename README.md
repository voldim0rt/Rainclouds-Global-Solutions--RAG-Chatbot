# Rainclouds-Global-Solutions-Assignment


# **System Security Plan (SSP) QA Chatbot**

This project builds a **QA Chatbot** for answering queries about **System Security Plan (SSP)** documents. It processes SSP content, stores it in a vector database, and uses a combination of retrieval and large language models (LLMs) to provide accurate answers to user queries.

---

## **Features**
1. **Data Preprocessing**:
   - Splits SSP content into manageable chunks.
   - Generates embeddings using open-source models.
   - Stores embeddings and metadata in a **Postgres** database with the **PGVector** extension.

2. **Intelligent Retrieval**:
   - Uses **semantic similarity search** with embeddings.
   - Supports additional retrieval methods like BM25 for enhanced accuracy.

3. **QA Chatbot**:
   - Leverages open-source LLMs (e.g., GPT4All, Dolly, or Llama) for answering questions.
   - Provides context-aware, detailed answers.

4. **FastAPI Integration**:
   - A REST API for seamless interaction.
   - Endpoint to query the chatbot and retrieve answers.

5. **Logging**:
   - Logs queries, retrieved context, and LLM responses for evaluation.

---

## **Tech Stack**
1. **Backend**:
   - Python
   - FastAPI
2. **Database**:
   - PostgreSQL with PGVector
3. **Embeddings Model**:
   - `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
4. **LLM**:
   - Open-source models like GPT4All, Dolly, or Llama.
5. **Retrieval Techniques**:
   - Vector similarity search
   - BM25 (using `rank-bm25`)

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/ssp-qa-chatbot.git
cd ssp-qa-chatbot
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Set Up PostgreSQL with PGVector**
- Install PostgreSQL and enable the PGVector extension:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- Create the database and table:
  ```sql
  CREATE TABLE ssp_vectors (
      id SERIAL PRIMARY KEY,
      chunk_id INT,
      content TEXT,
      embedding VECTOR(384)
  );
  ```

### **4. Preprocess Data**
- Update the `ssp_data` variable in the script with your SSP document content.
- Run the preprocessing and embedding script:
  ```bash
  python preprocess_and_store.py
  ```

### **5. Run the FastAPI App**
```bash
uvicorn app:app --reload
```

---

## **API Endpoints**

### **POST /ask**
**Description**: Query the chatbot and get an answer.

**Request**:
```json
{
  "question": "What is the AC-01 Control Policy?"
}
```

**Response**:
```json
{
  "question": "What is the AC-01 Control Policy?",
  "context": [
    {"content": "AC-01 Control Policy: This defines access control mechanisms for systems.", "similarity": 0.85}
  ],
  "answer": "AC-01 Control Policy defines access control mechanisms ensuring secure system access."
}
```

---

## **Usage Example**

### **Ask a Question**
1. Run the FastAPI server.
2. Use tools like `Postman`, `cURL`, or a web client to query the chatbot:
   ```bash
   curl -X POST http://127.0.0.1:8000/ask \
   -H "Content-Type: application/json" \
   -d '{"question": "What is the AC-01 Control Policy?"}'
   ```

### **Logs**
Check `ssp_qa.log` for logged queries, retrieved context, and answers.

---

## **Optional Enhancements**
1. **Hybrid Retrieval**:
   - Combine semantic search with BM25 for better relevance.
2. **OpenAI Integration**:
   - Use `text-embedding-ada-002` for embeddings and GPT-4 for LLM-based answers.
3. **Frontend**:
   - Develop a simple React or HTML interface for user-friendly interaction.

---

## **License**
This project is licensed under the MIT License.
