# Offline Resume RAG Chatbot

An offline, privacy-first chatbot that can read and understand resumes from PDF files and answer queries about them. Built using LangChain, Hugging Face’s Transformers, and Streamlit.

## Features

* **Offline Processing** – No API keys or internet connection required after setup.
* **RAG (Retrieval-Augmented Generation)** – Accurately retrieves relevant sections from the resume before generating answers.
* **PDF Resume Parsing** – Extracts structured text from PDF resumes.
* **Local LLM Model** – Uses a locally stored Hugging Face model for responses.

## Tech Stack

* **LangChain Community** – Document loaders and retrieval pipeline.
* **Hugging Face Transformers** – Local language model inference.
* **FAISS** – Vector store for efficient similarity search.
* **Streamlit** – User-friendly web interface.

## Installation

```powershell
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install streamlit langchain-community transformers faiss-cpu pypdf
```

## Usage

```powershell
# Run the app
streamlit run app.py
```

## How It Works

1. **Load Resume PDF** – Uses LangChain’s `PyPDFLoader` to extract text.
2. **Chunk the Text** – Splits resume into small sections for better retrieval.
3. **Embed the Text** – Converts chunks into vector embeddings using a local embedding model.
4. **Store in FAISS** – Saves embeddings in an in-memory vector database.
5. **Retrieve & Answer** – On user query, retrieves relevant chunks and uses the LLM to generate a response.

## Example

* Upload your resume PDF.
* Ask: *"What are my Python skills?"*
* The chatbot will find the relevant section in your resume and answer accurately.

## Project Structure

```
resume_rag_chatbot/
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── venv/                 # Virtual environment
└── README.md             # Project documentation
```

## Future Improvements

* Support for multiple resumes.
* Advanced filtering and ranking.
* Export answers to PDF/Word.

## License

MIT License – Free to use and modify.

## Author

Developed by Harish Krish – AI & ML Enthusiast.
