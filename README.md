# METIL
PDF AI Assistant

This project is a Python-based AI assistant that extracts information from  multiple PDF files in a folder, processes the text into chunks, generates embeddings using OpenAI's API, and stores them in a FAISS index for efficient querying. The assistant then uses GPT-3.5 (or GPT-4) to answer user questions based on the content of the PDFs.

Features

PDF Text Extraction: Extracts text from PDF files in a specified folder.

Text Chunking: Breaks the extracted text into manageable chunks with overlap for context preservation.

Embedding Generation: Uses OpenAI's text-embedding-ada-002 model to generate embeddings for the chunks.

Efficient Storage: Stores embeddings in a FAISS index for fast similarity search.

Interactive Chat: Allows users to query the PDF content and receive AI-generated answers.

Prerequisites

Python 3.8+

OpenAI API key (required for embeddings and GPT model access)

Required Python libraries:

openai

PyPDF2

numpy

faiss (FAISS must be installed via faiss-cpu or faiss-gpu depending on your setup)

Installation

Clone this repository or download the script.

Install the required dependencies:

pip install openai PyPDF2 numpy faiss-cpu

If you have GPU support, use faiss-gpu instead:

pip install faiss-gpu

Obtain your OpenAI API key from OpenAI.

Usage

Place your PDF files in a folder.

Update the folder_path and openai.api_key variables in the main function:

openai.api_key = "your api key here"
folder_path = "your folder path"

Run the script:

python script_name.py

Follow the prompts in the console to interact with the AI assistant.

Workflow

Extract Text: The script reads all PDFs in the specified folder and extracts the text.

Chunk Text: Text is split into chunks of 1000 characters with a 200-character overlap for better context retention.

Generate Embeddings: Each chunk is embedded using OpenAI's embedding model.

Store in FAISS: Embeddings are indexed using FAISS for fast similarity searches.

Interactive Chat: The user inputs questions, and the assistant retrieves relevant chunks and generates responses using OpenAI's GPT model.

Example

Start chatting with the AI. Type 'exit' to end the conversation.

You: What are the main points of the document?
AI: The document discusses...

You: Summarize the second section.
AI: The second section focuses on...

Troubleshooting

OpenAI API Key Error: Ensure your API key is valid and correctly assigned in the script.

Missing Dependencies: Install required libraries using pip install.

FAISS Errors: Ensure you have the correct version of FAISS (faiss-cpu or faiss-gpu) installed.

Customization

Chunk Size and Overlap: Modify chunk_size and overlap in the chunk_text_with_overlap function to adjust text chunking.

Model Selection: Change the embedding model (text-embedding-ada-002) or GPT model (gpt-3.5-turbo) to fit your requirements.

Query Results: Adjust the top_k parameter in the query_data function to retrieve more or fewer chunks.

License

This project is provided under the Apache 2.0 License. Feel free to use and modify it as needed.

