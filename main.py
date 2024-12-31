import openai
import os
import PyPDF2
import numpy as np
import faiss

# Step 1: Extract Text from PDFs in a Folder
def extract_text_from_pdfs_in_folder(folder_path):
    all_text = []
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for file in pdf_files:
        with open(file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            all_text.append(text)
    return all_text

# Step 2: Chunk Text with Overlap
def chunk_text_with_overlap(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# Step 3: Get Embeddings from OpenAI API
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def embed_chunks(chunks):
    embeddings = []
    for idx, chunk in enumerate(chunks):
        try:
            print(f"Embedding chunk {idx + 1}/{len(chunks)}...")
            embeddings.append(get_embedding(chunk))
        except openai.error.OpenAIError as e:
            print(f"Error embedding chunk {idx + 1}: {e}")
    return np.array(embeddings)

# Step 4: Store Embeddings in FAISS
def store_embeddings(embeddings, chunks):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

# Step 5: Query Data
def query_data(query, index, chunks, top_k=5):
    query_embedding = np.array([get_embedding(query)])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# Step 6: Generate Response with GPT
def generate_response(conversation_history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=conversation_history,
        max_tokens=300,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Chat Workflow
def start_conversation(index, stored_chunks):
    conversation_history = [
        {"role": "system", "content": "You are an AI assistant that answers questions based on the content of uploaded PDFs."}
    ]
    
    print("Start chatting with the AI. Type 'exit' to end the conversation.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == "exit":
            print("Ending conversation. Goodbye!")
            break
        
        # Retrieve top relevant chunks
        top_results = query_data(user_query, index, stored_chunks, top_k=10)
        context = "\n".join(top_results)
        
        # Add context and user query to conversation
        conversation_history.append({"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{user_query}"})
        
        # Generate AI response
        response = generate_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})
        
        print(f"AI: {response}")

# Main Workflow
def main():
    #put your openai key here
    openai.api_key = "your api key here"
    
    # Path to folder containing PDF files
    folder_path = "your folder path"  # Replace with your folder path
    
    # Extract and process PDFs
    print("Extracting text from PDFs...")
    texts = extract_text_from_pdfs_in_folder(folder_path)
    
    # Chunk Text with Overlap
    print("Chunking text...")
    chunks = []
    for text in texts:
        chunks.extend(chunk_text_with_overlap(text, chunk_size=1000, overlap=200))
    
    # Embed Chunks
    print("Embedding chunks...")
    embeddings = embed_chunks(chunks)
    
    # Store Embeddings in FAISS
    print("Storing embeddings...")
    index, stored_chunks = store_embeddings(embeddings, chunks)
    
    # Start Chat
    start_conversation(index, stored_chunks)

if __name__ == "__main__":
    main()
