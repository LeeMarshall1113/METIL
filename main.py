import openai
import os
import PyPDF2
import numpy as np
import faiss
import tiktoken

###############################################################################
# TOKEN MANAGEMENT HELPERS
###############################################################################

DEFAULT_MODEL = "gpt-3.5-turbo-16k"
MAX_CONTEXT_TOKENS = 16000  

def count_message_tokens(message, model=DEFAULT_MODEL):
    """
    Counts the approximate tokens for a single message dict {'role': ..., 'content': ...}
    using the tiktoken library.
    """
    encoding = tiktoken.encoding_for_model(model)
    # For chat models, we approximate tokens as:
    # Each message: "<im_start>{role}\n{content}<im_end>\n" plus some overhead
    # We'll do a simple approach: role + content
    text = f"{message['role']}: {message['content']}"
    tokens = len(encoding.encode(text))
    # Add ~4 tokens overhead as recommended by OpenAI for chat models
    tokens += 4
    return tokens

def count_messages_token_total(messages, model=DEFAULT_MODEL):
    """Returns the total token count for the entire list of messages."""
    total_tokens = 0
    for msg in messages:
        total_tokens += count_message_tokens(msg, model=model)
    return total_tokens

def summarize_messages(messages, model=DEFAULT_MODEL):
    """
    Calls the OpenAI API to summarize a list of messages into a short paragraph.
    """
    summary_prompt = [
        {"role": "system", "content": "You are a concise assistant. Summarize the following conversation in 1-2 sentences."},
        {
            "role": "user",
            "content": "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=summary_prompt,
        max_tokens=100,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def prepare_messages_for_chat_completion(messages, model=DEFAULT_MODEL, max_tokens=MAX_CONTEXT_TOKENS):
    """
    Ensures we don't exceed the model's context window by:
      1) Keeping as many recent messages as possible,
      2) Summarizing older messages into one short system message if needed.
    """
    # First, see if everything fits
    total_tokens = count_messages_token_total(messages, model)
    if total_tokens <= max_tokens:
        return messages  # all good

    # If we exceed, keep the newest messages until we hit the limit
    reversed_msgs = list(reversed(messages))  # newest -> oldest
    kept_msgs = []
    current_token_count = 0

    for msg in reversed_msgs:
        msg_tokens = count_message_tokens(msg, model)
        if current_token_count + msg_tokens <= max_tokens:
            kept_msgs.append(msg)
            current_token_count += msg_tokens
        else:
            break

    kept_msgs.reverse()  # restore chronological order
    # Summarize the messages we dropped
    num_dropped = len(messages) - len(kept_msgs)
    if num_dropped > 0:
        older_msgs = messages[:num_dropped]
        short_summary = summarize_messages(older_msgs, model=model)
        summary_message = {
            "role": "system",
            "content": f"Summary of older conversation:\n{short_summary}"
        }
        summary_tokens = count_message_tokens(summary_message, model=model)

        # If the summary fits, prepend it; if not, skip it entirely
        if summary_tokens + current_token_count <= max_tokens:
            kept_msgs.insert(0, summary_message)
    
    # Final check if we're still too big (rare, but possible)
    final_count = count_messages_token_total(kept_msgs, model=model)
    while final_count > max_tokens and len(kept_msgs) > 1:
        kept_msgs.pop(0)  # drop the oldest in kept
        final_count = count_messages_token_total(kept_msgs, model=model)

    return kept_msgs


###############################################################################
# ORIGINAL CODE WITH TOKEN LIMIT ADDED
###############################################################################

# Step 1: Extract Text from PDFs in a Folder
def extract_text_from_pdfs_in_folder(folder_path):
    all_text = []
    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.pdf')
    ]
    for file in pdf_files:
        with open(file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
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
            embeddings.append([0]*1536)  # fallback to a zero-vector
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

# Step 6: Generate Response with GPT (with context-window management)
def generate_response(conversation_history):
    """
    Generates a response from the conversation history, ensuring we don't exceed context.
    """
    # 1) Prepare the messages so we don't exceed the model's context
    safe_history = prepare_messages_for_chat_completion(
        messages=conversation_history, 
        model=DEFAULT_MODEL,
        max_tokens=MAX_CONTEXT_TOKENS
    )

    # 2) Call ChatCompletion with the truncated/summarized history
    response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        messages=safe_history,
        max_tokens=300,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Chat Workflow
def start_conversation(index, stored_chunks):
    conversation_history = [
        {
            "role": "system", 
            "content": "You are an AI assistant that answers questions based on the content of uploaded PDFs."
        }
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
        
        # Add context + user query as a new user message
        conversation_history.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuery:\n{user_query}"
        })
        
        # Generate AI response
        response_text = generate_response(conversation_history)
        
        # Append AI response to conversation
        conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        print(f"AI: {response_text}")

# Main Workflow
def main():
    # Put your OpenAI key here
    openai.api_key = "your api key"
    
    # Path to folder containing PDF files
    folder_path = "your folder path"  # Replace with your actual folder path
    
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
