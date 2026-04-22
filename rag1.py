import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all txt files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileExistsError(f"The directory {docs_path} does not exists. Please create it and add your company files.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileExistsError(f"No .txt files found in {docs_path}. Please add your company documents")
    
    for i, doc in enumerate(documents[:3]):
        print(f"\nDocument {i+1}:")
        print(f" Suurce: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into small chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n---- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    print("--- Finished creating vector store")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def LLM(vectorstore):
    query = "Who is the CEO of Tesla?"
    retriever = vectorstore.as_retriever(search_kwargs={"k":5})
    relevant_docs = retriever.invoke(query)

    print(f"User Query: {query}")
    # Display results
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

    combined_input = f"""Based on the following documents, please answer this question: {query}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o")

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = model.invoke(messages)

    # Display the full result and content only
    print("\n--- Generated Response ---")
    # print("Full result:")
    # print(result)
    print("Content only:")
    print(result.content)

def main():
    print("Main Function")

    documents = load_documents(docs_path="docs")
    # print(documents)

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks, persist_directory="db/chroma_db")
    
    LLM(vectorstore)

if __name__ == "__main__":
    main()