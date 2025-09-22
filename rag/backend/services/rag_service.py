import os
import asyncio
from typing import List, Dict, Any, Optional
import json
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from models.schemas import DocumentSource, ChatMessage

# Load .env automatically
load_dotenv()

class RAGService:
    """Retrieval-Augmented Generation service using Gemini Flash and ChromaDB"""
    
    def __init__(self, gemini_model: Optional[str] = None):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model_name = gemini_model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
        
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.genai_model = None
        
    async def initialize(self):
        """Initialize Gemini and ChromaDB"""
        if not self.gemini_api_key:
            print("ERROR: GEMINI_API_KEY not found. Make sure it is set in .env or environment variables.")
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        print(f"Initializing Gemini model: {self.gemini_model_name}")
        genai.configure(api_key=self.gemini_api_key)
        self.genai_model = genai.GenerativeModel(self.gemini_model_name)
        
        print("Initializing embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Initializing ChromaDB...")
        await self._initialize_chromadb()
        
    async def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=Settings(
                    chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                    chroma_client_auth_credentials_provider="chromadb.auth.basic.BasicAuthCredentialsProvider"
                )
            )
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Connected to ChromaDB at {self.chroma_host}:{self.chroma_port}")
        except Exception as e:
            print(f"ChromaDB HTTP connection failed: {e}, falling back to persistent client...")
            self.client = chromadb.PersistentClient(path="./chroma_data")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            print("Using persistent ChromaDB client at ./chroma_data")
    
    async def add_document(self, processed_doc: Dict[str, Any]):
        """Add a processed document to the vector store"""
        try:
            file_id = processed_doc["file_id"]
            chunks = processed_doc["chunks"]
            metadata = processed_doc["metadata"]

            # Generate embeddings for chunks
            embeddings = self.embedding_model.encode(chunks).tolist()

            # Prepare data for ChromaDB
            ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    **metadata,
                    "file_id": file_id,
                    "filename": processed_doc["filename"],
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "id": f"{file_id}_chunk_{i}"  # store ID explicitly
                }
                for i in range(len(chunks))
            ]

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )

            print(f"Added {len(chunks)} chunks for document {processed_doc['filename']}")

        except Exception as e:
            print(f"Error adding document to vector store: {e}")
            raise    
    async def search_documents(self, query: str, max_results: int = 5) -> List[DocumentSource]:
        """Search for relevant documents using vector similarity"""
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            document_sources = []

            if results["documents"] and results["documents"][0]:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    # Deserialize JSON strings back to list/dict if needed
                    for k, v in metadata.items():
                        if isinstance(v, str):
                            try:
                                parsed = json.loads(v)
                                if isinstance(parsed, (list, dict)):
                                    metadata[k] = parsed
                            except:
                                pass
                    source = DocumentSource(
                        file_id=metadata["file_id"],
                        filename=metadata["filename"],
                        content=doc,
                        metadata=metadata,
                        relevance_score=1 - distance
                    )
                    document_sources.append(source)

            return document_sources

        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    async def generate_response(
        self, 
        query: str, 
        documents: List[DocumentSource], 
        chat_history: List[ChatMessage] = None
    ) -> str:
        """Generate a response using Gemini with retrieved documents"""
        try:
            context_parts = []
            for doc in documents[:3]:
                context_parts.append(
                    f"Document: {doc.filename}\n"
                    f"Content: {doc.content[:1000]}...\n"
                    f"Relevance: {doc.relevance_score:.3f}\n"
                )
            context = "\n".join(context_parts)

            history_text = ""
            if chat_history:
                for msg in chat_history[-5:]:
                    history_text += f"{msg.role}: {msg.content}\n"

            prompt = f"""
You are an intelligent assistant that analyzes documents and performs calculations or data analysis as needed.

Based on the following context and chat history, answer the user's question accurately and helpfully.

Chat History:
{history_text}

Context from Documents:
{context}

User Question: {query}

Instructions:
1. Directly answer the user's question.
2. If the question requires mathematical calculation (e.g., average, sum, percentages), perform it step by step.
3. Reference specific information from the documents when relevant.
4. Suggest visualizations or further analysis if applicable.
5. Provide answers in a clear, concise, and well-structured format.

Answer:
"""

            response = await asyncio.create_task(self._generate_gemini_response(prompt))
            return response

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

    async def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        try:
            response = await asyncio.to_thread(self.genai_model.generate_content, prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            raise

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store"""
        try:
            results = self.collection.get(include=["metadatas"])
            documents = {}
            for metadata in results["metadatas"]:
                file_id = metadata["file_id"]
                if file_id not in documents:
                    documents[file_id] = {
                        "file_id": file_id,
                        "filename": metadata["filename"],
                        "file_type": metadata.get("type", "unknown"),
                        "chunk_count": metadata.get("chunk_count", 0),
                        "file_size": metadata.get("file_size", 0),
                        "processing_time": metadata.get("processing_time", 0)
                    }
            return list(documents.values())
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    async def delete_document(self, file_id: str) -> bool:
        """Delete a document and all its chunks from ChromaDB and disk"""
        try:
            # 1️⃣ Get all metadata for this file
            results = self.collection.get(include=["metadatas"])
            if not results["metadatas"]:
                print(f"No documents in collection.")
                return False

            # 2️⃣ Filter chunks for this file_id
            ids_to_delete = []
            filename_to_delete = None
            for meta in results["metadatas"]:
                if meta.get("file_id") == file_id:
                    ids_to_delete.append(meta["id"])
                    filename_to_delete = meta.get("filename", None)

            if not ids_to_delete:
                print(f"No chunks found for document {file_id}")
                return False

            # 3️⃣ Delete chunks from ChromaDB
            self.collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks from ChromaDB for document {file_id}")

            # 4️⃣ Delete file from disk
            if filename_to_delete:
                file_extension = os.path.splitext(filename_to_delete)[1]
                file_path = os.path.join("uploads", f"{file_id}{file_extension}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file from disk: {file_path}")

            return True

        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            try:
                # ChromaDB client does not require explicit cleanup
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
