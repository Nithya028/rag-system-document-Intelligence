'''
from fastapi import FastAPI, UploadFile, File, Form
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
from typing import List, Dict, Any
import asyncio
from contextlib import asynccontextmanager

from services.document_processor import DocumentProcessor
from services.rag_service import RAGService
from services.visualization_service import VisualizationService
from models.schemas import ChatRequest, ChatResponse, UploadResponse, AnalyticsRequest

# Global services
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
document_processor = None
rag_service = None
visualization_service = None

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_processor, rag_service, visualization_service

    document_processor = DocumentProcessor()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")

    # Initialize services
    rag_service = RAGService(gemini_model=os.getenv("GEMINI_MODEL"))
    visualization_service = VisualizationService(gemini_api_key=gemini_api_key)

    await rag_service.initialize()

    yield

    if rag_service:
        await rag_service.cleanup()

app = FastAPI(
    title="RAG System API",
    description="A comprehensive RAG system for document analysis and insights",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG System API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {
        "document_processor": bool(document_processor),
        "rag_service": bool(rag_service),
        "visualization_service": bool(visualization_service)
    }}

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents of any format"""
    try:
        processed_files = []

        for file in files:
            # Generate unique ID and filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{file_id}{file_extension}"
            file_path = os.path.join("uploads", unique_filename)

            # Save the file
            os.makedirs("uploads", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Process the document according to its type
            processed_doc = await document_processor.process_document(file_path, file.filename)
            processed_doc["file_id"] = file_id

            # Ensure metadata values are serializable for ChromaDB
            metadata = processed_doc.get("metadata", {})
            for k, v in metadata.items():
                if isinstance(v, list) or isinstance(v, dict):
                    metadata[k] = str(v)  # convert lists/dicts to string

            processed_doc["metadata"] = metadata

            # Add document to vector store
            await rag_service.add_document(processed_doc)

            processed_files.append(processed_doc)

        return UploadResponse(
            success=True,
            message=f"Successfully processed {len(processed_files)} files",
            files=processed_files
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get relevant documents
        relevant_docs = await rag_service.search_documents(request.query)
        
        # Generate response
        response = await rag_service.generate_response(
            query=request.query,
            documents=relevant_docs,
            chat_history=request.chat_history
        )
        
        # Check if visualization is needed
        visualization = None
        if request.include_visualization:
            viz = await visualization_service.generate_visualization(
                query=request.query,
                documents=relevant_docs,
                response=response
            )
            # Ensure it's either dict or None
            if isinstance(viz, dict):
                visualization = viz
            else:
                visualization = None  # fallback if not proper dict
        
        return ChatResponse(
            response=response,
            sources=relevant_docs,
            visualization=visualization,
            timestamp=None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    """Generate advanced analytics and insights"""
    try:
        # Search for relevant documents
        relevant_docs = await rag_service.search_documents(request.query)
        
        # Generate analytics
        analytics = await visualization_service.generate_analytics(
            query=request.query,
            documents=relevant_docs,
            chart_types=request.chart_types
        )
        
        return JSONResponse(content=analytics)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        documents = await rag_service.list_documents()
        return {"documents": documents}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """Delete a specific document"""
    try:
        print(f"Attempting to delete document: {file_id}")  # DEBUG
        success = await rag_service.delete_document(file_id)
        print(f"Delete success: {success}")  # DEBUG
        
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
@app.post("/visualize")
async def visualize(
    query: str = Form(...),
    documents: List[UploadFile] = File(None)
):
    try:
        # Convert uploaded docs into DocumentSource objects
        doc_sources = []
        if documents:
            for doc in documents:
                # ⚠️ Example: you probably already have preprocessing logic
                doc_sources.append({
                    "metadata": {"type": "csv", "processed_path": f"processed/{doc.filename}"}
                })

        # Generate visualization
        chart_base64 = await visualization_service.generate_visualization(
            query=query,
            documents=doc_sources,
            response="",
            rag_service=rag_service
        )

        return {"query": query, "chart_base64": chart_base64}

    except Exception as e:
        return {"error": str(e)}
@app.post("/visualization")
async def visualize_alias(
    query: str = Form(...),
    documents: List[UploadFile] = File(None)
):
    return await visualize(query=query, documents=documents)

@app.post("/analytics")
async def analytics(
    query: str = Form(...),
    chart_types: str = Form("bar"),
    documents: List[UploadFile] = File(None)
):
    try:
        # Convert to list
        chart_types_list = chart_types.split(",")

        doc_sources = []
        if documents:
            for doc in documents:
                doc_sources.append({
                    "metadata": {"type": "csv", "processed_path": f"processed/{doc.filename}"}
                })

        summary = await visualization_service.generate_analytics(
            query=query,
            documents=doc_sources,
            chart_types=chart_types_list
        )

        return summary

    except Exception as e:
        return {"error": str(e)}

'''

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
from typing import List, Dict, Any
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from services.document_processor import DocumentProcessor
from services.rag_service import RAGService
from services.visualization_service import VisualizationService
from models.schemas import ChatRequest, ChatResponse, UploadResponse, AnalyticsRequest, VisualizationRequest

# Load environment variables
load_dotenv()

# Global services
document_processor = None
rag_service = None
visualization_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_processor, rag_service, visualization_service

    # Get API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")

    # Initialize services
    document_processor = DocumentProcessor()
    rag_service = RAGService()
    visualization_service = VisualizationService(gemini_api_key=gemini_api_key)

    # Initialize RAG service
    await rag_service.initialize()

    yield

    # Cleanup
    if rag_service:
        await rag_service.cleanup()

app = FastAPI(
    title="RAG System API",
    description="A comprehensive RAG system for document analysis and insights",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG System API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {
        "document_processor": bool(document_processor),
        "rag_service": bool(rag_service),
        "visualization_service": bool(visualization_service)
    }}

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        processed_files = []

        for file in files:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{file_id}{file_extension}"
            file_path = os.path.join("uploads", unique_filename)

            # Save file
            os.makedirs("uploads", exist_ok=True)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Process document
            processed_doc = await document_processor.process_document(file_path, file.filename)
            processed_doc["file_id"] = file_id

            # Clean metadata for ChromaDB
            metadata = processed_doc.get("metadata", {})
            for k, v in metadata.items():
                if isinstance(v, (list, dict)):
                    metadata[k] = str(v)
            processed_doc["metadata"] = metadata

            # Add to vector store
            await rag_service.add_document(processed_doc)
            processed_files.append(processed_doc)

        return UploadResponse(
            success=True,
            message=f"Successfully processed {len(processed_files)} files",
            files=processed_files
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat queries and generate responses"""
    try:
        # Get relevant documents
        relevant_docs = await rag_service.search_documents(request.query)
        
        # Generate response
        response = await rag_service.generate_response(
            query=request.query,
            documents=relevant_docs,
            chat_history=request.chat_history
        )
        
        # Check if visualization is needed
        visualization = None
        if request.include_visualization:
            try:
                chart_base64 = await visualization_service.generate_visualization(
                    query=request.query,
                    documents=relevant_docs,
                    response=response
                )
                
                if chart_base64 and isinstance(chart_base64, str) and chart_base64.startswith("data:image"):
                    visualization = {
                        "type": "image",
                        "data": chart_base64,
                        "description": f"Visualization for: {request.query}"
                    }
                elif chart_base64 and not chart_base64.startswith("No") and not chart_base64.startswith("Failed"):
                    visualization = {
                        "type": "image", 
                        "data": f"data:image/png;base64,{chart_base64}",
                        "description": f"Visualization for: {request.query}"
                    }
            except Exception as e:
                print(f"Visualization error: {e}")
                visualization = None
        
        return ChatResponse(
            response=response,
            sources=relevant_docs,
            visualization=visualization,
            timestamp=None
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/visualization")
async def generate_visualization(request: VisualizationRequest):
    """Generate visualization based on query"""
    try:
        # Search for relevant documents
        relevant_docs = await rag_service.search_documents(request.query)
        
        if not relevant_docs:
            return {"error": "No relevant documents found for visualization"}
        
        # Generate visualization
        chart_base64 = await visualization_service.generate_visualization(
            query=request.query,
            documents=relevant_docs,
            response=""
        )
        
        if chart_base64 and isinstance(chart_base64, str):
            if chart_base64.startswith("No") or chart_base64.startswith("Failed") or chart_base64.startswith("Error"):
                return {"error": chart_base64}
            
            # Ensure proper base64 format
            if not chart_base64.startswith("data:image"):
                chart_base64 = f"data:image/png;base64,{chart_base64}"
            
            return {
                "query": request.query,
                "chart_base64": chart_base64,
                "description": f"Visualization for: {request.query}"
            }
        else:
            return {"error": "No visualization generated"}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    """Generate advanced analytics and insights"""
    try:
        # Search for relevant documents
        relevant_docs = await rag_service.search_documents(request.query)
        
        if not relevant_docs:
            return {"error": "No relevant documents found for analytics"}
        
        # Generate analytics
        analytics = await visualization_service.generate_analytics(
            query=request.query,
            documents=relevant_docs,
            chart_types=request.chart_types
        )
        
        return JSONResponse(content=analytics)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        documents = await rag_service.list_documents()
        return {"documents": documents}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """Delete a specific document"""
    try:
        success = await rag_service.delete_document(file_id)
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

'''
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
from typing import List, Dict, Any
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from services.document_processor import DocumentProcessor
from services.rag_service import RAGService
from services.visualization_service import VisualizationService
from models.schemas import ChatRequest, ChatResponse, UploadResponse, AnalyticsRequest, VisualizationRequest

# Load environment variables
load_dotenv()

# Global services
document_processor = None
rag_service = None
visualization_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_processor, rag_service, visualization_service

    # Get API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")

    # Initialize services
    document_processor = DocumentProcessor()
    rag_service = RAGService()
    visualization_service = VisualizationService(gemini_api_key=gemini_api_key)

    # Initialize RAG service
    await rag_service.initialize()

    yield

    # Cleanup
    if rag_service:
        await rag_service.cleanup()

app = FastAPI(
    title="RAG System API",
    description="A comprehensive RAG system for document analysis and insights",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG System API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {
        "document_processor": bool(document_processor),
        "rag_service": bool(rag_service),
        "visualization_service": bool(visualization_service)
    }}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents - Fixed version"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        processed_files = []

        for file in files:
            # Validate file
            if not file.filename:
                continue
                
            # Check file size (optional - 50MB limit)
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:  # 50MB
                raise HTTPException(status_code=413, detail=f"File {file.filename} too large")
            
            # Reset file pointer
            await file.seek(0)
            
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            # Validate file extension
            allowed_extensions = {'.pdf', '.docx', '.doc', '.pptx', '.csv', '.xlsx', '.xls', '.txt', '.json'}
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}"
                )
            
            unique_filename = f"{file_id}{file_extension}"
            file_path = os.path.join("uploads", unique_filename)

            # Save file
            os.makedirs("uploads", exist_ok=True)
            
            # Read content again for saving
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # Process document
            try:
                processed_doc = await document_processor.process_document(file_path, file.filename)
                processed_doc["file_id"] = file_id

                # Clean metadata for ChromaDB compatibility
                metadata = processed_doc.get("metadata", {})
                cleaned_metadata = {}
                
                for k, v in metadata.items():
                    if v is None:
                        cleaned_metadata[k] = ""
                    elif isinstance(v, (list, dict)):
                        cleaned_metadata[k] = str(v)
                    elif isinstance(v, (int, float, str, bool)):
                        cleaned_metadata[k] = v
                    else:
                        cleaned_metadata[k] = str(v)
                
                processed_doc["metadata"] = cleaned_metadata

                # Add to vector store
                await rag_service.add_document(processed_doc)
                processed_files.append(processed_doc)
                
            except Exception as e:
                print(f"Error processing document {file.filename}: {e}")
                # Clean up the file if processing failed
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

        if not processed_files:
            raise HTTPException(status_code=400, detail="No files were successfully processed")

        return {
            "success": True,
            "message": f"Successfully processed {len(processed_files)} files",
            "files": processed_files
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat queries and generate responses"""
    try:
        # Get relevant documents
        relevant_docs = await rag_service.search_documents(request.query)
        
        # Generate response
        response = await rag_service.generate_response(
            query=request.query,
            documents=relevant_docs,
            chat_history=request.chat_history
        )
        
        # Check if visualization is needed
        visualization = None
        if request.include_visualization:
            try:
                viz_result = await visualization_service.generate_visualization(
                    query=request.query,
                    documents=relevant_docs,
                    response=response
                )
                
                if isinstance(viz_result, dict) and "type" in viz_result:
                    visualization = viz_result
                    
            except Exception as e:
                print(f"Visualization error: {e}")
                visualization = None
        
        return ChatResponse(
            response=response,
            sources=relevant_docs,
            visualization=visualization,
            timestamp=None
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/visualization")
async def generate_visualization(request: VisualizationRequest):
    """Generate visualization based on query"""
    try:
        # Search for relevant documents
        relevant_docs = await rag_service.search_documents(request.query)
        
        if not relevant_docs:
            return {
                "error": "No relevant documents found for visualization. Please upload CSV or Excel files."
            }
        
        # Generate visualization
        viz_result = await visualization_service.generate_visualization(
            query=request.query,
            documents=relevant_docs,
            response=""
        )
        
        if isinstance(viz_result, dict):
            return viz_result
        else:
            return {"error": "Failed to generate visualization"}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    """Generate advanced analytics and insights"""
    try:
        # Search for relevant documents
        relevant_docs = await rag_service.search_documents(request.query)
        
        if not relevant_docs:
            return {
                "error": "No relevant documents found for analytics. Please upload CSV or Excel files with data."
            }
        
        # Generate analytics
        analytics = await visualization_service.generate_analytics(
            query=request.query,
            documents=relevant_docs,
            chart_types=request.chart_types
        )
        
        return analytics
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        documents = await rag_service.list_documents()
        return {"documents": documents}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """Delete a specific document"""
    try:
        success = await rag_service.delete_document(file_id)
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Add a test endpoint for debugging
@app.post("/test-upload")
async def test_upload():
    """Test endpoint to verify upload functionality"""
    return {"message": "Upload endpoint is working", "status": "ready"}

'''