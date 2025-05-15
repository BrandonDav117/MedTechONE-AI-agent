import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import PyPDF2
from openai import AsyncOpenAI
from supabase import create_client, Client
from crawl_MedTechONE_docs import chunk_text  # Import chunk_text

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Refresh Supabase schema cache
supabase.table("pdf_documents").select("*").limit(1).execute()

@dataclass
class ProcessedPDFChunk:
    title: str
    file_path: str
    chunk_number: int
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    associated_url: str
    ecr_metadata: Dict[str, Any]  # New field for ECR-specific metadata

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_pdf_chunks(pdf_path: str, associated_url: str) -> List[ProcessedPDFChunk]:
    """Process a PDF into text chunks with embeddings."""
    content = extract_text_from_pdf(pdf_path)
    title = os.path.splitext(os.path.basename(pdf_path))[0]
    chunks = chunk_text(content)
    processed_chunks = []
    
    # Extract ECR-specific metadata from the content
    ecr_metadata = {
        "development_stage": "unknown",  # Will be updated based on content analysis
        "device_type": "unknown",       # Will be updated based on content analysis
        "complexity_level": "medium",    # Default complexity
        "estimated_time": "unknown",     # Will be updated based on content analysis
        "prerequisites": [],            # List of prerequisites
        "common_pitfalls": [],          # List of common pitfalls
        "key_milestones": []            # List of key milestones
    }
    
    # Simple content analysis to determine metadata
    content_lower = content.lower()
    if any(stage in content_lower for stage in ["concept", "ideation", "initial"]):
        ecr_metadata["development_stage"] = "concept"
    elif any(stage in content_lower for stage in ["prototype", "development", "design"]):
        ecr_metadata["development_stage"] = "prototype"
    elif any(stage in content_lower for stage in ["pre-clinical", "preclinical"]):
        ecr_metadata["development_stage"] = "pre-clinical"
    elif any(stage in content_lower for stage in ["clinical", "trial"]):
        ecr_metadata["development_stage"] = "clinical"
    
    # Determine device type
    if any(device in content_lower for device in ["diagnostic", "diagnosis"]):
        ecr_metadata["device_type"] = "diagnostic"
    elif any(device in content_lower for device in ["therapeutic", "treatment"]):
        ecr_metadata["device_type"] = "therapeutic"
    elif any(device in content_lower for device in ["monitoring", "monitor"]):
        ecr_metadata["device_type"] = "monitoring"
    
    # Determine complexity level
    if any(word in content_lower for word in ["basic", "simple", "fundamental"]):
        ecr_metadata["complexity_level"] = "low"
    elif any(word in content_lower for word in ["advanced", "complex", "sophisticated"]):
        ecr_metadata["complexity_level"] = "high"
    
    for i, chunk in enumerate(chunks):
        embedding = await get_embedding(chunk)
        metadata = {
            "source": "MedTechONE_design_docs",
            "file_size": os.path.getsize(pdf_path),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(associated_url).path,
            "chunk_size": len(chunk)
        }
        processed_chunks.append(ProcessedPDFChunk(
            title=title,
            file_path=pdf_path,
            chunk_number=i,
            content=chunk,
            metadata=metadata,
            embedding=embedding,
            associated_url=associated_url,
            ecr_metadata=ecr_metadata
        ))
    return processed_chunks

async def insert_pdf_chunk(chunk: ProcessedPDFChunk):
    """Insert a processed PDF chunk into Supabase."""
    try:
        # First check if the document already exists
        existing = supabase.table("pdf_documents").select("*").eq("associated_url", chunk.associated_url).eq("title", chunk.title).eq("chunk_number", chunk.chunk_number).execute()
        
        if existing.data:
            print(f"Document already exists: {chunk.title} (chunk {chunk.chunk_number})")
            return existing.data[0]
            
        data = {
            "title": chunk.title,
            "file_path": chunk.file_path,
            "chunk_number": chunk.chunk_number,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
            "associated_url": chunk.associated_url,
            "ecr_metadata": chunk.ecr_metadata
        }
        print("Attempting to insert the following data:")
        print(json.dumps(data, indent=2, default=str)[:1000])
        result = supabase.table("pdf_documents").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for PDF {chunk.title} ({chunk.associated_url})")
        print(f"Supabase response: {result}")
        return result
    except Exception as e:
        print(f"Error inserting PDF chunk: {e}")
        import traceback
        traceback.print_exc()
        return None

async def process_pdf_directory(directory: str, url_mapping: Dict[str, List[str]]):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            associated_url = None
            for url, patterns in url_mapping.items():
                for pattern in patterns:
                    if pattern.lower() in filename.lower():
                        associated_url = url
                        break
                if associated_url:
                    break
            if associated_url:
                chunks = await process_pdf_chunks(pdf_path, associated_url)
                for chunk in chunks:
                    await insert_pdf_chunk(chunk)
            else:
                print(f"No matching URL found for PDF: {filename}")

async def main():
    # Define the directory containing PDFs
    pdf_directory = "design_docs"
    
    # Create URL mapping based on filename patterns
    url_mapping = {
        "https://medtechone-learning.com/pre-clinical-trials": ["pre-clinical", "Pre-Clinical"],
        "https://medtechone-learning.com/clinical-trials": ["clinical trials", "Clinical-Trials"],
        "https://medtechone-learning.com/data": ["data", "Data"],
        "https://medtechone-learning.com/usability": ["usability", "Usability"],
        "https://medtechone-learning.com/software": ["software", "Software"],
    }
    
    # Process all PDFs
    await process_pdf_directory(pdf_directory, url_mapping)

if __name__ == "__main__":
    asyncio.run(main()) 