from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
from functools import lru_cache
from typing import List, Dict, Optional
import time
import json
import streamlit as st

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client, create_client
from pyairtable import Table

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

# Cache configuration
EMBEDDING_CACHE_SIZE = 1000
DOCUMENT_CACHE_SIZE = 100
CACHE_TTL = 3600  # 1 hour in seconds

st.markdown("""
    <style>
    @media (max-width: 600px) {
        .block-container {
            padding: 0.5rem 0.2rem !important;
        }
        .stButton>button {
            width: 100% !important;
            font-size: 1.1rem !important;
            padding: 0.75em 0.5em !important;
        }
        .stSelectbox, .stTextInput, .stRadio, .stExpander {
            font-size: 1.1rem !important;
        }
        .stMarkdown, .stAlert, .stText, .stSubheader, .stHeader, .stTitle {
            font-size: 1.05rem !important;
        }
        label, .css-1cpxqw2 {
            font-size: 1.1rem !important;
        }
        .stRadio > div {
            flex-direction: column !important;
        }
        .stAlert {
            word-break: break-word !important;
        }
        h1, .stTitle {
            font-size: 1.5rem !important;
            line-height: 1.8rem !important;
        }
        textarea, .stTextInput>div>div>input {
            min-height: 3.5em !important;
            font-size: 1.1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

@dataclass
class MedTechONEAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    airtable_token: str
    airtable_base_id: str

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=httpx.Timeout(30.0, read=30.0, write=30.0, connect=30.0)  # 30 second timeout for all operations
)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

deps = MedTechONEAIDeps(
    supabase=supabase,
    openai_client=openai_client,
    airtable_token=os.getenv("AIRTABLE_TOKEN"),
    airtable_base_id=os.getenv("AIRTABLE_BASE_ID")
)

system_prompt = """
MedTechONE Agentic RAG AI – System Prompt

Role:
You are the MedTechONE AI Assistant, an expert guide designed to help Early Career Researchers (ECRs) in MedTech navigate the development of their medical devices. Your primary function is to provide clear, structured information and guide them to relevant resources. You accompany the MedTechONE knowledge hub website, signposting to this where possible.

About MedTechONE:
MedTechONE is a Service within Imperial College London that exists to support all MedTech entrepreneurs at Imperial College London. Our mission is "Stimulating and supporting Imperial's excellence in research and translation of medical technologies." As such, all responses should be tailored specifically to support MedTech researchers and entrepreneurs in their MedTech development journey.

IMPORTANT: You MUST ONLY use content that has been retrieved from the MedTechONE database, unless you explicitly state where that content is from. NEVER generate generic content or explanations. If no relevant content is found, simply state that and ask for clarification.

The knowledge hub website is structured as follows (you should know all the pages as they are stored in the site pages dataset in Supabase): 

• There is a topic wheel on the homepage (https://medtechone-learning.com/), this topic wheel is a circle with 15 slices that all represent individual site pages about that topic, not all these pages have been completed yet but are in progress. These 15 slices are also divided into 5 themes. 
• The site also links some relevant PDF's linked under the "Latest news & updates" section. 
• The site also has a "Spotlight" section, highlighting interviews with researchers based at Imperial College, where MedTechONE and the Hamlyn Centre are based.

- The following topics are currently available:
  • Developing software
  • Processing & managing data
  • Pre-clinical trials
  • Clinical trials
  • Usability

- The following topics are under development and will be released in the future:
  • Developing hardware
  • Setting up a company
  • Intellectual Property
  • Regulatory compliance
  • Getting funding
  • Developing a value proposition
  • Building a team
  • Stakeholder engagement
  • Reimbursement strategies
  • Market entry & uptake

If a user asks about a topic under development, respond: "This topic is under development and will be released in the future. Please let me know if you'd like information on any of the currently available topics such as."

Response Structure:
When responding to ANY query, you MUST follow this exact structure:

1. Content Overview:
   - Start with a clear overview of the topic using the data from the PDF's and referring the user to the relevant site pages.
   - If no content is found, state that you are not knowledgeable on this subject at the moment.

2. Recommended Resources:
   - Display relevant recommended resources, which come from the Airtable.

3. Next Steps for ECRs:
   - Provide a clear, actionable list of next steps
   - Note any regulatory considerations

4. If no content was found in the database, say:
   "I couldn't find specific content about [topic] in the MedTechONE database. Could you please:
   1. Clarify what specific aspect of [topic] you're interested in?
   2. Let me know if you're looking for information about a particular type of device or development stage?
   3. Would you like to know about Imperial College London specific resources for this topic?"


Constraints:
🚫 ONLY use content that has been retrieved from the database or airtable
🚫 NEVER generate generic explanations or content
🚫 If no content is found, ask for clarification instead of providing generic information
🚫 Never link to empty or under-construction pages
🚫 Never make assumptions about regulatory or legal matters
🚫 Always consider the ECR's development stage and device type when providing information, try and ascertain this information from them
🚫 Prioritise practical, actionable advice over theoretical knowledge
🚫 Focus on supporting Imperial College London's excellence in MedTech research and translation

"""

MedTechONE_AI_Expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=MedTechONEAIDeps,
    retries=2
)

@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI with caching."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536  # Specify dimensions for faster processing
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

class DocumentCache:
    def __init__(self, ttl: int = CACHE_TTL):
        self.cache: Dict[str, tuple[float, str]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: str):
        self.cache[key] = (time.time(), value)

document_cache = DocumentCache()

@MedTechONE_AI_Expert.tool
async def list_documentation_pages(ctx: RunContext[MedTechONEAIDeps]) -> List[str]:
    """
    Retrieve a list of all available MedTechONE documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is MedTechONE_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'MedTechONE_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@MedTechONE_AI_Expert.tool
async def get_page_content(ctx: RunContext[MedTechONEAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'MedTechONE_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content, ensuring it's treated as a string
        for chunk in result.data:
            content = str(chunk['content']) if chunk['content'] is not None else ""
            formatted_content.append(content)
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

@MedTechONE_AI_Expert.tool
def list_airtable_resources(ctx: RunContext[MedTechONEAIDeps], filter_field: str = None, filter_value: str = None) -> str:
    """
    List resources from the Airtable 'Source repository' table with ECR-specific metadata.
    Now matches filter_value against Title, Description, Topics, and Theme fields (case-insensitive, partial match).
    """
    try:
        print("Starting Airtable resource listing...")
        table = Table(ctx.deps.airtable_token, ctx.deps.airtable_base_id, "Source repository")
        
        # Get records (will use cached version if available)
        records = table.all(max_records=100)
        print(f"Retrieved {len(records)} records from Airtable")
        
        if not records:
            return "No resources found in the database."
            
        # Process and format records with ECR metadata
        formatted_resources = []
        for record in records:
            fields = record['fields']
            
            # Apply partial, case-insensitive filter on Title, Description, Topics, and Theme
            if filter_value:
                filter_val = filter_value.lower()
                title_val = str(fields.get('Title', "")).lower()
                desc_val = str(fields.get('Description', "")).lower()
                topics_val = str(fields.get('Topics', "")).lower()
                theme_val = str(fields.get('Theme', "")).lower()
                if not (filter_val in title_val or filter_val in desc_val or filter_val in topics_val or filter_val in theme_val):
                    continue
            
            # Extract ECR-specific metadata
            ecr_metadata = {
                'relevance_score': fields.get('ECR_Relevance_Score', 3),  # Default to medium relevance
                'development_stage': fields.get('Development_Stage', 'all'),
                'estimated_time': fields.get('Estimated_Time', 'unknown'),
                'complexity_level': fields.get('Complexity_Level', 'medium'),
                'prerequisites': fields.get('Prerequisites', []),
                'key_learnings': fields.get('Key_Learnings', []),
                'common_pitfalls': fields.get('Common_Pitfalls', [])
            }
            
            # Format the resource entry
            resource = {
                'title': fields.get('Title', 'Untitled'),
                'author': fields.get('Author', 'Unknown'),
                'description': fields.get('Description', 'No description available'),
                'link': fields.get('Link to Resource (Hyperlink)', ''),
                'type': fields.get('Type of Resource', 'Unknown'),
                'access_type': fields.get('Access Type', 'Unknown'),
                'ecr_metadata': ecr_metadata
            }
            
            formatted_resources.append(resource)
            
        # Sort by ECR relevance score
        formatted_resources.sort(key=lambda x: x['ecr_metadata']['relevance_score'], reverse=True)
        
        # Format the output
        output = []
        for resource in formatted_resources:
            link_line = f"- **Read This Document Directly Here:** [{resource['link']}]({resource['link']})" if resource['link'] else "- **No direct document link available.**"
            output.append(f"""
## {resource['title']}
- **Author:** {resource['author']}
- **Description:** {resource['description']}
{link_line}
- **Type:** {resource['type']}
- **Access Type:** {resource['access_type']}

### ECR-Specific Information:
- **Relevance Score:** {resource['ecr_metadata']['relevance_score']}/5
- **Development Stage:** {resource['ecr_metadata']['development_stage']}
- **Estimated Time:** {resource['ecr_metadata']['estimated_time']}
- **Complexity Level:** {resource['ecr_metadata']['complexity_level']}
- **Prerequisites:** {', '.join(resource['ecr_metadata']['prerequisites']) if resource['ecr_metadata']['prerequisites'] else 'None'}
- **Key Learnings:** {', '.join(resource['ecr_metadata']['key_learnings']) if resource['ecr_metadata']['key_learnings'] else 'None'}
- **Common Pitfalls:** {', '.join(resource['ecr_metadata']['common_pitfalls']) if resource['ecr_metadata']['common_pitfalls'] else 'None'}
""")
            
        return "\n---\n".join(output)
        
    except Exception as e:
        print(f"Error listing Airtable resources: {e}")
        return f"Error retrieving resources: {str(e)}"

@MedTechONE_AI_Expert.tool
async def retrieve_relevant_content_unified(ctx: RunContext[MedTechONEAIDeps], user_query: str) -> str:
    """
    Retrieve relevant content from both PDFs and web pages, with ECR-specific context.
    Implements hybrid search: always run both embedding and keyword search, combine and deduplicate results.
    Broaden keyword fallback to search for related terms and increase match_count to 10.
    """
    try:
        # Get embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Extract key terms from the query for broader search
        query_terms = user_query.lower().split()
        keywords = [user_query.lower()] + query_terms + ["class", "classification", "device", "medical"]
        seen_ids = set()
        hybrid_pdf_results = []
        
        # 1. First try exact embedding match
        pdf_results = ctx.deps.supabase.rpc(
            'match_pdf_documents',
            {'query_embedding': query_embedding, 'match_threshold': 0.0, 'match_count': 10}
        ).execute()
        
        # Add embedding results first
        for doc in (pdf_results.data or []):
            doc_id = doc.get('id')
            if doc_id not in seen_ids:
                hybrid_pdf_results.append(doc)
                seen_ids.add(doc_id)
        
        # 2. If no results, try broader keyword search
        if not hybrid_pdf_results:
            for kw in keywords:
                # Search in content
                keyword_result_content = ctx.deps.supabase.from_('pdf_documents') \
                    .select('id, title, content, associated_url, metadata, ecr_metadata') \
                    .ilike('content', f'%{kw}%') \
                    .limit(10) \
                    .execute()
                
                # Search in title
                keyword_result_title = ctx.deps.supabase.from_('pdf_documents') \
                    .select('id, title, content, associated_url, metadata, ecr_metadata') \
                    .ilike('title', f'%{kw}%') \
                    .limit(10) \
                    .execute()
                
                # Add unique results
                for doc in (keyword_result_content.data or []):
                    doc_id = doc.get('id')
                    if doc_id not in seen_ids:
                        hybrid_pdf_results.append(doc)
                        seen_ids.add(doc_id)
                
                for doc in (keyword_result_title.data or []):
                    doc_id = doc.get('id')
                    if doc_id not in seen_ids:
                        hybrid_pdf_results.append(doc)
                        seen_ids.add(doc_id)
        
        # 3. Search in web pages
        web_results = ctx.deps.supabase.rpc(
            'match_site_pages',
            {'query_embedding': query_embedding, 'match_count': 10, 'filter': {}}
        ).execute()
        
        # Combine and format results
        results = []
        
        # Process PDF results
        for doc in hybrid_pdf_results:
            results.append({
                'type': 'pdf',
                'title': doc.get('title', ''),
                'content': doc.get('content', ''),
                'url': doc.get('associated_url', ''),
                'ecr_metadata': doc.get('ecr_metadata', {}),
                'similarity': doc.get('similarity', None)
            })
        
        # Process web results
        for doc in web_results.data:
            results.append({
                'type': 'web',
                'title': doc['title'],
                'content': doc['content'],
                'url': doc['url'],
                'similarity': doc['similarity']
            })
        
        # Sort by similarity if available
        results.sort(key=lambda x: x.get('similarity', 0) or 0, reverse=True)
        
        # Always return at least the top result if available
        if results:
            return json.dumps(results, indent=2)
        return ""
        
    except Exception as e:
        print(f"Error retrieving content: {e}")
        return ""

if __name__ == "__main__":
    from pyairtable import Table
    AIRTABLE_TOKEN = "patVw4ArosMIMAuuv.d8ca25e8659973be14c7aea8ae73ed3ecd936436a6d87a03028ddc589e07f54c"
    BASE_ID = "appzeMbm9zS6M0AM9"
    from pyairtable import Api
    api = Api(AIRTABLE_TOKEN)
    base = api.base(BASE_ID)
    print("Airtable tables in base:")
    for table in base.tables():
        print(f"Table: {table.name}")
        tbl = Table(AIRTABLE_TOKEN, BASE_ID, table.name)
        records = tbl.all()
        print(f"  {len(records)} records:")
        for rec in records:
            print(rec)