-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on site_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_site_pages_metadata on site_pages using gin (metadata);

-- Create a function to search for documentation chunks
create function match_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Everything above will work for any PostgreSQL database. The below commands are for Supabase security

-- Enable RLS on the table
alter table site_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);

-- Create table for PDF documents
create table pdf_documents (
    id bigserial primary key,
    title varchar not null,
    file_path varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    ecr_metadata jsonb not null default '{}'::jsonb,  -- Added ECR-specific metadata
    embedding vector(1536),
    associated_url varchar not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate PDFs for the same URL
    unique(associated_url, title)
);

-- Create an index for better vector similarity search performance
create index on pdf_documents using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_pdf_documents_metadata on pdf_documents using gin (metadata);
create index idx_pdf_documents_ecr_metadata on pdf_documents using gin (ecr_metadata);  -- Added index for ECR metadata

-- Create a function to search for PDF documents
create function match_pdf_documents (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  ecr_filter jsonb DEFAULT '{}'::jsonb  -- Added ECR filter parameter
) returns table (
  id bigint,
  title varchar,
  content text,
  metadata jsonb,
  ecr_metadata jsonb,  -- Added ECR metadata to return
  associated_url varchar,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    title,
    content,
    metadata,
    ecr_metadata,  -- Added ECR metadata to select
    associated_url,
    1 - (pdf_documents.embedding <=> query_embedding) as similarity
  from pdf_documents
  where metadata @> filter
    and ecr_metadata @> ecr_filter  -- Added ECR filter condition
  order by pdf_documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the table
alter table pdf_documents enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on pdf_documents
  for select
  to public
  using (true);