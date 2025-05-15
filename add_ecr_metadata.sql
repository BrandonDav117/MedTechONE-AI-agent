-- Add ecr_metadata column to pdf_documents table
ALTER TABLE pdf_documents 
ADD COLUMN IF NOT EXISTS ecr_metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

-- Create index for ecr_metadata
CREATE INDEX IF NOT EXISTS idx_pdf_documents_ecr_metadata 
ON pdf_documents USING gin (ecr_metadata);

-- Update the match_pdf_documents function to include ecr_metadata
CREATE OR REPLACE FUNCTION match_pdf_documents(
    query_embedding vector(1536),
    match_threshold float,
    match_count int,
    ecr_filter jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id bigint,
    title text,
    content text,
    metadata jsonb,
    ecr_metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        pdf_documents.id,
        pdf_documents.title,
        pdf_documents.content,
        pdf_documents.metadata,
        pdf_documents.ecr_metadata,
        1 - (pdf_documents.embedding <=> query_embedding) as similarity
    FROM pdf_documents
    WHERE 1 - (pdf_documents.embedding <=> query_embedding) > match_threshold
    AND (ecr_filter = '{}'::jsonb OR pdf_documents.ecr_metadata @> ecr_filter)
    ORDER BY pdf_documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$; 