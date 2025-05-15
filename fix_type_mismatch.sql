-- Drop the existing function to avoid conflicts
DROP FUNCTION IF EXISTS match_pdf_documents(vector(1536), float, int, jsonb);

-- Update the match_pdf_documents function to use consistent types
CREATE OR REPLACE FUNCTION match_pdf_documents(
    query_embedding vector(1536),
    match_threshold float,
    match_count int,
    ecr_filter jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id bigint,
    title varchar,
    content text,
    metadata jsonb,
    ecr_metadata jsonb,
    associated_url varchar,
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
        pdf_documents.associated_url,
        1 - (pdf_documents.embedding <=> query_embedding) as similarity
    FROM pdf_documents
    WHERE 1 - (pdf_documents.embedding <=> query_embedding) > match_threshold
    AND (ecr_filter = '{}'::jsonb OR pdf_documents.ecr_metadata @> ecr_filter)
    ORDER BY pdf_documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$; 