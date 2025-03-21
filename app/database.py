from typing import Any, Dict, List, Tuple
from .utils.postgres import PostgresManager
from .config import settings
from openai import OpenAI
import asyncio


class DatabaseManager:
    def __init__(self):
        self.db = PostgresManager(
            min_connections=1,
            max_connections=10,
            retry_attempts=3,
            retry_delay=2
        )
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.schema = settings.DB_SCHEMA

        # Define which fields to embed for each table
        self.table_configs = {
            'clinical_trials_feed': {
                'id_column': 'id',
                'fields': [
                    'brief_title',
                    'description',
                    'eligibility_criteria',
                    'outcomes',
                    'search_text'
                ]
            },
            'publications_feed': {
                'id_column': 'publication_id',
                'fields': [
                    'title',
                    'abstract',
                    'journal_name'
                ]
            }
        }

    async def preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        # Convert to lowercase
        text = text.lower()
        # You can add more preprocessing steps here
        return text

    async def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI with preprocessing"""
        if not text or text.isspace():
            return None
        
        try:
            # Preprocess text before getting embedding
            processed_text = await self.preprocess_text(text)
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=processed_text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    async def create_embeddings_for_record(
        self, 
        table_name: str, 
        record_id: int, 
        fields: Dict[str, str]
    ) -> None:
        """Create embeddings for a single record's fields"""
        if record_id is None:
            print(f"Skipping record with None ID for {table_name}")
            return

        # Verify the record exists in the source table
        verify_query = f"""
            SELECT 1 FROM {self.schema}.{table_name}
            WHERE {self.table_configs[table_name]['id_column']} = %s
        """
        
        record_exists = self.db.execute_query(
            query=verify_query,
            params=(record_id,),
            fetch_all=True
        )
        
        if not record_exists:
            print(f"Record {record_id} not found in {table_name}, skipping")
            return

        for field_name, text in fields.items():
            if not text or text.isspace() or isinstance(text, (bytes, bytearray)):
                continue

            embedding = await self.get_text_embedding(text)
            if embedding:
                try:
                    upsert_query = f"""
                        INSERT INTO {self.schema}.text_embeddings 
                        (source_table, source_id, field_name, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source_table, source_id, field_name)
                        DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP
                    """
                    self.db.execute_query(
                        query=upsert_query,
                        params=(table_name, record_id, field_name, embedding),
                        fetch_all=False
                    )
                except Exception as e:
                    print(f"Error storing embedding for {table_name} record {record_id}, field {field_name}: {e}")

    async def determine_search_table(self, query: str) -> str:
        """Use LLM to determine which table to search based on query content"""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a classifier that determines whether a query is about clinical trials, scientific publications, or a general conversation. Respond with ONLY 'clinical_trials', 'publications', 'general', or 'both' if unclear between trials and publications."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0
            )
            decision = response.choices[0].message.content.strip().lower()
            
            if decision == 'clinical_trials':
                return 'clinical_trials_feed'
            elif decision == 'publications':
                return 'publications_feed'
            elif decision == 'general':
                return 'general'
            else:
                return 'both'
                
        except Exception as e:
            print(f"Error in table determination: {e}")
            return 'both'  # Default to searching both on error

    async def get_context_data(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced semantic search with improved field-specific matching"""
        try:
            processed_query = await self.preprocess_text(query)
            query_embedding = await self.get_text_embedding(processed_query)
            if not query_embedding:
                return []

            target_table = await self.determine_search_table(query)
            results = []

            # Clinical trials search
            if target_table in ['clinical_trials_feed', 'both']:
                clinical_trials_query = f"""
                    WITH semantic_results AS (
                        SELECT 
                            ct.id,
                            ct.nct_id,
                            ct.brief_title,
                            ct.description,
                            ct.phase,
                            ct.overall_status,
                            ct.sponsor_name,
                            ct.eligibility_criteria,
                            te_title.embedding as title_embedding,
                            te_desc.embedding as desc_embedding,
                            te_criteria.embedding as criteria_embedding,
                            GREATEST(
                                0.4 * (1 - (te_title.embedding::vector <-> %s::vector)),
                                0.3 * (1 - (te_desc.embedding::vector <-> %s::vector)),
                                0.3 * (1 - (te_criteria.embedding::vector <-> %s::vector))
                            ) as semantic_score,
                            ts_rank_cd(
                                setweight(to_tsvector('english', COALESCE(ct.brief_title, '')), 'A') ||
                                setweight(to_tsvector('english', COALESCE(ct.description, '')), 'B'),
                                plainto_tsquery('english', %s),
                                32
                            ) as text_score
                        FROM {self.schema}.clinical_trials_feed ct
                        LEFT JOIN {self.schema}.text_embeddings te_title ON 
                            te_title.source_table = 'clinical_trials_feed' 
                            AND te_title.source_id = ct.id
                            AND te_title.field_name = 'brief_title'
                        LEFT JOIN {self.schema}.text_embeddings te_desc ON 
                            te_desc.source_table = 'clinical_trials_feed' 
                            AND te_desc.source_id = ct.id
                            AND te_desc.field_name = 'description'
                        LEFT JOIN {self.schema}.text_embeddings te_criteria ON 
                            te_criteria.source_table = 'clinical_trials_feed' 
                            AND te_criteria.source_id = ct.id
                            AND te_criteria.field_name = 'eligibility_criteria'
                    )
                    SELECT 
                        'clinical_trial' as source_type,
                        nct_id,
                        brief_title,
                        description,
                        phase,
                        overall_status,
                        sponsor_name,
                        eligibility_criteria,
                        semantic_score,
                        (semantic_score * 0.7 + COALESCE(text_score, 0) * 0.3) as similarity
                    FROM semantic_results
                    WHERE semantic_score > 0.1 OR text_score > 0.1
                    ORDER BY similarity DESC
                    LIMIT 3;
                """
                clinical_trials = self.db.execute_query(
                    query=clinical_trials_query,
                    params=(query_embedding, query_embedding, query_embedding, processed_query),
                    fetch_all=True,
                    as_dict=True
                )
                results.extend(clinical_trials)

            # Publications search
            if target_table in ['publications_feed', 'both']:
                publications_query = f"""
                    WITH semantic_results AS (
                        SELECT 
                            p.publication_id,
                            p.title,
                            p.abstract,
                            p.journal_name,
                            p.doi,
                            p.publication_creation_date,
                            te_title.embedding as title_embedding,
                            te_abstract.embedding as abstract_embedding,
                            te_journal.embedding as journal_embedding,
                            GREATEST(
                                0.5 * (1 - (te_title.embedding::vector <-> %s::vector)),
                                0.3 * (1 - (te_abstract.embedding::vector <-> %s::vector)),
                                0.2 * (1 - (te_journal.embedding::vector <-> %s::vector))
                            ) as semantic_score,
                            ts_rank_cd(
                                setweight(to_tsvector('english', COALESCE(p.title, '')), 'A') ||
                                setweight(to_tsvector('english', COALESCE(p.abstract, '')), 'B') ||
                                setweight(to_tsvector('english', COALESCE(p.journal_name, '')), 'C'),
                                plainto_tsquery('english', %s),
                                32
                            ) as text_score
                        FROM {self.schema}.publications_feed p
                        LEFT JOIN {self.schema}.text_embeddings te_title ON 
                            te_title.source_table = 'publications_feed' 
                            AND te_title.source_id = p.publication_id
                            AND te_title.field_name = 'title'
                        LEFT JOIN {self.schema}.text_embeddings te_abstract ON 
                            te_abstract.source_table = 'publications_feed' 
                            AND te_abstract.source_id = p.publication_id
                            AND te_abstract.field_name = 'abstract'
                        LEFT JOIN {self.schema}.text_embeddings te_journal ON 
                            te_journal.source_table = 'publications_feed' 
                            AND te_journal.source_id = p.publication_id
                            AND te_journal.field_name = 'journal_name'
                        WHERE p.publication_creation_date IS NOT NULL
                    )
                    SELECT 
                        'publication' as source_type,
                        publication_id,
                        title,
                        abstract,
                        journal_name,
                        doi,
                        publication_creation_date,
                        semantic_score,
                        (semantic_score * 0.7 + COALESCE(text_score, 0) * 0.3) as similarity
                    FROM semantic_results
                    WHERE semantic_score > 0.1 OR text_score > 0.1
                    ORDER BY 
                        similarity DESC,
                        publication_creation_date DESC NULLS LAST
                    LIMIT 3;
                """
                publications = self.db.execute_query(
                    query=publications_query,
                    params=(query_embedding, query_embedding, query_embedding, processed_query),
                    fetch_all=True,
                    as_dict=True
                )
                results.extend(publications)

            # Enhanced filtering and sorting
            filtered_results = []
            for r in results:
                if r['similarity'] > 0.1:
                    if r['source_type'] == 'clinical_trial':
                        r['final_score'] = r['similarity'] * 1.1
                    else:
                        r['final_score'] = r['similarity']
                    filtered_results.append(r)

            sorted_results = sorted(filtered_results, key=lambda x: x['final_score'], reverse=True)
            return sorted_results[:5]

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    async def update_embeddings_for_table(self, table_name: str) -> None:
        """Update embeddings for all records in a table"""
        try:
            config = self.table_configs[table_name]
            id_column = config['id_column']
            fields = config['fields']
            fields_str = ', '.join(fields)
            
            # Modified query to only select records that exist in the source table
            select_query = f"""
                WITH valid_records AS (
                    SELECT {id_column}
                    FROM {self.schema}.{table_name}
                )
                SELECT t.{id_column}, {fields_str}
                FROM {self.schema}.{table_name} t
                INNER JOIN valid_records v ON t.{id_column} = v.{id_column}
                WHERE t.{id_column} NOT IN (
                    SELECT DISTINCT source_id 
                    FROM {self.schema}.text_embeddings 
                    WHERE source_table = %s
                )
                AND t.{id_column} IS NOT NULL
            """
            
            records = self.db.execute_query(
                query=select_query,
                params=(table_name,),
                fetch_all=True,
                as_dict=True
            )

            print(f"Processing {len(records)} records for {table_name}")
            
            for record in records:
                if record[id_column] is None:
                    continue
                    
                fields_dict = {
                    field: record[field] 
                    for field in fields 
                    if record.get(field) and not isinstance(record.get(field), (bytes, bytearray))
                }
                
                if fields_dict:  # Only process if we have valid fields
                    await self.create_embeddings_for_record(
                        table_name, 
                        record[id_column],
                        fields_dict
                    )

        except Exception as e:
            print(f"Error updating embeddings for {table_name}: {e}")

    async def save_conversation(
        self, 
        query: str, 
        response: str, 
        model: str,
        context_items: List[Dict[str, Any]]
    ) -> None:
        """Save conversation with context references"""
        try:
            sql_query = f"""
                INSERT INTO {self.schema}.conversation_history
                (query, response, model, created_at)
                VALUES (%s, %s, %s, DEFAULT);
            """
            
            self.db.execute_query(
                query=sql_query,
                params=(query, response, model),
                fetch_all=False
            )
        except Exception as e:
            print(f"Error saving conversation: {e}")

    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Fetch conversation history"""
        try:
            query = f"""
                SELECT query, response, model, created_at
                FROM {self.schema}.conversation_history
                ORDER BY created_at DESC
                LIMIT 10
            """
            return self.db.execute_query(query=query, fetch_all=True, as_dict=True)
        except Exception as e:
            print(f"Error fetching conversation history: {e}")
            return []


db_manager = DatabaseManager()
