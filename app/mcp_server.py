from mcp.server.fastmcp import FastMCP
from litellm import completion
from .database import db_manager
from .config import settings
from dotenv import load_dotenv

load_dotenv(override=True)


mcp = FastMCP("PostgreSQL-LLM-Server")


@mcp.resource("/context/{query}")
async def get_context(query: str) -> str:
    """Fetch relevant context from clinical trials and publications"""

    results = await db_manager.get_context_data(query)
    
    # Format results into readable text
    context_parts = []
    for result in results:
        if result['source_type'] == 'clinical_trial':
            context_parts.append(
                f"Clinical Trial: {result['nct_id']}\n"
                f"Title: {result['brief_title']}\n"
                f"Phase: {result['phase']}\n"
                f"Status: {result['overall_status']}\n"
                f"Description: {result['description']}\n"
            )
        else:
            context_parts.append(
                f"Publication: {result['doi']}\n"
                f"Title: {result['title']}\n"
                f"Journal: {result['journal_name']}\n"
                f"Abstract: {result['abstract']}\n"
            )
    
    return "\n\n".join(context_parts)


@mcp.tool("/query_llm")
async def query_llm(query: str, context: str = None) -> str:
    """Query LLM with context"""
    if context is None:
        # System prompt for general queries
        system_prompt = """You are a helpful assistant for a medical research portal called MedQuery AI. 
        For general queries and greetings:
        - Respond in a friendly, professional, and empathetic manner
        - Introduce yourself as MedQuery AI's assistant when appropriate
        - Explain that you can help search through clinical trials and scientific publications
        - Highlight your capabilities in medical research assistance
        
        Your key capabilities include:
        - Searching and analyzing clinical trials data (including trial phases, eligibility, and status)
        - Finding relevant medical publications and research papers
        - Providing context-aware responses with citations to sources
        - Understanding complex medical queries and research questions
        
        Remember to:
        - Be concise but informative
        - Encourage users to ask specific medical research questions
        - Maintain a professional tone while being approachable
        - Clarify that you're focused on medical research and clinical trials information"""
    else:
        # System prompt for research queries
        system_prompt = """You are a helpful assistant that answers questions based on provided context from clinical trials and publications. 
        Follow these guidelines:
        1. Use the provided context to ensure accurate, relevant information
        2. Always cite your sources using:
           - Clinical Trial ID (NCT number) when referencing trial information
           - DOI when referencing publications
        3. If the context doesn't contain sufficient information, clearly state this
        4. Structure your responses clearly with:
           - Direct answers to the query
           - Supporting evidence from the context
           - Relevant citations
        5. For medical terms, provide brief explanations when appropriate
        6. Maintain scientific accuracy while being understandable
        
        Remember to:
        - Stay within the scope of the provided context
        - Be clear about any limitations in the available information
        - Highlight key findings or relevant details
        - Maintain professional medical communication standards"""
    
    try:
        # Format the user prompt based on context availability
        if context:
            context_prompt = "Available Context:\n\n"
            context_items = [item for item in context.split("\n\n") if item.strip()]
            
            for item in context_items:
                if "Clinical Trial" in item:
                    # Extract and format clinical trial information
                    lines = item.split("\n")
                    trial_id = next((line.split(": ")[1] for line in lines if "Clinical Trial:" in line), "")
                    title = next((line.split(": ")[1] for line in lines if "Title:" in line), "")
                    description = next((line.split(": ")[1] for line in lines if "Description:" in line), "")
                    
                    context_prompt += f"CLINICAL TRIAL (ID: {trial_id})\n"
                    context_prompt += f"Title: {title}\n"
                    context_prompt += f"Description: {description}\n\n"
                
                elif "Publication" in item:
                    # Extract and format publication information
                    lines = item.split("\n")
                    doi = next((line.split(": ")[1] for line in lines if "Publication:" in line), "")
                    title = next((line.split(": ")[1] for line in lines if "Title:" in line), "")
                    abstract = next((line.split(": ")[1] for line in lines if "Abstract:" in line), "")
                    
                    context_prompt += f"PUBLICATION (DOI: {doi})\n"
                    context_prompt += f"Title: {title}\n"
                    context_prompt += f"Abstract: {abstract}\n\n"
            
            user_prompt = f"Based on the above context, please answer this question: {query}"
            final_prompt = context_prompt + "\n" + user_prompt
        else:
            final_prompt = system_prompt + query
        
        # Get response from LLM
        response = completion(
            model=settings.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        response_text = response.choices[0].message.content
        
        # Save the conversation to database
        await db_manager.save_conversation(
            query=query,
            response=response_text,
            model=settings.DEFAULT_MODEL,
            context_items=context.split("\n\n") if context else []
        )
        
        return response_text
        
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        print(error_message)  # Log the error
        return "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."



@mcp.prompt()
def create_prompt(query: str) -> str:
    """Create a prompt with context"""
    return f"Please answer this query using the provided context: {query}"


# if __name__ == "__main__":
#     mcp.run()