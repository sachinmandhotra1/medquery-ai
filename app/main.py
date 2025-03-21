from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .mcp_server import mcp, query_llm
from .database import db_manager
from pathlib import Path
from dotenv import load_dotenv
import uvicorn


load_dotenv(override=True)

app = FastAPI(title="LLM Context API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Set up templates and static files
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount MCP server
app.mount("/mcp", mcp, "MCP Server")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "active_page": "home"})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "active_page": "chat"})


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    # Fetch conversation history from database
    conversations = await db_manager.get_conversation_history()
    return templates.TemplateResponse(
        "history.html", 
        {"request": request, "active_page": "history", "conversations": conversations}
    )


@app.post("/query")
async def process_query(query: str = Form(...)):
    try:
        target_table = await db_manager.determine_search_table(query)
        
        if target_table == 'general':
            response = await query_llm(query, None)
            return {
                "response": response,
                "context": []
            }
        
        context_data = await db_manager.get_context_data(query)
        
        # Improved context formatting
        context_text = "Here is the relevant information:\n\n"
        for item in context_data:
            if item['source_type'] == 'clinical_trial':
                context_text += f"CLINICAL TRIAL (ID: {item['nct_id']})\n"
                context_text += f"Title: {item['brief_title']}\n"
                context_text += f"Phase: {item['phase']}\n"
                context_text += f"Status: {item['overall_status']}\n"
                context_text += f"Description: {item['description']}\n"
                if item.get('eligibility_criteria'):
                    context_text += f"Key Eligibility: {item['eligibility_criteria'][:500]}...\n"
                context_text += "\n"
            else:
                context_text += f"PUBLICATION (DOI: {item['doi']})\n"
                context_text += f"Title: {item['title']}\n"
                context_text += f"Journal: {item['journal_name']}\n"
                context_text += f"Abstract: {item['abstract']}\n\n"
        
        # Only proceed if we have meaningful context
        if len(context_data) > 0:
            response = await query_llm(
                f"Based on the following information, please provide a detailed and accurate response to this question: {query}\n\n{context_text}. NOTE: Do not include any introductory message to your response when generating summary on the provided context."
            )
        else:
            response = "I apologize, but I couldn't find enough relevant information to provide a detailed answer to your question. Could you please rephrase or provide more specific details?"
        
        return {
            "response": response,
            "context": context_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    try:
        # Test database connection
        await db_manager.get_context_data("test")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
