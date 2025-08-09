from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Talent Match Engine", version="0.1.0")

# Enable CORS for frontend integration later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AI Talent Match Engine is running!", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-talent-matcher"}