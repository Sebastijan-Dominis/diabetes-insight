import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .routers import reports 

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=[os.getenv("API_URL")],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reports.router)

@app.get("/")
async def health_check():
    return {"Healthy": 200}