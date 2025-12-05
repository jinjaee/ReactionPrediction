from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import this
from pydantic import BaseModel
from src.reaction_engine import ReactionEngine
import uvicorn
import traceback

app = FastAPI(title="EOH Reaction Engine")

# --- CRITICAL: ALLOW FRONTEND CONNECTION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL websites (e.g. localhost:5500, localhost:8080)
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS (Fixes 405 error)
    allow_headers=["*"],
)

engine = ReactionEngine()

class ReactionRequest(BaseModel):
    element_a: str
    element_b: str

@app.post("/predict_reaction")
async def predict(request: ReactionRequest):
    try:
        products, _ = engine.get_reaction_products(request.element_a, request.element_b)
        return {
            "reactants": [request.element_a, request.element_b],
            "stable_products": products,
            "status": "success"
        }
    except Exception as e:
        print("!!! SERVER CRASHED !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)