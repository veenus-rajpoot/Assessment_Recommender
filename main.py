import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from agent import handle_chat
from retrieval import Catalog
from schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-agent")

app = FastAPI(title="SHL Assessment Recommender")

# Loaded once at process startup, not per-request: catalog is small (377
# items) and TF-IDF fitting takes well under a second, but there's no reason
# to pay that cost inside the 30s per-call budget on every /chat call.
catalog = Catalog()


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return handle_chat(request.messages, catalog)
    except Exception:
        # The endpoint must never 500 into a schema-breaking response — an
        # uncaught exception here would fail the hard-eval schema-compliance
        # check outright. Degrade to a valid, in-scope reply instead.
        logger.exception("unhandled error in /chat")
        return JSONResponse(
            status_code=200,
            content=ChatResponse(
                reply="Sorry, I hit an issue processing that — could you rephrase your request?",
                recommendations=[],
                end_of_conversation=False,
            ).model_dump(),
        )
