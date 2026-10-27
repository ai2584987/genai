from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.digit_generator import get_digit_generator
from app.embedding_model import EmbeddingModel
from app.image_classifier import get_classifier

app = FastAPI(title="GenAI API")

# ---------- Assignment 1: bigram text generation ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)

# ---------- Assignment 1: spaCy word embeddings ----------
embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(10, ge=1, le=200)


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    return {"generated_text": bigram_model.generate_text(request.start_word, request.length)}


@app.post("/embedding")
def calculate_embedding(request: EmbeddingRequest):
    result = embedding_model.get_embedding(request.word)
    if not result["has_vector"]:
        raise HTTPException(status_code=404, detail=f"No vector found for '{request.word}'")
    return result


# ---------- Assignment 2: CIFAR-10 CNN classifier ----------
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB")
    try:
        classifier = get_classifier()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Classifier weights are missing")
    try:
        result = await run_in_threadpool(classifier.predict, image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"filename": file.filename, **result}


# ---------- Assignment 3: GAN digit generator (MNIST) ----------
@app.get(
    "/generate_digits",
    responses={200: {"content": {"image/png": {}}, "description": "Grid of generated digits"}},
    response_class=Response,
)
async def generate_digits(
    num_samples: int = Query(16, ge=1, le=64, description="How many digits to generate"),
    seed: int | None = Query(None, description="Optional seed for repeatable output"),
):
    try:
        generator = get_digit_generator()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="GAN weights are missing")
    png = await run_in_threadpool(generator.generate_png, num_samples, seed)
    return Response(content=png, media_type="image/png")
