import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.image_classifier import get_classifier

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

router = APIRouter()


@router.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Read one byte past the limit so oversized uploads are detected without loading them fully
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB")

    try:
        classifier = get_classifier()
    except FileNotFoundError:
        logger.exception("CNN weights file is missing")
        raise HTTPException(status_code=503, detail="Classifier model is not available")

    try:
        result = await run_in_threadpool(classifier.predict, image_bytes)
    except ValueError as error:  # only "not a readable image" is reported as a client error
        raise HTTPException(status_code=400, detail=str(error))

    return {"filename": file.filename, **result}
