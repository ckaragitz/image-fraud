from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
import logging

from utils.fraud_utils import analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Fraud Detection API")

class ImageRequest(BaseModel):
    source_type: Literal['base64'] = Field(
        ...,
        description="Type of image source"
    )
    source: str = Field(
        ...,
        description="Image data (base64)"
    )
    analysis_type: Literal['web_search', 'exif', 'classification'] = Field(
        ...,
        description="Type of analysis to perform"
    )

@app.post("/api/v1.1/analyze", response_class=JSONResponse)
async def analyze_fraud(request: ImageRequest):

    try:
        logger.info(f"Received request for analysis type: {request.analysis_type}")

        # Perform analysis
        response = analyzer.analyze(
            base64_content=request.source,
            analysis_type=request.analysis_type
        )

        return JSONResponse(content=response)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)