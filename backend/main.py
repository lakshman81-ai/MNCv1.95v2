from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import tempfile
from typing import Optional

from transcription import transcribe_audio_pipeline, transcribe_audio

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: mock mode via env variable
USE_MOCK = bool(int(os.getenv("MNC_USE_MOCK", "0")))


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    stereo_mode: bool = Form(False),
    start_offset: float = Form(0.0),
    max_duration: Optional[float] = Form(None),
):
    """
    Endpoint to handle audio file upload and return MusicXML.

    - stereo_mode: whether to keep stereo processing (for now, mid-channel).
    - start_offset: segment start (seconds) – supports your 10s-segment idea.
    - max_duration: maximum duration (seconds) to process from the offset.
    """
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename or "upload")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            # Run full pipeline
            result = transcribe_audio_pipeline(
                tmp_path,
                stereo_mode=stereo_mode,
                use_mock=USE_MOCK,
                start_offset=start_offset,
                max_duration=max_duration,
            )

            # For now, keep API body as MusicXML for compatibility
            xml_bytes = result.musicxml.encode("utf-8")
            return Response(content=xml_bytes, media_type="application/xml")

            # If later you want timeline + notes, you can:
            # return {
            #     "musicxml": result.musicxml,
            #     "analysis": result.analysis_data.to_dict(),
            # }

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "mock_mode": USE_MOCK}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
