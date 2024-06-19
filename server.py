import os
from tempfile import TemporaryDirectory
from typing import Optional

import aiofiles
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from transcriber import Transcriber

BATCH_SIZE = 16

#model_name = os.getenv("MODEL", "large-v2")
model_name = os.getenv("MODEL", "base")
device_type = os.getenv("DEVICE_TYPE", "gpu")


class Audio(BaseModel):
    file: UploadFile
    start: Optional[float] = None
    end: Optional[float] = None


app = FastAPI()

device = "cpu"
compute_type = "int8"
if device_type == "gpu":
    device = "cuda"
    compute_type = "float16"

transcriber = Transcriber(
    model_name,
    device=device,
    batch_size=BATCH_SIZE,
    compute_type=compute_type,
    language="cs"
)


@app.get("/status/")
async def status():
    import torch

    return {
        "status": "ok",
        "gpu": torch.cuda.is_available()
    }


@app.post("/transcribe/")
async def transcribe(file: UploadFile):
    with TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)

        async with aiofiles.open(file_path, "wb") as fp:
            await fp.write(await file.read())

        print(f"Transcribing {file_path}")
        transcript = transcriber.transcribe_file(file_path)

    return {"transcript": transcript}


"""
@app.post("/segments/")
async def segments(file: UploadFile):
    with TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)

        async with aiofiles.open(file_path, "wb") as fp:
            await fp.write(await file.read())

        print(f"Transcribing {file_path}")
        transcript = transcriber.transcribe_file(file_path)

    return {"transcript": transcript}
"""
