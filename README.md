# whisperx-api
Simple API for WhisperX, long-attention solution to the original model.

https://github.com/m-bain/whisperX

## Installation

	pip install -r requirements.txt

Make sure you have GPU acceleration configured and turned on.

## Run

    docker build --build-arg MODEL=large-v2 -t whisperx-api .
    docker run -p 8000:8000 -t whisperx-api

## Usage

/GET /status

Find out service is on and GPU is available

/POST /transcribe

:file: File to upload
