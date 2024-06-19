import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import whisperx


class Transcriber:

    def __init__(
            self,
            model_name: str,
            device: str,
            batch_size: int,
            compute_type: str,
            language: Optional[str] = None
    ):
        self.device = device
        self.model = whisperx.load_model(model_name, device, compute_type=compute_type)
        self.batch_size = batch_size
        self.default_sr = 16_000
        self.default_language = language

    def transcribe_file(
            self,
            file_path: str,
            sr: Optional[int] = None,
    ) -> str:
        """
        Transcribe file on hard drive
        :param file_path: File path
        :param sr: Sample rate
        :return: Transcribed text
        """
        if sr is None:
            sr = self.default_sr

        audio = whisperx.load_audio(file_path, sr=sr)
        return self.transcribe_audio(audio)

    def transcribe_audio(
            self,
            audio: np.ndarray,
    ) -> Tuple[List[Dict[str, Union[str, float]]], str]:
        """
        Transcribe audio samples
        :param audio: Samples in numpy float32
        :return: Transcribed text
        """

        # convert to mono if stereo as required by whisper
        if len(audio.shape) == 2:
            # channels always first
            if audio.shape[1] < audio.shape[0]:
                audio = audio.T
            audio = np.mean(audio, axis=0)

        if len(audio.shape) > 2:
            raise ValueError("Too many dimensions")

        result = self.model.transcribe(audio, batch_size=self.batch_size)
        return result["segments"], result["language"]

    def segments(
            self,
            audio: np.ndarray,
            segments: List[Dict[str, Union[str, float]]],
            language: Optional[str] = None
    ) -> List[Dict[str, Union[str, float, dict]]]:
        if language is None:
            language = self.default_language

        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=self.device)
        result = whisperx.align(segments, model_a, metadata,
                                audio, self.device, return_char_alignments=False)

        return result["segments"]

    def speakers(
            self,
            audio: np.ndarray,
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None,
    ) -> List[Dict[str, Union[str, float, dict]]]:
        hf_token = os.getenv("HF_TOKEN")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=self.device)
        return diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
