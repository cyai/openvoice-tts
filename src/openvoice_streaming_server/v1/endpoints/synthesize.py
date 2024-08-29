import logging
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from MeloTTS.melo.api import TTS

from openvoice_streaming_server.core.libs import (
    StreamingBaseSpeakerTTS,
    StreamingCloneSpeakerTTS,
    StreamingMeloSpeakerTTS,
)
from openvoice_streaming_server.core.schemas import SynthesisRequest, SynthesisResponse
from openvoice import se_extractor

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


async def send_audio_stream(websocket: WebSocket, audio_stream):
    try:
        async for audio_chunk in audio_stream:
            response = SynthesisResponse(audio_chunk=audio_chunk)
            await websocket.send_bytes(response.audio_chunk)
    except WebSocketDisconnect:
        pass


class WebSocketHandler:
    def __init__(self, tts_model, clone_model, melo_model, device):
        self.model = tts_model
        self.clone_model = clone_model
        self.connections = set()
        # self.source_se = torch.load(
        #     "../resources/checkpoints/base_speakers/EN/en_default_se.pth"
        # ).to(self.model.device)
        source_speaker = "../resources/Source.mp3"
        refrence_speaker = "../resources/Abdulla.mp3"
        self.source_se, audio_name = se_extractor.get_se(
            source_speaker, self.clone_model, vad=True
        )
        self.target_se, audio_name = se_extractor.get_se(
            refrence_speaker, self.clone_model, vad=True
        )
        self.melo_model = melo_model
        self.melo_source_se = torch.load(
            "../resources/checkpoints_v2/base_speakers/ses/en-india.pth",
            map_location=device,
        )
        # self.speaker_ids = self.model.hps.data.spk2id

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
            self.connections.remove(websocket)

    async def handle_websocket(self, websocket: WebSocket):
        await self.connect(websocket)
        try:
            # source_se = "checkpoints/base_speakers/EN/en_default_se.pth"
            while True:
                data = await websocket.receive_text()
                request = SynthesisRequest.parse_raw(data)
                if request.text == "":
                    await self.disconnect(websocket)
                    return
                text = request.text
                speaker = request.speaker
                language = request.language
                speed = request.speed
                logger.info(
                    f"Received text: {text}, speaker: {speaker}, language: {language}, speed: {speed}"
                )

                # audio_stream = self.model.tts_stream(text, speaker, language, speed)
                async for audio_chunk in self.model.generate_audio_chunks(
                    text, speaker, language, speed
                ):
                    cloned_audio_stream = self.clone_model.tts_stream(
                        audio_chunk, self.source_se, self.target_se
                    )
                    await send_audio_stream(websocket, cloned_audio_stream)

                # await send_audio_stream(websocket, audio_stream)
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error during text synthesis: {e}")
            await self.disconnect(websocket)

    async def melo_handle_websocket(self, websocket: WebSocket):
        await self.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                request = SynthesisRequest.parse_raw(data)
                if request.text == "":
                    await self.disconnect(websocket)
                    return
                text = request.text
                speaker = request.speaker
                language = request.language
                speed = request.speed
                logger.info(
                    f"Received text: {text}, speaker: {speaker}, language: {language}, speed: {speed}"
                )

                audio_stream = self.melo_model.tts_stream(
                    text=text,
                    speaker_id=2,  # "spk2id": { "EN-US": 0, "EN-BR": 1,"EN-INDIA": 2, "EN-AU": 4}
                )
                await send_audio_stream(websocket, audio_stream)

                # async for audio_chunk in self.melo_model.generate_audio_chunks(
                #     text=text,
                #     speaker_id=2,  # "spk2id": { "EN-US": 0, "EN-BR": 1,"EN-INDIA": 2, "EN-AU": 4}
                # ):
                #     cloned_audio_stream = self.clone_model.tts_stream(
                #         audio_chunk, self.melo_source_se, self.target_se
                #     )
                #     await send_audio_stream(websocket, cloned_audio_stream)

        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error during text synthesis: {e}")
            await self.disconnect(websocket)


en_checkpoint_base = "../resources/checkpoints/base_speakers/EN"
converter_checkpoint = "../resources/checkpoints/converter"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load models and resources
clone_model = StreamingCloneSpeakerTTS(
    f"{converter_checkpoint}/config.json", device=device
)
model = StreamingBaseSpeakerTTS(f"{en_checkpoint_base}/config.json", device=device)
melo_model = StreamingMeloSpeakerTTS(language="EN_V2", device=device)
model.load_ckpt(f"{en_checkpoint_base}/checkpoint.pth")
clone_model.load_ckpt(f"{converter_checkpoint}/checkpoint.pth")

handler = WebSocketHandler(model, clone_model, melo_model, device)


@router.websocket("/synthesize")
async def synthesize(websocket: WebSocket):
    await handler.handle_websocket(websocket)
