import re
import torch

from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice.mel_processing import spectrogram_torch
import librosa


class StreamingBaseSpeakerTTS(BaseSpeakerTTS):
    language_marks = {
        "english": "EN",
        "chinese": "ZH",
    }

    async def generate_audio_chunks(self, text, speaker, language="English", speed=1.0):
        mark = self.language_marks.get(language.lower(), None)
        assert mark is not None, f"language {language} is not supported"

        texts = self.split_sentences_into_pieces(text, mark)

        for t in texts:
            t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            t = f"[{mark}]{t}[{mark}]"
            stn_tst = self.get_text(t, self.hps, False)
            device = self.device
            speaker_id = self.hps.speakers[speaker]
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                sid = torch.LongTensor([speaker_id]).to(device)
                audio = (
                    self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        sid=sid,
                        noise_scale=0.667,
                        noise_scale_w=0.6,
                        length_scale=1.0 / speed,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
                yield audio

    async def tts_stream(self, text, speaker, language="English", speed=1.0):
        async for audio_chunk in self.generate_audio_chunks(
            text, speaker, language, speed
        ):
            yield audio_chunk.tobytes()


class StreamingCloneSpeakerTTS(ToneColorConverter):
    language_marks = {
        "english": "EN",
        "chinese": "ZH",
    }

    async def generate_audio_chunks(
        self,
        audio_data,
        src_se,
        tgt_se,
        output_path=None,
        tau=0.3,
        message="default",
    ):
        hps = self.hps
        # load audio
        # audio, sample_rate = librosa.load(audio_data, sr=hps.data.sampling_rate)
        audio = torch.tensor(audio).float()

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = (
                self.model.voice_conversion(
                    spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
            yield audio

    async def tts_stream(self, audio_data, src_se, tgt_se, output_path=None):
        async for audio_chunk in self.generate_audio_chunks(
            audio_data=audio_data,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=output_path,
        ):
            yield audio_chunk.tobytes()
