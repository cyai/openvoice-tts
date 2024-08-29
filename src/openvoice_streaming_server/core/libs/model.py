import re
import torch

from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice.mel_processing import spectrogram_torch
from MeloTTS.melo.api import TTS
from MeloTTS.melo import utils
import librosa
from tqdm import tqdm


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
        audio = torch.tensor(audio_data).float()

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
        try:
            async for audio_chunk in self.generate_audio_chunks(
                audio_data=audio_data,
                src_se=src_se,
                tgt_se=tgt_se,
                output_path=output_path,
            ):
                yield audio_chunk.tobytes()
        except Exception as e:
            print("Error in cloned tts_stream", e)


class StreamingMeloSpeakerTTS(TTS):
    language_marks = {
        "english": "EN",
        "chinese": "ZH",
    }

    async def generate_audio_chunks(
        self,
        text,
        speaker_id,
        language="English",
        output_path=None,
        sdp_ratio=0.2,
        noise_scale=0.6,
        noise_scale_w=0.8,
        speed=1.0,
        pbar=None,
        format=None,
        position=None,
        quiet=False,
    ):
        mark = self.language_marks.get(language.lower(), None)
        assert mark is not None, f"language {language} is not supported"

        texts = self.split_sentences_into_pieces(text, mark)

        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ["EN", "ZH_MIX_EN"]:
                t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
                t, language, self.hps, device, self.symbol_to_id
            )
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = (
                    self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1.0 / speed,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                #
            # audio_list.append(audio)
            yield audio

    async def tts_stream(
        self,
        text,
        speaker_id,
        output_path=None,
        sdp_ratio=0.2,
        noise_scale=0.6,
        noise_scale_w=0.8,
        speed=1.0,
        pbar=None,
        format=None,
        position=None,
        quiet=False,
    ):
        try:
            async for audio_chunk in self.generate_audio_chunks(
                text=text,
                speaker_id=speaker_id,
                output_path=output_path,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                speed=speed,
                pbar=pbar,
                format=format,
                position=position,
                quiet=quiet,
            ):
                yield audio_chunk.tobytes()
        except Exception as e:
            print("Error in Melo tts_stream", e)
