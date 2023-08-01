from argparse import ArgumentParser
from pathlib import Path

import torch

from data.datamodule import VoiceGenDataModule
from model.voicegen import VoiceGen
from utils.audio import (
    audio_to_codec,
    codec_to_audio,
    get_encodec_model,
    load_audio,
    save_audio,
)
from utils.text import encode_text


def main(
    text: str,
    target_text: str | None,
    target_audio_path: Path | None,
    checkpoint_path: str,
):
    model = VoiceGen.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda")

    cfg = model.cfg

    encodec_model = get_encodec_model(cfg.data.sample_rate, torch.device("cuda"))

    with torch.no_grad():
        encoded_text = torch.from_numpy(encode_text(text)).unsqueeze(0).to("cuda")
        text_len = torch.tensor([encoded_text.shape[1]]).to("cuda")

        if target_text is not None and target_audio_path is not None:
            encoded_target_text = (
                torch.from_numpy(encode_text(target_text)).unsqueeze(0).to("cuda")
            )
            target_audio = (
                torch.from_numpy(
                    load_audio(
                        target_audio_path,
                        cfg.data.sample_rate,
                        encodec_model.channels,
                    )
                )
                .unsqueeze(0)
                .to("cuda")
            )
            target_audio_codec = audio_to_codec(target_audio, encodec_model)

            target_text_len = torch.tensor([encoded_target_text.shape[1]]).to("cuda")
            target_audio_len = torch.tensor([target_audio_codec.shape[2]]).to("cuda")
        else:
            cfg.train.batch_size = 1
            datamodule = VoiceGenDataModule(cfg)
            datamodule.prepare_data()
            datamodule.setup()
            (
                encoded_target_text,
                target_text_len,
                target_audio_codec,
                target_audio_len,
            ) = next(iter(datamodule.test_dataloader()))
            target_text_len = target_text_len[:1]
            encoded_target_text = encoded_target_text[:1, : target_text_len[0]]
            target_audio_len = target_audio_len[:1]
            target_audio_codec = target_audio_codec[:1, :, : target_audio_len[0]]

        generated_audio = model.inference(
            encoded_text,
            encoded_target_text,
            target_audio_codec,
            text_len,
            target_text_len,
            target_audio_len,
        )

        raw_audio = codec_to_audio(generated_audio, encodec_model)
        raw_audio_numpy = raw_audio.squeeze(0).cpu().numpy()

        save_audio(Path("generated.wav"), raw_audio_numpy, 24000)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--target_text", type=str, required=False)
    parser.add_argument("--target_audio", type=Path, required=False)
    args = parser.parse_args()

    text: str = args.text
    checkpoint_path: str = args.checkpoint_path
    target_text: str | None = args.target_text
    target_audio: Path | None = args.target_audio

    main(text, target_text, target_audio, checkpoint_path)
