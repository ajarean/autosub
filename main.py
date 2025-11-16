import argparse
import os
import subprocess
import sys
import shutil
import datetime as dt
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import sherpa_onnx


ENCODER = "./models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/encoder-epoch-99-avg-1.int8.onnx"
DECODER = "./models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/decoder-epoch-99-avg-1.int8.onnx"
JOINER = "./models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/joiner-epoch-99-avg-1.int8.onnx" 
TOKENS =  "./models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt"
SILERO_VAD = "./models/silero_vad.onnx"

def get_args():
    parser = argparse.ArgumentParser(
        description="Automatically generates a subtitle file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="higher = less sensitive vad (finds more silence, lower = more)",
    )
    
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=0.25,
        help="min silence for vad splitting",
    )

    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=0.5, 
        help="min speech duration.",
    )
    
    parser.add_argument(
        "--max-speech-duration",
        type=float,
        default=10.0,
        help="max speech duration",
    )
    
    parser.add_argument(
        "--force-max-duration",
        type=float,
        default=6.0,
        help="is a segment is longer than this, force it to split",
    )
    
    parser.add_argument(
        "--segment-padding",
        type=float,
        default=0.0,
        help="adds padding before and after segments",
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="input video path",
    )

    return parser.parse_args()


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03d}"


@dataclass
class Segment:
    start: float
    duration: float
    text: str = ""
    samples: np.ndarray = None

    @property
    def end(self):
        return self.start + self.duration

    def __str__(self):
        start_ts = format_timestamp(self.start)
        end_ts = format_timestamp(self.end)
        return f"{start_ts} --> {end_ts}\n{self.text}"


def split_long_segment(segment: Segment, max_duration: float, sample_rate: int) -> list:
    """
    if a segment exceeds max duration, then this is where it gets split
    """
    if segment.duration <= max_duration or segment.samples is None:
        return [segment]
    
    chunks = []
    samples_per_chunk = int(max_duration * sample_rate)
    total_samples = len(segment.samples)
    
    for i in range(0, total_samples, samples_per_chunk):
        chunk_samples = segment.samples[i:i + samples_per_chunk]
        chunk_duration = len(chunk_samples) / sample_rate
        
        chunks.append(Segment(
            start=segment.start + (i / sample_rate),
            duration=chunk_duration,
            samples=chunk_samples,
            text="" 
        ))
    
    return chunks


def create_recognizer(args: argparse.Namespace) -> sherpa_onnx.OfflineRecognizer:
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=ENCODER,
        decoder=DECODER,
        joiner=JOINER,
        tokens=TOKENS,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
    )

    return recognizer


def main():
    args = get_args()
    recognizer = create_recognizer(args)

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = SILERO_VAD
    vad_config.silero_vad.threshold = args.vad_threshold
    vad_config.silero_vad.min_silence_duration = args.min_silence_duration
    vad_config.silero_vad.min_speech_duration = args.min_speech_duration
    vad_config.silero_vad.max_speech_duration = args.max_speech_duration
    vad_config.sample_rate = args.sample_rate

    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=300)

    print(f"Opening {args.input_file} with FFmpeg...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        args.input_file,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(args.sample_rate),
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    srt_filename = args.output_srt or args.input_file + ".srt"
    segment_list = []
    segment_index = 1
    
    frames_per_read = args.sample_rate * 10
    
    start_time = dt.datetime.now()
    print("Starting transcription stream...")
    num_processed_samples = 0
    is_eof = False
    
    all_audio_samples = []

    while not is_eof:
        data = process.stdout.read(frames_per_read * 2)
        if not data:
            is_eof = True
            vad.flush()
        else:
            samples = np.frombuffer(data, dtype=np.int16)
            samples = samples.astype(np.float32) / 32768.0
            num_processed_samples += len(samples)
            all_audio_samples.append(samples)
            vad.accept_waveform(samples)

        while not vad.empty():
            vad_segment = vad.front
            padding_samples = int(args.segment_padding * args.sample_rate)
            start_sample = max(0, vad_segment.start - padding_samples)
            end_sample = vad_segment.start + len(vad_segment.samples) + padding_samples
            
            if all_audio_samples:
                full_audio = np.concatenate(all_audio_samples)
                padded_samples = full_audio[start_sample:min(end_sample, len(full_audio))]
            else:
                padded_samples = vad_segment.samples
                
            segment = Segment(
                start=start_sample / args.sample_rate,
                duration=len(padded_samples) / args.sample_rate,
                samples=padded_samples.copy()
            )
            
            print(f"VAD detected segment: {segment.start:.2f}s - {segment.end:.2f}s (duration: {segment.duration:.2f}s)")
            
            sub_segments = split_long_segment(segment, args.force_max_duration, args.sample_rate)
            
            if len(sub_segments) > 1:
                print(f"  -> Split into {len(sub_segments)} sub-segments")
            
            for sub_seg in sub_segments:
                stream = recognizer.create_stream()
                stream.accept_waveform(args.sample_rate, sub_seg.samples)
                recognizer.decode_stream(stream)
                
                sub_seg.text = stream.result.text.strip()
                if sub_seg.text:
                    segment_list.append(sub_seg)
                    print(f"Segment {segment_index}: {sub_seg}")
                    segment_index += 1
            
            vad.pop()

    end_time = dt.datetime.now()

    print(f"\nwriting {len(segment_list)} segments to {srt_filename}...")
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for i, seg in enumerate(segment_list):
            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{seg}\n\n")

    print(f"\nsubtitles saved at: {os.path.abspath(srt_filename)}")
    print(f"\nenjoy!")


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found!!!!!! go install it!!!! (not my problem)")
        sys.exit(-1)
    
    main()