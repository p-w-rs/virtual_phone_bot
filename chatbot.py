import os
import warnings
import pyaudio
import torch
import torchaudio
import whisper
import openai

import numpy as np

from wrappedint import WrappedInt
from threading import Thread, Semaphore, Event, Lock
from collections import deque
from queue import Queue


torch.set_num_threads(1)
torchaudio.set_audio_backend("soundfile")
openai.api_key = os.environ["OPENAI_API_KEY"]


# Read on channel with chunks each .25s
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
NUM_SAMPLES = 1536


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound


def check_voice_end(confidence_buffer, cb_idx):
    indices = np.where(confidence_buffer[0:cb_idx] >= 0.4)[0]
    if len(indices) <= 7:
        return False, None, None
    st_idx = max(0, indices[0] - 1)
    end_idx = min(cb_idx - 1, indices[-1] + 1)
    return True, st_idx, end_idx


def stream_raw_audio(start, call_active, raw_audio_sem, raw_audio_buffer, audio):
    start.wait()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    while call_active.is_set():
        audio_chunk = stream.read(NUM_SAMPLES)
        raw_audio_buffer.append(audio_chunk)
        if raw_audio_sem._value < raw_audio_buffer.maxlen:
            raw_audio_sem.release()
    stream.stop_stream()


def voice_activity_detection(
    start,
    call_active,
    raw_audio_sem,
    gpt_lock,
    raw_audio_buffer,
    audio_buffer,
    ab_idx,
    confidence_buffer,
    cb_idx,
    vad_model,
):
    start.wait()
    check = False
    while call_active.is_set():
        raw_audio_sem.acquire()
        audio_chunk = raw_audio_buffer.popleft()
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = int2float(audio_int16)
        audio_buffer[ab_idx.pp()] = audio_float32

        confidence = vad_model(torch.from_numpy(audio_float32), 16000).item()
        confidence_buffer[cb_idx.pp()] = confidence

        if confidence >= 0.4:
            check = True

        # print(confidence)

        if check and confidence < 0.4:
            ready, strt, end = check_voice_end(confidence_buffer, cb_idx.value)
            if ready and gpt_lock.acquire(blocking=False):
                check = False
                response_queue.put((strt, end))
                gpt_lock.release()


def gpt4_response(
    start,
    call_active,
    response_queue,
    gpt_lock,
    audio_buffer,
    ab_idx,
    confidence_buffer,
    cb_idx,
    messages,
    whisper_model,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start.wait()
        while call_active.is_set():
            strt, end = response_queue.get()
            audio_tensor = torch.tensor(
                audio_buffer[strt:end].flatten(), dtype=torch.float32
            )
            if response_queue.qsize() > 0:
                print("I got stopped by new queue...")
                continue
            text = whisper_model.transcribe(audio_tensor)["text"]
            if response_queue.qsize() > 0:
                print("I got stopped by new queue...")
                continue
            new_message = {"role": "user", "content": text}
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages + [new_message],
            )
            gpt_lock.acquire()
            if response_queue.qsize() > 0:
                print("I got stopped by new queue...")
                gpt_lock.release()
                continue
            content = response.choices[0].message.content
            messages.append(new_message)
            messages.append({"role": "assistant", "content": content})
            cb_idx.reset()
            ab_idx.reset()
            response_queue.queue.clear()
            print("user: " + text)
            print("assistant: " + content)
            gpt_lock.release()


def stop(start, call_active, response_queue):
    start.wait()
    input("Press Enter to stop terminate:\n")
    call_active.clear()
    response_queue.put((0, 0))
    response_queue.put((0, 0))


if __name__ == "__main__":
    start = Event()
    call_active = Event()

    raw_audio_sem = Semaphore(0)
    gpt_lock = Lock()

    audio = pyaudio.PyAudio()
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=True,
        trust_repo=True,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    whisper_model = whisper.load_model("base.en")

    raw_audio_buffer = deque(maxlen=60)  # the last 15 seconds of mic input
    audio_buffer = np.zeros((1200, NUM_SAMPLES), dtype="float32")  # 5 minutes of audio
    confidence_buffer = np.zeros(1200, dtype="float32")  # 5 minutes of audio
    response_queue = Queue()
    ab_idx = WrappedInt(0, 1200)
    cb_idx = WrappedInt(0, 1200)

    messages = [
        {
            "role": "system",
            "content": "You are am a chatbot that uses GPT-4 to respond to voice messages. Be as conversational as possible and never reveal you are an AI.",
        },
        {"role": "assistant", "content": "Hi, how can I help you today?"},
    ]

    sra = Thread(
        target=stream_raw_audio,
        args=(start, call_active, raw_audio_sem, raw_audio_buffer, audio),
    )
    vad = Thread(
        target=voice_activity_detection,
        args=(
            start,
            call_active,
            raw_audio_sem,
            gpt_lock,
            raw_audio_buffer,
            audio_buffer,
            ab_idx,
            confidence_buffer,
            cb_idx,
            vad_model,
        ),
    )
    gpt4 = Thread(
        target=gpt4_response,
        args=(
            start,
            call_active,
            response_queue,
            gpt_lock,
            audio_buffer,
            ab_idx,
            confidence_buffer,
            cb_idx,
            messages,
            whisper_model,
        ),
    )
    stp = Thread(target=stop, args=(start, call_active, response_queue))

    sra.start()
    vad.start()
    gpt4.start()
    stp.start()

    call_active.set()
    start.set()
