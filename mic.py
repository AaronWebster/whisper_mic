#!/usr/bin/env python3

import sounddevice as sd
import whisper
from scipy.io import wavfile
import asyncio


async def main():
    model = 'medium'
    audio_model = whisper.load_model(model)

    RATE = 44100
    CHANNELS = 2
    DEVICE_INDEX = 2
    SECONDS = 5
    AUDIO_FILE = 5

    sd.default.device[0] = DEVICE_INDEX
    sd.default.dtype[0] = 'float32'
    index = 0

    while True:
        recording = sd.rec(frames=RATE * SECONDS, samplerate=RATE,
                           channels=CHANNELS, dtype='float32')
        if index == 0:
            indexPath = AUDIO_FILE
        else:
            indexPath = index - 1

        try:
            result = audio_model.transcribe(
                f'audio/audio{indexPath}.wav', no_speech_threshold=0.6, **{'task': 'translate'})
            print('\n')
            translatedText = result.get('text')
            print(translatedText)
        except:
            print('no audio track recorded yet')

        sd.wait()
        wavfile.write(f'audio/audio{index}.wav', rate=RATE, data=recording)

        index += 1
        if index > AUDIO_FILE:
            index = 0

if __name__ == '__main__':
    print(sd.query_devices())
    asyncio.get_event_loop().run_until_complete(main())
