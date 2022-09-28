
import sounddevice as sd
import whisper
from scipy.io import wavfile
import argostranslate.package
import argostranslate.translate
import translate as translate
import asyncio

async def main():
    #there are no english models for large
    model = "medium"
    verbose = True
    audio_model = whisper.load_model(model)    

    RATE = 44100
    CHANNELS = 2
    DEVICE_INDEX = 2
    SECONDS = 10
    AUDIO_FILE = 5
    TO_CODE = 'en'

    #print(sd.query_devices())
    sd.default.device[0] = DEVICE_INDEX
    index = 0
    isPackage = True
    while True:
        #try:
        print('listening...')
        audioPath = f"audio/audio{index}.wav"
        recording = sd.rec(frames=RATE * SECONDS, samplerate = RATE, blocking=True, channels=CHANNELS)
        sd.wait()
        wavfile.write(audioPath, rate=RATE, data=recording)

        print('transcribing...')
        result = audio_model.transcribe(audioPath)
        language: str= result.get('language')
        
        # Translate
        if language != TO_CODE:
            translatedText = await translate.translate(result.get('text'), language, TO_CODE)
        else:
            translatedText = result.get('text')
        print(translatedText)

        index += 1
        if index > AUDIO_FILE:
            index = 0

if __name__ == '__main__':    
    asyncio.get_event_loop().run_until_complete(main())