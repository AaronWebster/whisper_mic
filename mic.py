#!/usr/bin/env python3

import speech_recognition as sr
import whisper
from scipy.io import wavfile
import asyncio

async def main():
    model = 'tiny'
    audio_model = whisper.load_model(model)

    RATE = 44100
    DEVICE_INDEX = 9

    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8

    with sr.Microphone(sample_rate=RATE, device_index=DEVICE_INDEX) as source:
        while True:
            audio = r.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            audio_clip.export(save_path, format="wav")

            result = audio_model.transcribe(save_path,language='english')
            predicted_text = result["text"]
            print(predicted_text)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())


# Make a Dict Lookup table thing with  https://pages.uoregon.edu/jgarman/Weebles/drillzzz.txt
# trailing zero's = no / ZERO POINT in the begining yes - why? just cause I guess -
#DrillLookup={
#     "0.12" : "Number 31",
#     "0.1221" : "3.1 mm",
#     "0.125" : "1/8 in",
#     "0.126" : "3.2 mm",
#     "0.1285" : "Number 30",
#     "0.1299" : "3.3 mm",
#     "0.1339" : "3.4 mm",
#     "0.136" : "Number 29",
#     "0.1378" : "3.5 mm",
#     "0.1405" : "Number 28",
#     "0.1406" : "9/64 in",
#     "0.1417" : "3.6 mm",
#     "0.144" : "Number 27",
#     "0.1457" : "3.7 mm",
#     "0.147" : "Number 26",
#     "0.1495" : "Number 25",
#     "0.1496" : "3.8 mm",
#     "0.152" : "Number 24",
#     "0.1535" : "3.9 mm",
#     "0.154" : "Number 23",
#     "0.1563" : "5/32 in",
#     "0.157" : "Number 22",
#     "0.1575" : "4 mm",
#     "0.159" : "Number 21",
#     "0.161" : "Number 20",
#     "0.1614" : "4.1 mm",
#     "0.1654" : "4.2 mm",
#     "0.166" : "Number 19",
#     "0.1693" : "4.3 mm",
#     "0.1695" : "Number 18",
#     "0.1719" : "11/64 in",
#     "0.173" : "Number 17",
#     "0.1732" : "4.4 mm",
#     "0.177" : "Number 16",
#     "0.1772" : "4.5 mm",
#     "0.18" : "Number 15",
#     "0.1811" : "4.6 mm",
#     "0.182" : "Number 14",
#     "0.185" : "Number 13",
#     "0.185" : "4.7 mm",
#     "0.1875" : "3/16 in",
#     "0.189" : "4.8 mm",
#     "0.189" : "Number 12",
#     "0.191" : "Number 11",
#     "0.1929" : "4.9 mm",
#     "0.1935" : "Number 10",
#     "0.196" : "Number 9",
#     "0.1969" : "5 mm",
#     "0.199" : "Number 8",
#     "0.2008" : "5.1 mm",
#     "0.201" : "Number 7",
#}

