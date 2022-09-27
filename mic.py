import io
from pydub import AudioSegment
import speech_recognition as sr
import pyaudio
import wave
import whisper
import tempfile
import os
import click
from datetime import datetime
import json

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)

def main(model, english,verbose, energy, pause, dynamic_energy):
    #there are no english models for large
    model = "medium"
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)    
    
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = False

    RATE = 44100
    CHANNELS = 2
    DEVICE_INDEX = 5
    FORMAT = pyaudio.paInt16
    FRAMESBUFF = 1024
    SECONDS = 5
    AUDIO_FILE = 5

    daudio = pyaudio.PyAudio()
    
    '''
    deviceList = []
    for i in range(0, daudio.get_device_count()):
        deviceList.append(daudio.get_device_info_by_index(i))
    
    with open('device.json', 'w') as w:
        json.dump(deviceList, w , indent=4, separators=[',',':'])
        print(daudio.get_default_input_device_info().get('index'))
        print(daudio.get_default_output_device_info().get('index'))
    '''
    audiostream = daudio.open(output_device_index=DEVICE_INDEX,
                                rate=RATE, 
                                input=True, 
                                channels=CHANNELS,
                                format=FORMAT,
                                frames_per_buffer=FRAMESBUFF,
                                )
    index = 0
    while True:
        #try:
        print('listening...')
        frame = []
        audiostream.start_stream()
        for i in range(0, int(RATE / FRAMESBUFF *  SECONDS)):
            data = audiostream.read(FRAMESBUFF)
            frame.append(data)

        audiostream.stop_stream()
        print('saving audio file')
        audioPath = f"audio/audio{index}.wav"
        with wave.open(audioPath, 'wb') as w:
            w.setnchannels(2)
            w.setsampwidth(daudio.get_sample_size(FORMAT))
            w.setframerate(RATE)
            w.writeframes(b''.join(frame))
            w.close()

        print('transcribing...')
        result = audio_model.transcribe(audioPath)
        if not verbose:
            predicted_text = result["text"]
            print("You said: " + predicted_text)
        else:
            print(result)

        index += 1
        if index > AUDIO_FILE:
            index = 0
                
main()