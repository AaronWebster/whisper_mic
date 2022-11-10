import io
from pydub import AudioSegment
import whisper
import tempfile
import os
import click
import pyttsx3

MODEL = 'tiny'
ENERGY = 300
DYNAMIC_ENERGY = None
PAUSE = 0.8

temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, 'temp.wav')

audio_model = whisper.load_model(model)

recognizer = speech_recognition.Recognizer()
recognizer.energy_threshold = energy
recognizer.pause_threshold = pause
recognizer.dynamic_energy_threshold = dynamic_energy

voice_engine = pyttsx3.init()

with speech_recognition.Microphone() as source:
  while True:
    audio = recognizer.listen(source)
    data = io.BytesIO(audio.get_wav_data())
    audio_clip = AudioSegment.from_file(data)
    audio_clip.export(save_path, format='wav')

    result = audio_model.transcribe(save_path, language='english')
    predicted_text = result['text']

    voice_engine.say(predicted_text)
    voice_engine.runAndWait()
