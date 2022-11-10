"""Professor Drill Index
                                                                                                                                                                                                                              
Master of memory and machine.
"""

import io
import os
import tempfile
from collections.abc import Sequence

import pyttsx3
import speech_recognition
import whisper
from absl import app, logging
from pydub import AudioSegment

MODEL = 'tiny'
ENERGY = 300
DYNAMIC_ENERGY = None
PAUSE_SEC = 0.8
DRILL_SIZES_FILENAME = 'drill_size_chart.txt'
EXPECTED_DRILL_SIZES_LENGTH = 107


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    drill_sizes = {}
    drill_index = 1
    with open(DRILL_SIZES_FILENAME, 'r') as f:
        for line in f.readlines():
            drill_sizes[drill_index] = line.strip()
            drill_index = drill_index + 1

    if len(drill_sizes) != EXPECTED_DRILL_SIZES_LENGTH:
        logging.fatal('Wrong number of drill sizes in chart.  Got %d, should be %d', len(
            drill_sizes), EXPECTED_DRILL_SIZES_LENGTH)

    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, 'temp.wav')

    logging.info('Initializing whisper with model model \'%s\'.', MODEL)
    audio_model = whisper.load_model(MODEL)

    logging.info('Initializing speech recognition.')
    recognizer = speech_recognition.Recognizer()
    recognizer.energy_threshold = ENERGY
    recognizer.pause_threshold = PAUSE_SEC
    recognizer.dynamic_energy_threshold = DYNAMIC_ENERGY

    logging.info('Initializing voice engine.')
    voice_engine = pyttsx3.init()

    with speech_recognition.Microphone() as source:
        logging.info('Listening.')
        while True:
            audio = recognizer.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            audio_clip.export(save_path, format='wav')

            result = audio_model.transcribe(save_path, language='english')
            predicted_text = result['text']

            voice_engine.say(predicted_text)
            voice_engine.runAndWait()


if __name__ == '__main__':
    app.run(main)
