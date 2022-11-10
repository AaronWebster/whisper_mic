"""Professor Drill Index
                                                                                                                                                                                                                              
Master of memory and machine.  Also known as CHAMFERMAN.
"""

import io
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime

import pyttsx3
import speech_recognition
import sounddevice
import whisper
import inflect
from absl import app, logging
from pydub import AudioSegment

MODEL = 'tiny'
ENERGY = 300
DYNAMIC_ENERGY = None
PAUSE_SEC = 0.8
DRILL_INDEX_FILENAME = 'data/drill_size_chart.txt'
EXPECTED_NUM_DRILL_INDEX_ENTRIES = 107


def CurrentTime():
    p = inflect.engine()
    now = datetime.now()
    return '{} {}'.format(p.number_to_words(now.hour), p.number_to_words(now.minute))


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    active = False
    logging.info('Available sound devices: \n%s', sounddevice.query_devices())

    logging.info('Reading drill size data from %s.', DRILL_INDEX_FILENAME)
    drill_sizes = {}
    drill_index = 1
    with open(DRILL_INDEX_FILENAME, 'r') as f:
        for line in f.readlines():
            drill_sizes[drill_index] = line.strip()
            drill_index = drill_index + 1

    if len(drill_sizes) != EXPECTED_NUM_DRILL_INDEX_ENTRIES:
        logging.fatal('Wrong number of entries in drill index file.  Got %d, should be %d', len(
            drill_sizes), EXPECTED_NUM_DRILL_INDEX_ENTRIES)

    audio_tmpdir = tempfile.mkdtemp()
    audio_tmpfile = os.path.join(audio_tmpdir, 'audio.wav')

    logging.info('Initializing whisper with model model \'%s\'.', MODEL)
    audio_model = whisper.load_model(MODEL)

    logging.info('Initializing speech recognition.')
    recognizer = speech_recognition.Recognizer()
    recognizer.energy_threshold = ENERGY
    recognizer.pause_threshold = PAUSE_SEC
    recognizer.dynamic_energy_threshold = DYNAMIC_ENERGY

    logging.info('Initializing voice engine.')
    voice_engine = pyttsx3.init()

    with speech_recognition.Microphone(sample_rate=16000, device_index=1) as source:
        logging.info('Listening.')
        while True:
            audio = recognizer.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            audio_clip.export(audio_tmpfile, format='wav')

            result = audio_model.to('gpu').transcribe(
                audio_tmpfile, language='english')
            command = result['text']
            logging.info(command)

            # Turn on/off taking action.
            if 'chamferman off' in command and active:
                logging.info('Switching to active=False')
                active = False

            if 'chamferman on' in command and not active:
                active = True
                logging.info('Switching to active=True')

            if not active:
                continue

            if 'what time is it' in command:
                current_time = CurrentTime()
                logging.info('Response: ', current_time)
                voice_engine.say(current_time)
                voice_engine.runAndWait()


if __name__ == '__main__':
    app.run(main)
