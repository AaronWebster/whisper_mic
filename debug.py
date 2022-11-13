import os
import re
import tempfile
import wave
import inflect

import pyaudio
import rhasspysilence
import whisper
from rhasspysilence.utils import trim_silence


def PrintTranscription(transcription):
    print(transcription)

    # if transcription.get('no_speech_prob') == None:
    #     return

    # if transcription['no_speech_prob'] > 0.75:
    #     return

    command = transcription['text']
    if not command:
        return

    command = command.lower()

    print('0 -> \'{}\''.format(command))
    p = inflect.engine()
    for i in range(20):
        command = command.replace(p.number_to_words(i), str(i))
    command = command.replace('oh', '0')

    print('1 -> \'{}\''.format(command))
    command = command.strip()
    print('2 -> \'{}\''.format(command))
    command = command.strip('.')
    print('3 -> \'{}\''.format(command))
    command = re.sub(r'[^0-9.]', '', command)
    print('4 -> \'{}\''.format(command))

    if '.' not in command:
        return

    if command == len(command) * command[0]:
        return
    print('5 -> \'{}\''.format(command))
    # print(command)


p = pyaudio.PyAudio()
audio_model = whisper.load_model('base.en', device='cuda')

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                frames_per_buffer=960,
                input=True)

recorder = rhasspysilence.WebRtcVadRecorder(
    vad_mode=1,
    skip_seconds=0.0,
    min_seconds=0.75,
    speech_seconds=0.3,
    silence_seconds=0.75,
    before_seconds=0.75,
    silence_method=rhasspysilence.SilenceMethod.VAD_ONLY,
)

recorder.start()


audio_tmpdir = tempfile.mkdtemp()
audio_tmpfile = os.path.join(audio_tmpdir, 'audio.wav')

print('ready')
while True:
    chunk = stream.read(960)
    if not chunk:
        break

    result = recorder.process_chunk(chunk)

    if result:
        audio_bytes = recorder.stop()

        with wave.open(audio_tmpfile, 'wb') as w:
            w.setframerate(recorder.sample_rate)
            w.setsampwidth(2)
            w.setnchannels(1)
            w.writeframes(audio_bytes)

        transcription = audio_model.transcribe(
            audio_tmpfile, language='english')

        PrintTranscription(transcription)

        recorder.start()
