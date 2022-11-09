import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import tempfile
import os
import click


temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "temp.wav")


@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
@click.option("--language", default=None, help="Spoken language in whisper.torknizer.LANGUAGES", type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option('--fp16', default=False, help='Whether to perform inference in fp16; False by default', is_flag=True, type=bool)
def main(model, language, verbose, energy, pause,dynamic_energy, fp16):
    #there are no english models for large
    if model != "large" and language == 'en':
        model = model + ".en"
    audio_model = whisper.load_model(model)
    
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(device_index=1, sample_rate=16000) as source:
        print('Microphone:', source.list_microphone_names()[1])
        print("Say something!")
        while True:
            #get and save audio to wav file
            audio = r.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            audio_clip.export(save_path, format="wav")

            if language is not None:
                result = audio_model.transcribe(save_path, language=language, fp16=fp16)
            else:
                result = audio_model.transcribe(save_path, fp16=fp16)

            if not verbose:
                predicted_text = result["text"]
                print(">", predicted_text)
            else:
                print(result)
                
main()