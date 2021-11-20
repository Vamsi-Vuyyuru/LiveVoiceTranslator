from __future__ import division

import os
import re
import sys
import streamlit as st
#from PIL import Image
import time

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'spinalcode2021-5b069f553152.json'
from google.cloud import speech

import pyaudio
from six.moves import queue
from playsound import playsound
import streamlit as st




import tornado.iostream
import tornado.web
import tornado.gen
import tornado.websocket
import asyncio

class SocketHandler(tornado.websocket.WebSocketHandler):
    waiters = set()
    def initialize(self):
        self.client_name = "newly_connected"

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self):
        print('connection opened')
        # SocketHandler.waiters.add(self)

    def on_close(self):
        print("CLOSED!", self.client_name)
        try:
            SocketHandler.waiters.remove(self)
        except KeyError:
            print('tried removing new client')

    def check_origin(self, origin):
        # Override the origin check if needed
        return True

    @classmethod
    async def send_updates(cls, message):
        if len(cls.waiters) < 2:
            while True:
                chat = {}
                # Prevent RuntimeError: Set changed size during iteration
                waiters_copy = cls.waiters.copy()
                for waiter in waiters_copy:
                    try:
                        await waiter.write_message(chat)
                    except tornado.websocket.WebSocketClosedError:
                        pass
                    except tornado.iostream.StreamClosedError:
                        pass
                    except Exception as e:
                        print('Exception e:', waiter.client_name)
                        pass

                # sleep a bit
                await asyncio.sleep(0.02)
        else:
            print('broadcast loop already running')

    async def on_message(self, message):
        print("RECEIVED :", message)
        self.client_name = message
        await self.first_serve_cache_on_connnect()
        SocketHandler.waiters.add(self)
        await SocketHandler.send_updates(message)

    async def first_serve_cache_on_connnect(self):
        print('serving cache on connect')
        temp_calc_results = self.namespace.results
        try:
            await self.write_message(temp_calc_results)
        except Exception as e:
            pass


###### Input microphone capturing with buffer for streaming ######
# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, #Changed to 2 by me
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=0,
            output_device_index=1,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses,target,mod):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    count = 0
    c1, c2 = st.columns((1, 1))
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)

            text_before_translation = transcript + overwrite_chars

            with c1:
                st.text("\n")
                st.text("-----------------------------------------")
                st.write("Source transcript (punctuations ignored)")
                st.markdown(f'<p style="background-color:LightGray;color:black;font-size:18px;">{text_before_translation}</p>',
                            unsafe_allow_html=True)
                st.text("\n")
                st.text("\n")
                st.text("\n")

            text_after_translation = translate_text(text=text_before_translation,target=target,c1=c1,c2=c2) #Translation
            synthesize_text(text=text_after_translation,target=target, counter=count,c1=c1,c2=c2,modd=mod)  # Calling Text to Speech Function


            count+=1


            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit|stop)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0

        if count == 5843:
            count=0



###### Text To Speech ######
def synthesize_text(text,target, counter,c1,c2,modd):

    """Synthesizes speech from the input string of text."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code=target, #"en-IN",#"en-US",#"te-IN",
        #name="en-US-Standard-C", #"Telugu (India)",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3

    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )


    Current_Dir = os.getcwd()   # Storing the Home(current) directory


    os.chdir('Audio_Output/')  # Changing directory to Audio-Output


    # The response's audio_content is binary.
    with open("{}.mp3".format(counter), "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file ' + "{}.mp3".format(counter))

    if modd=='Discrete':
        audio_file = open("{}.mp3".format(counter),"rb")
        audio_bytes = audio_file.read()
        with c2:
            st.audio(audio_bytes, format='audio/ogg')

    elif modd=='Live':
        playsound("{}.mp3".format(counter))  ### For playing the audio files



    os.chdir(Current_Dir)      # Back to Home Directory


###### Translation Function with detection ######
# Imports the Google Cloud Translation library
from google.cloud import translate


# Initialize Translation client
def translate_text(c1,c2,project_id="lvtproject-331210",text="No Voice Input", target="en-US"):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/lvtproject-331210/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            #"source_language_code": "en-US",
            "target_language_code": target,
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print("Translated text: {}".format(translation.translated_text))

        with c2:
            st.text("______________________________")
            st.write('The translated transcript is...')


            st.markdown(f'<p style="background-color:LightGray;color:green;font-size:18px;">{translation.translated_text}</p>',
                        unsafe_allow_html=True)
            #st.markdown(translation.translated_text)

    return response.translations[0].translated_text






###### Speech to Text ######
def main():



    # img = Image.open('/Users/syamsi/Downloads/icon.jpg')
    # st.image(img,use_column_width= True)

    ##### This code is for labelling the language selection options #####
    SrCHOICES = {'te-IN': "Telugu",
               'ta-IN': "Tamil",
               'en-US': "English(US)",
               'en-IN':"English(IN)",
               'hi-IN':"Hindi",
               'fr-FR':"French(FR)"
               }

    def format_func(option):
        return SrCHOICES[option]

    TrCHOICES = {'en-IN':"English(IN)",
                 'en-US': "English(US)",
                 'hi-IN': "Hindi",
                 'fr-FR': "French(FR)",
                 'ta-IN': "Tamil",
                 'te-IN': "Telugu"}

    def format_func(option):
        return TrCHOICES[option]

    ##### This code is for aligning the language selection options #####
    c1, c2 = st.columns((1, 1))

    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.

    with c1:
        source_language_code = st.selectbox('Select your source language ',  options=list(SrCHOICES.keys()), format_func=format_func) #"te-IN"  #"en-US"  # a BCP-47 language tag - Select the source language
    with c2:
        target_language_code = st.selectbox('Select your target language ',  options=list(TrCHOICES.keys()), format_func=format_func) #"en-IN"  # Select the target language










    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        enable_automatic_punctuation=True,
        language_code=source_language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    #import pyaudio
    #pya = pyaudio.PyAudio()
    #st.write(pya.get_default_input_device_info())
    #st.write(pya.get_default_output_device_info())



    cl1,cl2 = st.columns((1,1))

    with cl1:
        liv = st.radio("Choose Live or Discrete", ('Discrete', 'Live'))

    with cl2:
        strm = st.select_slider("Streaming:", ["Stop","Start"])





    while strm=='Start':#st.button('Start Streaming'):   # Only when button is pressed, the stream will start
        st.write("Live Streaming has started, please start talking... ")
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)


            # Now, put the transcription responses to use.
            listen_print_loop(responses,target=target_language_code, mod=liv)




if __name__ == "__main__":

    st.markdown("<h1 style='text-align: center; color: red;'>Syamsi Team</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: red;'>presents</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Live Voice Translator</h1>", unsafe_allow_html=True)

    main()
