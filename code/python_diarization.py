from pyannote.audio import Pipeline
import whisper
import torch
import assemblyai as aai
import json
from llamaapi import LlamaAPI

# Initialize the llamaapi with your api_token
llama = LlamaAPI("LA-2f3b783951f64f4d8e688af2a465cb36f8f9a8049ea04b5e8907d66709f5e706")

# Define your API request
# Make your request and handle the response
# Setup assemblyAI api key
aai.settings.api_key = "a83396a1fe984f0da9e7a4f4ea528fe9"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Whisper ASR model
asr_model = whisper.load_model("base")

# Function to get GPT's interpretation of speaker names from the transcript
def interpret_speakers_with_gpt(transcript):
    prompt = f"""
    The following is a transcript of a conversation. Some speakers are labeled with generic names (e.g., Speaker 1, Speaker 2).
    Based on the dialogue, infer and provide more specific or suitable names for the speakers if possible. 
    If the names cannot be inferred, keep the default label.

    Transcript:
    {transcript}

    Please rewrite the transcript with the inferred speaker names or the default labels if a name cannot be inferred.
    """
    
    
    try:
        
        api_request_json = {
        "messages": [
            {"role": "user", "content": "What is the weather like in Boston?"},
        ],
        "stream": False,
        }
        response = llama.run(api_request_json)
        interpreted_transcript = response.json().get("content", "")
        return interpreted_transcript

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None  # Return None or handle the error as needed


# Function to perform diarization, transcription, and speaker labeling
def diarize_and_label_speakers(audio_file_path):
    try:
        # Step 1: Run speaker diarization
        transcriber = aai.Transcriber()
       
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript_aai = transcriber.transcribe(audio_file_path, config=config)

        transcript = ""

        for utterance in transcript_aai.utterances:
            transcript += f"Speaker {utterance.speaker}: {utterance.text}\n"
        print(transcript)
        # Step 5: Use GPT to interpret speaker names from the raw transcript
        interpreted_transcript = interpret_speakers_with_gpt(transcript)

        # Step 6: Return the final interpreted transcript
        return interpreted_transcript

    except Exception as e:
        print(f"Error occurred: {e}")
        return None  # Return None or handle the error as needed


# Path to the audio file
audio_file_path = "../data/Data.wav"

# Step 7: Run the diarization and labeling process
final_transcript = diarize_and_label_speakers(
    '../data/Data.wav'
)

# Output the final transcript
print(final_transcript)
