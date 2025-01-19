# Speaker Diarization and Labeling with Whisper, Pyannote, AssemblyAI, and GPT

This project demonstrates the process of speaker diarization, transcription, and speaker labeling using various tools and APIs, including Whisper ASR, AssemblyAI, and LlamaAPI for GPT integration.

## Prerequisites

Before running the code, ensure the following are installed and configured:

1. Python 3.7 or higher
2. Required libraries:
   - `pyannote.audio`
   - `whisper`
   - `torch`
   - `assemblyai`
   - `llamaapi`
3. API keys:
   - AssemblyAI API key
   - LlamaAPI token

## Setup

1. Clone the repository or copy the code to your local environment.
2. Install the required Python libraries:
   ```bash
   pip install pyannote.audio whisper torch assemblyai llamaapi
   ```
3. Replace placeholders for `aai.settings.api_key` and `LlamaAPI` token with your actual API keys.
4. Ensure the audio file `Data.wav` is placed in the `../data/` directory relative to the script.

## Code Description

### Key Components

1. **Initialization**
   - Load the Whisper ASR model.
   - Configure device setup (GPU or CPU).

2. **Speaker Diarization and Transcription**
   - Use AssemblyAI to transcribe the audio file and provide speaker labels.

3. **Speaker Name Interpretation with GPT**
   - Send the raw transcript to GPT (via LlamaAPI) for more specific or suitable speaker name inference.

4. **Output**
   - Return the interpreted transcript with improved speaker labels or default labels if inference is not possible.

### Code Structure

#### Import Libraries
```python
from pyannote.audio import Pipeline
import whisper
import torch
import assemblyai as aai
import json
from llamaapi import LlamaAPI
```

#### Initialize APIs
```python
llama = LlamaAPI("YOUR_LLAMA_API_KEY")
aai.settings.api_key = "YOUR_ASSEMBLYAI_API_KEY"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

asr_model = whisper.load_model("base")
```

#### Functions

1. **Interpret Speaker Names**
```python
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
        return None
```

2. **Diarization and Labeling**
```python
def diarize_and_label_speakers(audio_file_path):
    try:
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript_aai = transcriber.transcribe(audio_file_path, config=config)

        transcript = ""
        for utterance in transcript_aai.utterances:
            transcript += f"Speaker {utterance.speaker}: {utterance.text}\n"
        print(transcript)

        interpreted_transcript = interpret_speakers_with_gpt(transcript)
        return interpreted_transcript

    except Exception as e:
        print(f"Error occurred: {e}")
        return None
```

#### Main Execution
```python
audio_file_path = "../data/Data.wav"
final_transcript = diarize_and_label_speakers(audio_file_path)
print(final_transcript)
```

## Running the Code

1. Ensure all dependencies are installed and API keys are set.
2. Execute the script:
   ```bash
   python script_name.py
   ```
3. The output will display the final interpreted transcript with improved speaker labels.

## Notes

- Ensure the audio file is clear and of good quality for better transcription accuracy.
- LlamaAPI integration requires an active internet connection and valid API token.
- Handle any exceptions or API errors as needed to ensure robustness.

