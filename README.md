# Autosub
Andy Jarean

## Overview
A tool used to automatically generate .srt subtitle files (viewable in video players such as VLC) for videos \
Intended for language learners \

## Usage
supply your own model and silero_vad.onnx file and put them in a /models/ folder \
the model I'm using is sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01\
also supply your own videos

call it like this and replace ./my_video.mp4 with your video\
adjust the params however you like (look in main.py for params)
```
python ./main.py `
--vad-threshold 0.3 `
--min-silence-duration 0.15 `
--min-speech-duration 0.1 `
--max-speech-duration 3.0 `
--segment-padding 0.2 `
--force-max-duration 6.0 `
"./my_video.mp4"
```

## Flaws/Plans
- Originally this was planned as a language learning tool, with Anki (flashcard) integration and a GUI but I didn't really have time for that
    - hence why I chose Japanese 
- the subtitles aren't perfect
    - some lines are cut out, it performs poorly when many people are speaking at once, etc
    - for the most part, some of these issues can be remedied by playing around with the parameters. honestly I haven't found the optimal settings yet but the above seems to work okay?