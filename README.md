python3 -m venv .venv
source .venv/bin/activate

which python
.venv/bin/python

====

[//]: # (DOES NOT WORK!!)

[//]: # (fastapi dev main.py)

pip install fastapi uvicorn
uvicorn main:app --reload 


Deployment

heroku login

heroku create

git push heroku main

If 

# Add changes to git
git add requirements.txt

# Commit the changes
git commit -m "Update gradio version in requirements.txt"

# Push to Heroku
git push heroku main


heroku logs -n 200

https://dashboard.heroku.com/apps/quiet-fjord-79931/settings


Live audio here:
https://github.com/gaborvecsei/whisper-live-transcription/blob/main/standalone-poc/live-transcribe.py

https://github.com/ufal/whisper_streaming?tab=readme-ov-file

https://github.com/QuentinFuxa/whisper_streaming_web



$ sudo -i // then enter password

$ mkdir /Users/ekantikachaudhuri/.config/github-copilot.
$ touch /Users/ekantikachaudhuri/.config/github-copilot/hosts.json
$ touch /Users/ekantikachaudhuri/.config/github-copilot/app.json
$ sudo chmod -R 777 /Users/ekantikachaudhuri/.config/github-copilot