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