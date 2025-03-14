python3 -m venv .venv
source .venv/bin/activate

which python
.venv/bin/python

====

[//]: # (DOES NOT WORK!!)

[//]: # (fastapi dev main.py)

pip install fastapi uvicorn
uvicorn main:app --reload 