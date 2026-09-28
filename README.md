# Interview Review Prototype

A local Flask prototype that extracts facial-action and head-pose signals from selected video frames and presents an experimental review brief. It does not transcribe speech, evaluate interview answers, or assess job skills.

## Important limitations

The video signal and mapped OCEAN-style cues are experimental and are not validated measures of personality, confidence, performance, or job suitability. Do not use them to rank, screen out, or make hiring decisions. Use a consistent role-related interview rubric and human review.

Candidate recordings and generated frames, scores, and reports are local data. They are excluded from this repository by `.gitignore`.

## Requirements

- Windows
- Python 3.11 or newer
- OpenFace 2.2.0 for Windows x64, installed separately

## Setup

Create and install the Python environment from PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `OPENFACE_PATH` to the installed `FeatureExtraction.exe` for the current PowerShell session. For example:

```powershell
$env:OPENFACE_PATH = 'C:\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe'
```

Start the app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5001/>. Enter candidate details and video consent before uploading a recording or selecting the webcam.

The app uses OpenFace's `model/main_clnf_general.txt` landmark model. Keep the OpenFace executable and its model files together in the OpenFace installation directory; OpenFace is not included in this repository.

## Feedback

The recruiter brief and candidate practice suggestions are generated locally from the available experimental video signals. Suggested interview practice is general guidance; it is not an evaluation of a candidate's spoken answers. No local language-model service is required.