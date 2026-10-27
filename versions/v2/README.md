# ML Experiment Investigator

Explore why a classification run behaved unexpectedly. Upload a small ZIP containing Python, configuration, and run logs. The app extracts evidence with line numbers, calls an NVIDIA Nemotron model via **Nebius Token Factory**, checks whether cited lines exist, and retrieves scikit-learn resources via **Tavily**. Citation existence is not proof the claim is supported. The app never executes uploaded code.

## Get started (Windows)

Install Python 3.10 or newer. In PowerShell, open this project folder, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:NEBIUS_API_KEY = "your-nebius-key"
$env:TAVILY_API_KEY = "your-tavily-key"
python app.py
```

If PowerShell blocks activation, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and then `.\.venv\Scripts\python.exe app.py` instead. Visit http://127.0.0.1:8000 and upload `sample_project.zip`. Do not paste keys in chat or commit them to GitHub.

macOS/Linux: use `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`, then `export NEBIUS_API_KEY="..."` and `export TAVILY_API_KEY="..."` before `python app.py`.

## Real sample experiment

The ZIP includes measured results from the included `sample_project/train.py`. It generates synthetic data, trains an MLP on a fixed split, and repeats with a StandardScaler pipeline. Only scaling changes; this is one experiment and does not prove general model superiority. To reproduce it locally, run `python sample_project/train.py` and then recreate `sample_project.zip` from the three files in `sample_project/`. The run record includes seeds, versions, model parameters, metrics, convergence warnings, and a SHA-256 hash of the script. The hash identifies the recorded code; it does not prove the record is genuine. Results may vary by library version or platform.

## LangSmith and local metadata

Optional: set `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` before startup. This traces the model call through LangSmith and may send the prompt (including uploaded source) and response to LangSmith. Use approved sample data only. No live LangSmith integration was verified without credentials. Set `LANGSMITH_PROJECT` to choose a project.

Set `SAVE_RUN_METADATA=true` to record timestamps and counts of files, metrics and citations in `investigations.sqlite3`. This records metadata, not full configurations, results or experiment lineage; it is not a full experiment tracking system. It is off by default.

## Tests and Docker

Run `python -m unittest discover -s tests -v`. GitHub Actions runs these tests and the sample experiment after pushes and pull requests.

To build locally: `docker build -t ml-investigator .`. To run: `docker run --rm -p 127.0.0.1:8000:8000 -e NEBIUS_API_KEY -e TAVILY_API_KEY ml-investigator` after setting those variables in your host environment. Set additional tracing variables explicitly if desired.

## Scope, privacy and deployment

Default model ID is `nvidia/nemotron-3-super-120b-a12b`; override it with `NEBIUS_MODEL` if your account's catalog uses a different ID. Model receives uploaded text; Tavily receives a predefined documentation query selected from detected code, not an autonomous agent search. Optional LangSmith tracing can receive the full prompt and response. Do not upload secrets or sensitive data.

The application is built for trusted, local use and has no authentication, rate limiting, user isolation, or production hardening. The included Docker configuration does not by itself make it safe for public hosting. A deployment and a free judge-accessible link are still needed for the hackathon. Do not expose this server to the public before adding access and abuse controls. No uploaded code is run. ZIPs are never extracted. Evidence checks are limited; model output can be incorrect and should be reviewed.

See `EVALUATION.md` for the human-reviewed test-set plan and possible future Toloka use. Toloka/Tendem are not integrated. Tracing, live API calls for this version, Docker, GitHub Actions and hosting have not been validated in this environment without credentials or access to those services.
