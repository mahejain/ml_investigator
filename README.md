# ML Experiment Investigator

ML Experiment Investigator helps explore why a machine-learning classification run behaved unexpectedly. Upload a ZIP containing a small project, and the app examines its code, configuration and logs to produce an evidence-based investigation.

The app uses an NVIDIA Nemotron model through **Nebius Token Factory** and retrieves relevant scikit-learn documentation with **Tavily Search and Extract**. It presents observations, possible explanations and suggested next steps, with citations to the uploaded files.

This is a local prototype for assisting investigation. Its answers can be wrong and should be reviewed by a human.

## What it does

- Reads supported files from an uploaded project ZIP.
- Detects available metrics and errors and provides file and line references.
- Retrieves relevant scikit-learn documentation using Tavily.
- Sends the evidence and retrieved references to a Nemotron model for analysis.
- Checks whether cited file lines exist and were included in the evidence sent to the model. This does **not** guarantee that a claim is supported.
- Asks for more evidence when it cannot detect a supported metric or error.
- Does **not** execute uploaded code or extract the ZIP to disk.

The investigation is a fixed retrieval-and-analysis workflow, not an autonomous tool-calling agent.

## Getting started

### Requirements

- Python 3.10 or newer
- A Nebius Token Factory API key
- A Tavily API key

### Windows

Open PowerShell in the project folder and run these commands **one at a time**:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install -r requirements.txt
```

Set your API keys for the current terminal session:

```powershell
$env:NEBIUS_API_KEY = "your-nebius-key"
```

```powershell
$env:TAVILY_API_KEY = "your-tavily-key"
```

Start the app:

```powershell
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and upload `sample_project.zip`.
Alternatively, choose **Load included sample**, then select **Investigate project**.

If PowerShell blocks virtual-environment activation, you can use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

After setting the API keys as shown above, start the app with:

```powershell
.\.venv\Scripts\python.exe app.py
```

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export NEBIUS_API_KEY="your-nebius-key"
export TAVILY_API_KEY="your-tavily-key"
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and upload `sample_project.zip`.

**Keep API keys private.** Do not put real keys in the code, upload them in a project ZIP, or commit them to GitHub.

## Try the sample experiment

The included `sample_project.zip` contains results from a small, reproducible classification experiment. The sample script generates synthetic data and trains an MLP classifier on a fixed split, then repeats the experiment using a `StandardScaler` pipeline.

This comparison illustrates one change in one experiment. It does not establish that scaling always improves a model.

To reproduce the sample locally:

```bash
python sample_project/train.py
```

Then recreate `sample_project.zip` from the three files inside `sample_project/` before uploading it to the app.

The sample run record includes seeds, library versions, model parameters, metrics, convergence warnings and a SHA-256 hash of the script. The hash identifies the recorded code; it does not independently prove that the record is genuine. Results may vary across library versions or platforms.

## Experiment tracking with MLflow

The project includes `track_mlflow.py` to import the existing sample experiment results into MLflow:

Install the optional MLflow dependency first with `python -m pip install mlflow`.

```bash
python track_mlflow.py
```

Start the local MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), select **ML investigator sample**, and compare the `baseline` and `scaled` runs.

This tracks the included sample results; it does not automatically track every investigation performed by the app. The local MLflow database and run files are excluded from Git.

To investigate a run from an MLflow tracking store, first install MLflow, find the run ID in MLflow, and export its metadata locally:

```bash
python export_mlflow_run.py YOUR_RUN_ID --tracking-uri sqlite:///mlflow.db --output mlflow_run_export.zip
```

Open and review the ZIP before uploading it: parameters can contain secrets or private data. The ZIP includes run parameters, latest metrics, status, and the MLflow Git commit tag if present. It does not download or execute artifacts, import code, or verify that a recorded commit produced the run. The investigator still requires a supported metric or error. Use `--tracking-uri` only with a tracking store you trust.

## Evaluation

Run the synthetic evaluation cases with your API keys set:

```bash
python evaluate_agent.py
```

The evaluation includes five cases, covering situations such as an error-only log, an F1 score, a generalization gap and missing evidence. Results and a review CSV are written to `evaluation_results/`.

These cases are **review aids, not ground truth**. A completed request or `status=ok` does not mean that the model's answer is correct. The missing-evidence guard can respond without calling the model, so that case should not be counted as a model-quality comparison.

To compare models, select a model available in your Nebius catalog using `NEBIUS_MODEL`, run the evaluation again, and compare answers on the same cases. Responses can vary between runs; meaningful assessment requires repeated runs and human review.

A Toloka human-review pilot has been explored but is **not integrated or reported here as a completed evaluation**. See `EVALUATION.md` for the evaluation plan.

## Tests

Run the local test suite:

```bash
python -m unittest discover -s tests -v
```

Twelve tests passed in a local run. The repository also includes a GitHub Actions workflow; check the **Actions** tab for the current status of the remote run.

## Optional: LangSmith tracing

To enable LangSmith tracing, set `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` before starting the app. You can also set `LANGSMITH_PROJECT` to choose a project.

Tracing may send the model prompt—including uploaded source text—and the response to LangSmith. Use only data you are permitted to share. Live LangSmith tracing has not been verified for this project.

## Optional: local investigation metadata

Set `SAVE_RUN_METADATA=true` before starting the app to save timestamps and counts of files, metrics and citations in `investigations.sqlite3`.

This is basic local metadata, not full experiment tracking: it does not record complete configurations, results or experiment lineage. It is off by default.

## Docker

Build the image:

```bash
docker build -t ml-investigator .
```

With `NEBIUS_API_KEY` and `TAVILY_API_KEY` set in your host environment, run:

```bash
docker run --rm -p 127.0.0.1:8000:8000 -e NEBIUS_API_KEY -e TAVILY_API_KEY ml-investigator
```

Docker operation has not yet been verified for this version. Binding to localhost keeps the example intended for local access; Docker alone does not make the application safe for public deployment.

## Privacy, limitations and deployment

- The model receives text from the uploaded project. Do not upload secrets or sensitive data.
- Tavily receives a documentation query selected from detected code, followed by an extraction request for a documentation URL. The uploaded code is not sent as the Tavily query.
- Only results from `scikit-learn.org` are accepted. If extraction fails, the app uses search snippets where available.
- Retrieved documentation is treated as untrusted reference material. It describes general library behavior and is not proof of what happened in a particular run.
- File and line checks confirm citation existence, not whether the cited text supports a claim.
- The model can invent details, make incorrect technical claims or suggest experiments that need further scrutiny. Review its output.
- The prototype includes basic daily request limits and a concurrency limit, but still lacks authentication, user isolation and other production protections. It is intended for trusted local use until deployment is reviewed.
- There is currently no verified public deployment or judge-accessible link. Add suitable access and abuse controls before exposing the server publicly.

The default model ID is `nvidia/nemotron-3-super-120b-a12b`. Set `NEBIUS_MODEL` to another model ID if needed for your account.

Request limits use a local SQLite database and UTC calendar days. Configure `DAILY_REQUEST_LIMIT` (default 30), `DAILY_IP_LIMIT` (default 5), `MAX_CONCURRENT_INVESTIGATIONS` (default 2), and optionally `RATE_LIMIT_DB`. These controls do not guarantee an API spending cap. Instances with separate databases do not share a limit, and a reverse proxy may cause all users to share one IP address. Public deployments require shared persistent storage, trusted proxy configuration, provider-side budget limits, and a separate security review.

Nebius Sandboxes, Tavily Map/Crawl/Research, Toloka and Tendem are not integrated. Optional LangSmith tracing covers the Nemotron call; Tavily steps do not have separate traces.

## Previous versions

The latest version is in the repository's main directory. Earlier snapshots are available in:

- [`versions/v1`](versions/v1)
- [`versions/v2`](versions/v2)
- [`versions/v3`](versions/v3)
