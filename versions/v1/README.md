# ML Experiment Investigator

An evidence-led prototype for investigating small scikit-learn classification projects. Upload a ZIP of code, configuration and run logs; the app highlights metric lines and asks an NVIDIA Nemotron model on Nebius Token Factory to propose evidence-linked hypotheses and a controlled test. Tavily optionally retrieves documentation. It does not run uploaded code or prove causality.

## Setup (Python 3.10+)

No packages are needed. Set your API keys in your terminal; never commit them or paste them into chat.

Windows PowerShell:

```powershell
$env:NEBIUS_API_KEY = "your-nebius-key"
$env:TAVILY_API_KEY = "your-tavily-key"
python app.py
```

macOS / Linux:

```bash
export NEBIUS_API_KEY="your-nebius-key"
export TAVILY_API_KEY="your-tavily-key"
python3 app.py
```

Visit http://127.0.0.1:8000. ZIP the contents of `sample_project` and upload the ZIP. Sample numbers are fictional, not results of executing a model. Optionally set `NEBIUS_MODEL` to a Nemotron model ID accessible in your account. Requests send selected uploaded text to Nebius; a general documentation query is sent to Tavily. Never upload secrets or sensitive data. The server is local only, with no authentication; do not expose it to the public internet.

## Limitations

Only small UTF-8 code, config, CSV and log files are read. ZIPs aren't extracted or executed. Uploaded instructions are untrusted and model output can be wrong; verify evidence manually. Search currently uses a fixed scikit-learn query; it is not agent-selected. The tool does not verify the uploaded files match the supplied metrics. It has not been tested with live API credentials. Hosting and a public demo remain to be set up separately.
