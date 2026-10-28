# Deploy the controlled demo on Render

This guide describes a single-instance Docker deployment. It has not been executed against your Render account. API keys stay in Render's private server environment, not in GitHub, HTML, URLs or the video. A separate shared demo password gates paid API requests; it is not a Nebius or Tavily key.

## Prepare locally

1. Run `python -m unittest discover -s tests -v` and test the sample.
2. Commit the current changes and push them to GitHub.
3. Generate a demo password privately on your computer:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep the output private. Do not paste it into source code, a public description, this repository or a URL.

## Create the service

1. Sign in to Render and choose New, then Web Service.
2. Connect your GitHub account and select `mahejain/ml_investigator`, branch `main`.
3. Use the repository root and Docker runtime. The existing Dockerfile supplies the build and start instructions.
4. Choose a region and a compute plan after reviewing its price. Free is suitable for an initial test but sleeps when inactive and loses its local SQLite request counters on restart, sleep or redeploy. Paid services can attach a persistent disk; confirm current pricing before choosing paid resources.
5. Add the following variables using Render's Environment settings. Secret values must be entered in the dashboard, not committed.

| Variable | Value |
| --- | --- |
| `NEBIUS_API_KEY` | Your actual key, entered privately |
| `TAVILY_API_KEY` | Your actual key, entered privately |
| `DEMO_ACCESS_TOKEN` | The new random demo password |
| `HOST` | `0.0.0.0` |
| `PORT` | `10000` |
| `DAILY_REQUEST_LIMIT` | `50` initially; monitor use and adjust for judging |
| `DAILY_IP_LIMIT` | `50` initially; a proxy may group visitors under one IP |
| `MAX_CONCURRENT_INVESTIGATIONS` | `2` |

6. Set the health check path to `/health` if available.
7. Deploy and inspect the build and runtime logs. A successful health check only confirms the server is running.

## Persistent counters

For a paid single-instance service with a persistent disk, mount it at `/var/data` and set `RATE_LIMIT_DB=/var/data/request_limits.sqlite3`. Do not use this path on a service without that mount. Keep one service instance unless you replace SQLite with a correctly shared datastore.

The free filesystem is ephemeral. On a free initial demo, do not rely on the daily counter as a durable spending limit. The shared password helps prevent casual unauthenticated requests from reaching the APIs, but someone who knows it can consume credits up to the limits. Monitor provider usage; use provider-side budget controls where available. This guide does not guarantee costs or judging-period uptime.

## Verify before recording or submitting

1. Open the HTTPS URL in an incognito window.
2. Load the sample and attempt an investigation with no password. It should reject the request.
3. Enter the separate demo password under Demo access and run the sample. Check the result, numeric warnings and citation checks.
4. Verify successful runtime requests in your Nebius and Tavily usage views. Never show keys, password fields or provider account details in a screenshot or video.
5. Test again after a redeploy. Confirm environment settings remain and, if configured, the disk-backed counter persists.
6. Test how the selected plan behaves after inactivity, and ensure access, credentials and credits remain available for the required judging period.
7. Give judges the demo URL and the separate demo password using the intended testing-instructions field. Check whether that field is private before assuming the password is judge-only. If it is public, the password does not restrict access to judges.

Use one YouTube video demonstrating this same app. The hosted app URL is a separate submission item, not a request for another video. Keep the project described as a controlled prototype, not a production service.

## Provider references

- https://render.com/docs/web-services
- https://render.com/docs/configure-environment-variables
- https://render.com/docs/free
- https://render.com/docs/disks
