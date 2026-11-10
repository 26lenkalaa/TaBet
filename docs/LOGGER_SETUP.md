# Setting up the external trigger for the forecast logger

The logger runs on GitHub Actions, so your laptop does not need to be on. GitHub's own schedule is
unreliable (in gameweek 10 it started runs only every 4–8 hours), so a free cron service starts the
workflow every 30 minutes instead. GitHub's schedule stays as a backup. Amendment A4 in
`docs/PROSPECTIVE_PROTOCOL.md` explains why.

You need to do this once, before **2026-11-20 04:30 UTC**, when the first gameweek-11 window opens.

## 1. Create a narrow GitHub token

The token is stored on a third-party site, so it should be able to do one thing only: start
workflows in this repo.

1. On GitHub, open **Settings → Developer settings → Personal access tokens → Fine-grained tokens**
   and click **Generate new token**.
2. **Token name:** `tabet-cron-trigger`.
3. **Expiration:** pick a date, for example 90 days, and put a reminder in your calendar a week
   before it so the logger does not silently stop.
4. **Resource owner:** `26lenkalaa`.
5. **Repository access:** **Only select repositories** → `TaBet-ML-Model-Edition`.
6. **Permissions → Repository permissions → Actions:** **Read and write**. Leave everything else at
   "No access" (Metadata becomes read-only automatically, which is required).
7. Click **Generate token** and copy it. GitHub shows it only once.

If the token ever leaks, delete it on the same page and make a new one.

## 2. Test the token from your terminal (dry run, logs nothing)

Replace `YOUR_TOKEN` and run:

```
curl -i -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/26lenkalaa/TaBet-ML-Model-Edition/actions/workflows/prospective.yml/dispatches \
  -d '{"ref":"main","inputs":{"mode":"dry-run"}}'
```

- **Expected:** `HTTP/2 204` with no body.
- **Then:** open the repo's **Actions** tab. Within about a minute a new "Prospective forecast log" run
  appears with the event `workflow_dispatch`. Its summary shows the heartbeat (next fixture and when
  its window opens), and the Run step prints the fixtures a real run would log.
- **401 or 403:** the token is wrong or lacks Actions: Read and write.
- **404:** the repository or workflow name is mistyped, or the token was not given access to this repo.

## 3. Create the cron job

1. Sign up at **cron-job.org** (free) and click **Create cronjob**.
2. **Title:** `TaBet logger`.
3. **URL:** `https://api.github.com/repos/26lenkalaa/TaBet-ML-Model-Edition/actions/workflows/prospective.yml/dispatches`
4. **Schedule:** every 30 minutes.
5. Open the **Advanced** tab:
   - **Request method:** `POST`
   - **Headers:**
     - `Authorization: Bearer YOUR_TOKEN`
     - `Accept: application/vnd.github+json`
     - `X-GitHub-Api-Version: 2022-11-28`
     - `Content-Type: application/json`
     - `User-Agent: tabet-cron-trigger`
   - **Request body:** `{"ref":"main","inputs":{"mode":"dry-run"}}` for the first test.
6. Turn on **notify me when the job fails**, so a broken token or a GitHub outage reaches your email.
7. Save, then use **Test run**. It should report status 204, and a dry-run run should appear in the
   Actions tab, as in step 2.
8. When the test works, change the request body to `{"ref":"main","inputs":{"mode":"log"}}` and save.
   From then on every run can log real forecasts.

## 4. Check it is working

- **Actions tab:** a `workflow_dispatch` run about every 30 minutes, each taking under a minute when
  nothing is due.
- **Run summary:** each run's heartbeat shows the next fixture and the hours until its window opens.
- **Issues tab:** an issue titled "Prospective logger alert" appears if a fixture's window closes
  without a record, or a window is open and no run has succeeded for 3 hours.
- **Results file and odds archive (amendment A6):** the summary also shows the results file's match
  count against football-data's, and how many of the current gameweek's fixtures have an archived odds
  snapshot. The alert issue also opens if the results file stays behind for 2 days in a row, or a fixture
  is within 24 hours of kickoff with no odds snapshot. Repeated runs with the same alert don't add comments.
- **Minutes:** 48 runs a day is about 1,440 Actions minutes a month, plus at most 360 for the backup
  schedule, within the 2,000 free minutes. You can check usage under **Settings → Billing**.

## 4b. The Odds API key (amendment A7)

The logger takes a market snapshot from The Odds API whenever it logs a forecast.

1. Sign up for the free plan at the-odds-api.com and copy the key from the email.
2. In the repo, open **Settings → Secrets and variables → Actions → Secrets → New repository secret**.
   Name it `ODDS_API_KEY` and paste the key. Use a repository secret, not a variable (variables are
   shown in plain text) and not an environment secret (the workflow declares no environment).
3. Test it once: **Actions → Prospective forecast log → Run workflow**, mode `odds-api`. This makes one
   call (2 credits), commits the snapshot, and the run log shows the number of matches and credits left.

Without the key, everything else runs as before and the heartbeat says "no key". The free plan has
500 credits a month; normal use is about 110–160 (one call per logging run with records, plus one a day). Below 50 the calls pause until the 1st and the alert
issue says so.

## 5. Things that will stop it

- **The token expires.** Renew it before the date you set, and paste the new one into the cron
  job's Authorization header.
- **cron-job.org disables the job** after repeated failures. Its failure emails tell you.
- **The workflow file is renamed.** The URL contains `prospective.yml`.

None of these lose a fixture immediately. If the trigger stops, GitHub's backup schedule still
produces fallback records 26–32 hours before kickoff; only the fresher primary record (24–26 hours)
depends on the 30-minute trigger.
