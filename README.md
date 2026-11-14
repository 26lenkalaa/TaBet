# TaBet

TaBet forecasts sports results and compares them with the betting market. The odds are a benchmark only; they are never a model input.

Each sport has its own folder:

| Folder | What's in it |
| --- | --- |
| [`soccer (premier league)/`](<soccer (premier league)/>) | Premier League home/draw/away forecasts: the model (`tabet/`), data, results, the live forecast log, tests, docs and the website (`site/`). Start with its [README](<soccer (premier league)/README.md>). |
| [`backend/`](backend/) | The API (Express on Vercel, `/api/*`) and its Supabase database, shared by every sport. See its [README](backend/README.md). |

Shared at the top level:

- `.github/workflows/`: the CI tests and the forecast logger. Each job runs inside its sport's folder.
- `api/index.mjs`: the Vercel entry point for the backend's Express app.
- `vercel.json`: deploys `soccer (premier league)/site` to https://ta-bet-ml-model-edition.vercel.app.

To work on the soccer project, run commands from its folder:

```sh
cd "soccer (premier league)"
python -m pytest -q
python -m tabet --help
```
