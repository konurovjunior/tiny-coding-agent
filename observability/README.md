# Self-hosted Langfuse for Tiny Coding Agent (lesson 14/24)

`./setup.sh` writes `.env` (random secrets, git-ignored) on the first run and starts Langfuse v4 with Docker:
`langfuse-web` (UI + API, http://localhost:3000) and `langfuse-worker`, plus their stores:

| Container | Why it is there |
|---|---|
| Postgres | projects, users, API keys, prompts, score configs: small, relational, transactional |
| ClickHouse | traces, observations, scores: written once, read in aggregates (a column store) |
| Redis | the queue between the API (accepts spans fast) and the worker (writes them to ClickHouse) |
| MinIO | S3-compatible storage for the raw event batches and large payloads |

`docker-compose.yml` is Langfuse's official file, unchanged (copied 2026-11-09); `docker-compose.override.yml` binds
the web UI and MinIO to localhost. The AI gateway service in the official file is not started.
Login: `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD` in `.env` (headless init, no sign-up).

- Stop: `./setup.sh stop` (data kept in Docker volumes) · remove: `./setup.sh down` · wipe: `docker compose down -v`
- Send traces: `python eval_suite.py --runs 3 --trace langfuse`, `python jobs.py nightly --trace langfuse`
- Check them: `python langfuse_check.py --session <label> --results results/eval-<label>.jsonl`
- Langfuse Cloud instead: set `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`; nothing else changes.

v4 runs in "events only" mode: the old read endpoints (`/api/public/traces`, `/api/public/scores`) answer 404;
read with `/api/public/v2/observations`; scores are still written with `POST /api/public/scores`.
