# Tiny Coding Agent: notes for agents

A small Python repair agent used to study agent reliability (durable runs, approval gate, queue, evals, guards,
injection defences, tracing, MCP, A2A). Public repo: github.com/konurovjunior/tiny-coding-agent.

- **This folder is the only copy.** Until 2026-11-18 the project lived inside the owner's private knowledge base;
  it was moved here with its history. Make every change and commit here, not in the old location.
- **Tests before every commit:** `python -m unittest discover -p 'test_*.py'` (scripted model, no API key, no cost).
  CI runs the same on every push.
- **Numbers are evidence.** Every number in README.md must come from a file in `results/`. A new live run writes its
  own `results/` file; update the README table from it and keep the "What these numbers do not show" list honest.
- **Never commit secrets.** API keys live in `.env` (git-ignored). `sk-CANARY-SECRET-42` in injection_suite.py is a
  deliberate fake.
- Code from Rohit Ghumare's course (MIT) stays credited in LICENSE and README: cli.py, examples/, vendor/.
- Commit author for this repo: the GitHub noreply address set in `.git/config`, never a work email.
