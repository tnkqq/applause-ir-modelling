# Diploma project instructions
This repository is used for a diploma project:
"Система мониторинга температурных аномалий с использованием БПЛА и инфракрасных датчиков".

## Goal
Work with the infrared sensor simulation codebase and support diploma experiments.

## Domain context
The project focuses on:
- mathematical modeling of an IR sensor;
- synthetic thermal image generation;
- temperature anomaly detection scenarios;
- signal formation, noise, non-uniformity, defective pixels;
- analysis of SNR and related metrics.

## Rules
- Do not aggressively restructure the repository.
- Prefer minimal and reversible changes.
- Before editing, identify the relevant files and explain why they matter.
- Save all generated outputs under `results/`.
- Save short reports under `results/reports/`.
- When running experiments, always save:
  - parameters,
  - metrics,
  - figures,
  - a short markdown summary.

## Validation
- Run existing tests if available.
- If there are no tests, run the minimal reproducible experiment and report what happened.

## Diploma-specific expectations
- Keep variable names and comments technically precise.
- Preserve physical meaning of equations and parameters.
- If assumptions are introduced, document them explicitly.