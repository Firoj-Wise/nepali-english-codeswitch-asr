# Experiment Protocol

## 1. Before a run

1. Choose a run id: `<system id>-<yyyymmdd>-<short slug>`, for example `S4-20261021-scr-lambda05`.
2. Write the config to `experiments/<run id>/config.yaml`. The config is copied, never referenced by path, so later edits to `configs/` do not change history.
3. Record the git commit hash. Uncommitted changes are not allowed for reported runs.
4. Append a row to `experiments/registry.md` with status `RUNNING`.

## 2. During a run

- Logs go to `experiments/<run id>/run.log` through the project logger.
- Checkpoints go to `experiments/<run id>/checkpoints/`.
- For GRPO, log per step: mean reward per component, KL, mean length ratio, empty hypothesis count, and a fixed set of dev samples every `log_samples_every` steps.

## 3. After a run

1. Evaluate on dev with `csrl.application.evaluate`. Results go to `experiments/<run id>/eval_dev.json`, including `metric_version`, manifest hash, and checkpoint hash.
2. Save hypotheses per utterance to `experiments/<run id>/hyps_dev.jsonl`. Reports depend on these files, so they are never deleted.
3. Update the registry row with status `DONE` or `INVALID` and the reason.

## 4. Test evaluation

Only after all dev decisions are final. One evaluation per final system. Each test evaluation is logged in the registry with the word `TEST`. Re-running test after a change is a new experiment and must be disclosed in the report.

## 5. Reporting

Use `reports/TEMPLATE.md`. Every number cites a run id and a result file. Comparisons include paired bootstrap intervals.
