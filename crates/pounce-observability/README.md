# pounce-observability

Observability wiring for POUNCE (pounce#71): the `tracing` subscriber
install and the bridge between the structured per-iteration event and
the JSON solve report.

This crate is kept separate from the leaf `pounce-common` (which holds
the pure color palette) because the collector depends on both
`pounce_nlp::solve_statistics::IterRecord` and `tracing-subscriber`.

## Two output channels, one event

Each Newton iteration emits a single structured `tracing` event at the
`pounce::iteration` target ([`ITER_TARGET`]) carrying `iter`, `mu`,
`alpha_primal`, … The human terminal never sees this event — its visual
form is the colored fixed-width iteration table printed directly by
`pounce-algorithm`, so the text console layer filters
`pounce::iteration` out. Machines get it two ways:

- `POUNCE_LOG_FORMAT=json` → the JSON layer prints it to stderr;
- the `IterCollectorLayer` rebuilds an `IterRecord` from the event
  fields and appends it to the active `IterCaptureGuard` slot, which
  the application drains into the solve report.

The collector skips events nested inside a `restoration` span, so the
report captures only the outer solve's iterations (including
`'R'`-marked outer iters) and not the restoration sub-solve's inner IPM
iterations — matching the pre-tracing behavior exactly.

## Surface

- `init_subscriber()` — install the global subscriber for a normal
  run. Idempotent (`try_init`), so multiple frontends or repeated
  Python imports are safe. Reads `RUST_LOG` (filtering, default
  `info`), `POUNCE_LOG_FORMAT` (`text` | `json`), and
  `NO_COLOR` / `CLICOLOR_FORCE` (color policy). Also bridges the `log`
  crate into `tracing` so transitive `log::*` call sites obey
  `RUST_LOG`.
- `init_for_tests()` — same layers, for test setup.
- `IterCaptureGuard::start()` / `.finish()` — per-solve capture slot;
  `finish` returns the collected `Vec<IterRecord>`.
- `IterCollectorLayer` — the `tracing-subscriber` layer that reconstructs
  iteration records from `pounce::iteration` events.
- `iteration_event_wanted()` — whether the per-iteration event should be
  emitted at all (true when JSON logging is on or a capture is active).
- `ITER_TARGET` / `RESTORATION_SPAN` — the target and span-name constants.

## Environment variables

| Variable | Effect |
|---|---|
| `RUST_LOG` | Verbosity / per-target filtering (default `info`). |
| `POUNCE_LOG_FORMAT` | `text` (default) or `json` (structured stderr sink). |
| `NO_COLOR` / `CLICOLOR_FORCE` | Disable / force colored output. |

## License

EPL-2.0.
