""" Progress markers the pre-annotation CLIs print, so the dashboard (app.py) can follow their steps. """


def announce_stage(step: str, event: str) -> None:
    """ Marks the start or end of a step. The dashboard matches this format (STAGE_MARKER in
        src/belhisapp/pipeline_runner.py). """
    print(f"[stage:{step}] {event}", flush=True)
