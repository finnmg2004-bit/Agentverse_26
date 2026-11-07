"""
Local voice output for Nudge, shared by both processes (no sim / brain imports, 100% offline).

    from speech.tts import speak, synthesize      # Piper text-to-speech ("Nudge says")

Needs the project root on sys.path (sim/run_sim.py, nudge-brain/main.py and gui/ already add
it). The voice comes from `pixi run fetch-models` (assets/models/, gitignored). Fails soft: a
missing model, package or audio device is logged once and the call returns.
"""
