BENCHMARK_LEVELS = [
    {
        "id": "L1_MONO_SIMPLE",
        "description": "Single monophonic melody, fixed tempo, no accompaniment.",
        "examples": ["happy_birthday", "old_macdonald"],
        "polyphony": "monophonic",
        "dominant_voice": "single",
    },
    {
        "id": "L2_MONO_EXPRESSIVE",
        "description": "Monophonic melody with dynamics, small tempo variations, pedal-like sustain.",
        "examples": ["happy_birthday_expressive"],
        "polyphony": "monophonic",
        "dominant_voice": "single",
    },
    {
        "id": "L3_POLY_DOMINANT",
        "description": "2-3 voices; 1 clearly louder dominant melody + soft chordal accompaniment.",
        "examples": ["happy_birthday_poly_dominant"],
        "polyphony": "polyphonic_dominant",
        "dominant_voice": "melody_top",
    },
    {
        "id": "L4_POLY_FULL",
        "description": "Full polyphony: broken chords, inner voices, 3-5 notes at once.",
        "examples": ["old_macdonald_poly_full"],
        "polyphony": "polyphonic_full",
        "dominant_voice": "none",
    }
]
