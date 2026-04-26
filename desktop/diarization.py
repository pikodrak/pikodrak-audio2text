from config import DIAR_MODEL_ID

_pipeline_cache = None
_pipeline_token = None


def speaker_constraints(speaker_mode, custom_count=2):
    """Return (min_speakers, max_speakers) tuple for pyannote, or (None, None)."""
    if speaker_mode == "Two persons":
        return 2, 2
    if speaker_mode == "Custom":
        return max(2, custom_count), max(2, custom_count)
    return None, None


def normalize_speaker(label, speaker_mode="Auto", two_person_names=None, label_map=None):
    """Map a pyannote speaker label to a display name.

    label_map is a session-level dict that persists assignments across chunks so
    the same physical speaker keeps the same display name throughout a session.
    """
    if label_map is not None and label in label_map:
        return label_map[label]

    try:
        idx = int(label.split("_")[-1])
    except (ValueError, IndexError):
        name = label
        if label_map is not None:
            label_map[label] = name
        return name

    if speaker_mode == "Two persons" and two_person_names:
        a_name, b_name = two_person_names
        if label_map is not None:
            assigned = set(label_map.values())
            if a_name not in assigned:
                name = a_name
            elif b_name not in assigned:
                name = b_name
            else:
                name = a_name if idx == 0 else b_name
            label_map[label] = name
            return name
        return a_name if idx == 0 else b_name

    name = f"Speaker {idx + 1}"
    if label_map is not None:
        label_map[label] = name
    return name


def assign_speakers(segments, diarization, offset=0.0,
                    speaker_mode="Auto", two_person_names=None, label_map=None):
    """Return list of (display_speaker, text) for each non-empty segment."""
    turns = list(diarization.itertracks(yield_label=True))
    result = []
    for seg in segments:
        if not seg.text.strip():
            continue
        seg_start = offset + seg.start
        seg_end = offset + seg.end
        best_speaker = "Speaker ?"
        best_overlap = 0.0
        for turn, _, raw_label in turns:
            overlap = max(0.0, min(seg_end, turn.end) - max(seg_start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = raw_label
        display = normalize_speaker(best_speaker, speaker_mode, two_person_names, label_map)
        result.append((display, seg.text.strip()))
    return result


def format_labeled_segments(labeled):
    """Group consecutive same-speaker segments into one display line each."""
    if not labeled:
        return ""
    lines = []
    cur_speaker, cur_parts = None, []
    for speaker, text in labeled:
        if speaker != cur_speaker:
            if cur_parts:
                lines.append(f"[{cur_speaker}] {' '.join(cur_parts)}")
            cur_speaker, cur_parts = speaker, [text]
        else:
            cur_parts.append(text)
    if cur_parts:
        lines.append(f"[{cur_speaker}] {' '.join(cur_parts)}")
    return "\n".join(lines)


def format_timestamped(segments):
    """Return segments as '[HH:MM:SS] text' lines."""
    lines = []
    for seg in segments:
        if not seg.text.strip():
            continue
        h = int(seg.start // 3600)
        m = int((seg.start % 3600) // 60)
        s = int(seg.start % 60)
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {seg.text.strip()}")
    return "\n".join(lines)


def get_pipeline(hf_token):
    """Load (or return cached) pyannote diarization pipeline."""
    global _pipeline_cache, _pipeline_token

    if _pipeline_cache is not None and _pipeline_token == hf_token:
        return _pipeline_cache

    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        raise RuntimeError(
            "Speaker diarization requires pyannote.audio.\n\n"
            "Install it with:\n"
            "    pip install pyannote.audio\n\n"
            "Note: this pulls in PyTorch (~2 GB) and is not available\n"
            "in the bundled EXE. Run from source to use diarization."
        )
    if not hf_token:
        raise ValueError(
            "Speaker diarization requires a HuggingFace token.\n\n"
            "Steps:\n"
            "1. Create a free account at huggingface.co\n"
            "2. Accept the model license at:\n"
            "   huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Generate a token at huggingface.co/settings/tokens\n"
            "4. Enter the token in Settings → Diarization"
        )

    pipeline = PyannotePipeline.from_pretrained(
        DIAR_MODEL_ID,
        use_auth_token=hf_token,
    )
    _pipeline_cache = pipeline
    _pipeline_token = hf_token
    return _pipeline_cache


def reset_cache():
    """Force pipeline reload on next get_pipeline() call."""
    global _pipeline_cache, _pipeline_token
    _pipeline_cache = None
    _pipeline_token = None
