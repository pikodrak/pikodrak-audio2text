"""Real-time streaming transcription engine.

Implements the *LocalAgreement-2* policy (Macháček et al., "Turning Whisper into
Real-Time Transcription System", 2023): the whole growing audio buffer is
re-transcribed every step, and a word is **committed** only once two consecutive
passes agree on it. Committed words never change; the still-unstable tail is
shown to the user as *tentative* text that keeps getting rewritten until it
stabilises. This is what lets the GUI "edit what it already wrote so it makes
sense" while keeping latency low (a word is usually committed 1-2 passes — i.e.
a second or two — after it is spoken).

The engine is deliberately framework-free: it knows nothing about tkinter,
faster-whisper or pyannote. You feed it audio and a transcription callable; it
hands back committed/tentative word lists. That keeps the core unit-testable
without a model or a sound card (see test_streaming.py).
"""

from __future__ import annotations

import numpy as np


# Punctuation stripped before comparing tokens, so "cat" == "cat," == "Cat".
_PUNCT = ".,!?;:…\"'-)(_/„“”‘’–—"


def _norm(word: str) -> str:
    """Normalise a token for agreement comparison (case/space/punct insensitive)."""
    return word.strip().lower().strip(_PUNCT)


class Word:
    """A single transcribed word with absolute timestamps (seconds from start)."""

    __slots__ = ("start", "end", "text", "speaker")

    def __init__(self, start: float, end: float, text: str, speaker=None):
        self.start = start
        self.end = end
        self.text = text          # raw text, keeps leading space from whisper
        self.speaker = speaker     # filled in later by the diarizer (or None)

    def __repr__(self):
        return f"Word({self.start:.2f},{self.end:.2f},{self.text!r})"


class HypothesisBuffer:
    """LocalAgreement-2 confirmation buffer working on (start, end, text) tuples."""

    def __init__(self):
        self.committed_in_buffer: list[Word] = []  # committed, still near the tail
        self.buffer: list[Word] = []               # previous pass, awaiting agreement
        self.new: list[Word] = []                  # current pass
        self.last_committed_time = 0.0

    def insert(self, words: list[Word]):
        """Feed the words of a fresh transcription pass (absolute timestamps)."""
        # Only consider words starting after what we've already committed.
        self.new = [w for w in words if w.start > self.last_committed_time - 0.1]

        if not self.new:
            return
        first = self.new[0]
        if abs(first.start - self.last_committed_time) < 1.0 and self.committed_in_buffer:
            # Drop an n-gram that repeats the committed tail (Whisper re-emits the
            # overlap region). Compare up to 5 trailing committed words.
            cn = len(self.committed_in_buffer)
            nn = len(self.new)
            for i in range(1, min(cn, nn, 5) + 1):
                tail = " ".join(_norm(self.committed_in_buffer[-j].text)
                                for j in range(i, 0, -1))
                head = " ".join(_norm(self.new[j].text) for j in range(i))
                if tail == head:
                    del self.new[:i]
                    break

    def flush(self) -> list[Word]:
        """Commit the longest common prefix shared by this and the previous pass."""
        committed: list[Word] = []
        while self.new and self.buffer:
            if _norm(self.new[0].text) == _norm(self.buffer[0].text):
                w = self.new.pop(0)
                self.buffer.pop(0)
                self.last_committed_time = w.end
                committed.append(w)
            else:
                break
        # Whatever is left of the current pass becomes next round's reference.
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(committed)
        return committed

    def pop_committed(self, time: float):
        """Forget committed words that end before `time` (they left the audio window)."""
        while self.committed_in_buffer and self.committed_in_buffer[0].end <= time:
            self.committed_in_buffer.pop(0)


class OnlineASR:
    """Drives a transcription model over a growing, periodically-trimmed buffer.

    transcribe_fn(audio: np.ndarray[float32]) -> list[Word]
        Words must carry timestamps **relative to the start of `audio`**.
    """

    def __init__(self, transcribe_fn, sample_rate=16000, buffer_trim_sec=15.0):
        self.transcribe_fn = transcribe_fn
        self.sr = sample_rate
        self.buffer_trim_sec = buffer_trim_sec
        self.audio = np.zeros(0, dtype="float32")
        self.offset = 0.0            # absolute time of audio[0]
        self.hyp = HypothesisBuffer()
        self.committed: list[Word] = []   # every word ever committed, in order

    def insert_audio(self, chunk: np.ndarray):
        if chunk is None or len(chunk) == 0:
            return
        self.audio = np.concatenate([self.audio, chunk.astype("float32")])

    def buffered_seconds(self) -> float:
        return len(self.audio) / self.sr

    def process(self):
        """Run one transcription pass. Returns (newly_committed, tentative_tail)."""
        if len(self.audio) < self.sr * 0.4:
            return [], list(self.hyp.buffer)

        rel_words = self.transcribe_fn(self.audio)
        abs_words = [Word(w.start + self.offset, w.end + self.offset, w.text)
                     for w in rel_words]
        self.hyp.insert(abs_words)
        committed = self.hyp.flush()
        self.committed.extend(committed)
        self._maybe_trim()
        return committed, list(self.hyp.buffer)

    def _maybe_trim(self):
        if self.buffered_seconds() <= self.buffer_trim_sec:
            return
        cut = self.hyp.last_committed_time
        if cut <= self.offset:
            # Nothing committed inside the window yet it's over the cap (e.g. a
            # long stretch the decoder never stabilises). Force-drop the oldest
            # overflow so the buffer can't grow without bound.
            overflow = self.buffered_seconds() - self.buffer_trim_sec
            cut_samples = min(int(overflow * self.sr), len(self.audio))
        else:
            self.hyp.pop_committed(cut)
            cut_samples = min(int((cut - self.offset) * self.sr), len(self.audio))
        if cut_samples > 0:
            self.audio = self.audio[cut_samples:]
            self.offset += cut_samples / self.sr

    def finish(self) -> list[Word]:
        """Flush the tentative tail as committed (call once recording stops)."""
        remaining = list(self.hyp.buffer)
        self.committed.extend(remaining)
        self.hyp.buffer = []
        return remaining


def assign_by_turns(words, turns, name_fn=None):
    """Label `words` in place from a SINGLE diarization of the whole recording.

    Because the turns come from diarizing the entire audio at once (not separate
    windows), pyannote's speaker labels are already self-consistent — so we just
    map them to display names by order of first appearance and tag each word with
    the turn it overlaps most. This is far more reliable for "who said what" than
    stitching independently-diarized windows.

    turns: list of {"start", "end", "speaker"} with absolute timestamps.
    """
    name_fn = name_fn or (lambda i: f"Speaker {i + 1}")
    # Order raw speaker labels by when each first speaks -> stable display names.
    first_seen = {}
    for t in sorted(turns, key=lambda x: x["start"]):
        first_seen.setdefault(t["speaker"], t["start"])
    order = sorted(first_seen, key=lambda s: first_seen[s])
    name_of = {raw: name_fn(i) for i, raw in enumerate(order)}

    for w in words:
        best_label, best_ov = None, 0.0
        for t in turns:
            ov = min(w.end, t["end"]) - max(w.start, t["start"])
            if ov > best_ov:
                best_ov, best_label = ov, t["speaker"]
        if best_label is None and turns:
            # Word fell in a gap (e.g. between turns) — snap to the nearest turn.
            mid = (w.start + w.end) / 2
            best_label = min(
                turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"]))
            )["speaker"]
        if best_label is not None:
            w.speaker = name_of.get(best_label, w.speaker)


class WhisperTranscribeFn:
    """Adapts a faster-whisper model to the transcribe_fn(audio)->[Word] contract.

    Locks the detected language after the first non-empty pass so streaming does
    not flip languages mid-conversation.
    """

    def __init__(self, model, language=None, task="transcribe", vad_filter=True,
                 vad_parameters=None, beam_size=1, initial_prompt=None,
                 hallucination_silence_threshold=2.0):
        self.model = model
        self.language = language          # None => auto-detect then lock
        self.task = task
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.detected_language = language

    def __call__(self, audio: np.ndarray) -> list[Word]:
        kwargs = dict(
            language=self.language,
            task=self.task,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            # Crucial for streaming: stop the decoder from looping on the
            # repeated buffer prefix / hallucinating during silence.
            condition_on_previous_text=False,
            # Skip long silent gaps where a hallucination is detected (needs
            # word_timestamps); the most effective anti-hallucination knob.
            hallucination_silence_threshold=self.hallucination_silence_threshold,
        )
        if self.initial_prompt:
            kwargs["initial_prompt"] = self.initial_prompt
        if self.vad_filter and self.vad_parameters:
            kwargs["vad_parameters"] = self.vad_parameters

        segments, info = self.model.transcribe(audio, **kwargs)
        words: list[Word] = []
        for seg in segments:
            seg_words = getattr(seg, "words", None)
            if seg_words:
                for w in seg_words:
                    if w.word.strip():
                        words.append(Word(w.start, w.end, w.word))
            elif seg.text.strip():
                words.append(Word(seg.start, seg.end, seg.text))
        if words and self.language is None and getattr(info, "language", None):
            self.language = info.language       # lock language after first hit
        if getattr(info, "language", None):
            self.detected_language = info.language
        return words
