"""Unit tests for the LocalAgreement streaming core (no model / audio needed)."""

import numpy as np

from streaming import Word, HypothesisBuffer, OnlineASR, assign_by_turns


def W(start, end, text):
    return Word(start, end, text)


def test_commits_only_on_agreement():
    hb = HypothesisBuffer()
    # First pass: nothing committed yet (no previous buffer to agree with).
    hb.insert([W(0.0, 0.5, "hello"), W(0.5, 1.0, "world")])
    assert hb.flush() == []
    assert [w.text for w in hb.buffer] == ["hello", "world"]

    # Second pass agrees on "hello world" and extends with "again".
    hb.insert([W(0.0, 0.5, "hello"), W(0.5, 1.0, "world"), W(1.0, 1.4, "again")])
    committed = hb.flush()
    assert [w.text for w in committed] == ["hello", "world"]
    assert [w.text for w in hb.buffer] == ["again"]


def test_disagreement_revises_tail():
    hb = HypothesisBuffer()
    hb.insert([W(0.0, 0.5, "the"), W(0.5, 1.0, "kat")])   # misheard "cat"
    hb.flush()
    # Next pass corrects the tail before it was ever committed.
    hb.insert([W(0.0, 0.5, "the"), W(0.5, 1.0, "cat"), W(1.0, 1.5, "sat")])
    committed = hb.flush()
    assert [w.text for w in committed] == ["the"]          # only "the" agreed
    assert [w.text for w in hb.buffer] == ["cat", "sat"]    # revised tail


def test_normalisation_ignores_punctuation_and_case():
    hb = HypothesisBuffer()
    hb.insert([W(0.0, 0.5, "Hello,")])
    hb.flush()
    hb.insert([W(0.0, 0.5, "hello"), W(0.5, 0.9, "there")])
    committed = hb.flush()
    # Agreement succeeds despite the case/comma difference; the latest pass's
    # spelling ("hello") is the one committed.
    assert [w.text for w in committed] == ["hello"]


def test_ngram_dedup_drops_repeated_committed_tail():
    hb = HypothesisBuffer()
    hb.last_committed_time = 1.0
    hb.committed_in_buffer = [W(0.0, 0.5, "good"), W(0.5, 1.0, "morning")]
    # New pass re-emits "morning" (overlap) then continues.
    hb.insert([W(0.6, 1.0, "morning"), W(1.0, 1.4, "everyone")])
    assert [w.text for w in hb.new] == ["everyone"]


class _FakeModel:
    """Returns a deterministic word list for a prefix of a fixed sentence,
    growing as the audio buffer grows — mimics Whisper on a streaming buffer."""

    SENTENCE = [(0.0, 0.4, "one"), (0.4, 0.8, "two"), (0.8, 1.2, "three"),
                (1.2, 1.6, "four"), (1.6, 2.0, "five")]

    def transcribe_fn(self, audio):
        secs = len(audio) / 16000
        return [Word(s, e, t) for (s, e, t) in self.SENTENCE if e <= secs + 1e-6]


def test_online_asr_commits_incrementally():
    fake = _FakeModel()
    asr = OnlineASR(fake.transcribe_fn, sample_rate=16000, buffer_trim_sec=100)
    all_committed = []
    # Feed audio in 0.4s blocks; each pass should eventually commit stable words.
    for _ in range(6):
        asr.insert_audio(np.zeros(int(16000 * 0.4), dtype="float32"))
        committed, _tail = asr.process()
        all_committed.extend(committed)
    # Final flush commits the last tentative word.
    all_committed.extend(asr.finish())
    assert [w.text for w in all_committed] == ["one", "two", "three", "four", "five"]


def test_online_asr_trims_buffer():
    fake = _FakeModel()
    asr = OnlineASR(fake.transcribe_fn, sample_rate=16000, buffer_trim_sec=1.0)
    for _ in range(6):
        asr.insert_audio(np.zeros(int(16000 * 0.4), dtype="float32"))
        asr.process()
    # Buffer must have been trimmed below the full 2.4s of fed audio.
    assert asr.buffered_seconds() < 2.4
    assert asr.offset > 0.0


def test_online_asr_caps_buffer_when_nothing_commits():
    # A model that never returns words (e.g. unintelligible noise) must not let
    # the audio buffer grow without bound.
    asr = OnlineASR(lambda audio: [], sample_rate=16000, buffer_trim_sec=2.0)
    for _ in range(20):  # feed 10s of audio in 0.5s blocks
        asr.insert_audio(np.zeros(int(16000 * 0.5), dtype="float32"))
        asr.process()
    assert asr.buffered_seconds() <= 2.5   # capped near buffer_trim_sec
    assert asr.offset > 0.0


def test_assign_by_turns_names_by_first_appearance():
    words = [W(0.0, 1.0, "a"), W(1.0, 2.0, "b"), W(2.0, 3.0, "c")]
    turns = [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_01"},
             {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_00"}]
    assign_by_turns(words, turns)
    # Whoever speaks first (local SPEAKER_01 here) becomes Speaker 1.
    assert words[0].speaker == "Speaker 1"
    assert words[2].speaker == "Speaker 2"


def test_assign_by_turns_overlap_and_gap_snap():
    words = [W(0.0, 1.0, "a"), W(1.2, 1.4, "gap"), W(2.0, 3.0, "c")]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "S0"},
             {"start": 1.5, "end": 3.0, "speaker": "S1"}]
    assign_by_turns(words, turns)
    assert words[0].speaker == "Speaker 1"   # overlaps S0
    assert words[2].speaker == "Speaker 2"   # overlaps S1
    # "gap" falls between turns and snaps to the nearest one (S1 at 1.5).
    assert words[1].speaker == "Speaker 2"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
