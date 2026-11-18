"""How a known person looks in words, and bringing the rest of their skin to the face."""

import numpy as np

from pixora_service import appearance
from pixora_service.faces import _ARCFACE_DST

SKIN = (190, 140, 110)
DARKER = (150, 100, 75)
WALL = (120, 90, 170)
BACKGROUND = (60, 140, 70)


def flat(h, w, colour):
    return np.full((h, w, 3), colour, dtype=np.float32)


def test_skin_is_read_from_the_cheeks_of_an_aligned_face():
    samples = appearance.skin_samples(flat(112, 112, SKIN))
    assert samples is not None and len(samples) > 500
    lightness, a, b = np.median(samples, axis=0)
    assert 55 < lightness < 65 and a > 10 and b > 15  # a warm, medium tone


def test_a_face_in_shadow_or_blown_out_gives_no_skin():
    assert appearance.skin_samples(flat(112, 112, (10, 8, 6))) is None
    assert appearance.skin_samples(flat(112, 112, (255, 255, 255))) is None


def test_glasses_and_stubble_are_left_out_of_the_skin():
    face = flat(112, 112, SKIN)
    face[62:76, 28:46] = (20, 20, 20)  # one cheek hidden, say by a hand
    samples = appearance.skin_samples(face)
    assert samples is not None and np.median(samples, axis=0)[0] > 55


def logits(*scores):
    """One reference photo's scores for black, dark brown, light brown, blond, red, grey, white, bald."""
    return np.array([scores], dtype=np.float32)


def test_hair_colour_is_said_only_when_the_photos_agree():
    assert appearance.hair_words(logits(5, 1, 0, 0, 0, 0, 0, 0)) == "black hair"
    assert appearance.hair_words(logits(1, 5, 0, 0, 0, 0, 0, 0)) == "dark brown hair"
    assert appearance.hair_words(logits(0, 0, 0, 6, 0, 0, 0, 0)) == "blond hair"
    assert appearance.hair_words(logits(1, 1, 1, 1, 1, 1, 1, 1)) is None  # no idea
    assert appearance.hair_words(logits(0, 0, 0, 0, 0, 0, 0, 6)) is None  # bald: nothing to describe


def test_the_person_is_described_without_contradicting_the_user():
    look = appearance.Look(person="a 30 year old woman", hair="black hair")
    assert look.words("a person in a garden") == "a 30 year old woman with black hair"
    assert look.words("a person with blonde hair") == "a 30 year old woman"  # the user said
    assert look.words("a person in a garden", "fuller") == "a 30 year old woman with a fuller, rounder build and black hair"
    assert look.words("a person", "not a build") == "a 30 year old woman with black hair"
    assert appearance.Look(person="a young boy").words("a person") == "a young boy"


def scene():
    """A face, a neck and hands of skin below it, a purple wall, and green behind."""
    points = (_ARCFACE_DST + 44).astype(np.float32)  # the aligned crop is then the square at 44..156
    before = flat(240, 200, BACKGROUND)
    before[44:156, 44:156] = SKIN  # the face
    before[170:230, 30:90] = (185, 136, 106)  # skin in the same light: a hand, say
    before[170:230, 110:170] = WALL  # clothing that is not skin
    after = before.copy()
    after[44:156, 44:156] = DARKER  # the face painted again as someone darker
    repainted = np.zeros((240, 200), dtype=np.float32)
    repainted[44:156, 44:156] = 1.0
    return points, before, after, repainted


def lab_of(image, y, x):
    return appearance.to_lab(image)[y, x]


def test_the_rest_of_the_skin_follows_the_face():
    points, before, after, repainted = scene()
    out = appearance.match_skin(before, after, points, repainted)
    hand_before, hand_after = lab_of(before, 200, 60), lab_of(out, 200, 60)
    face_before, face_after = lab_of(before, 100, 100), lab_of(after, 100, 100)
    # The hand moved the way the face did, and is now about as dark as it.
    assert hand_after[0] < hand_before[0] - 10
    assert abs((hand_after[0] - hand_before[0]) - (face_after[0] - face_before[0])) < 4
    assert abs(hand_after[0] - face_after[0]) < 8
    # The face is as painted; what is not skin is untouched.
    assert np.allclose(out[44:156, 44:156], after[44:156, 44:156])
    assert np.allclose(out[170:230, 110:170], before[170:230, 110:170], atol=1.0)
    assert np.allclose(out[0:30, :], before[0:30, :], atol=1.0)


def test_skin_is_only_matched_where_the_person_is():
    points, before, after, repainted = scene()
    figure = np.zeros((240, 200), dtype=np.float32)
    figure[:, :100] = 1.0  # the person is the left half; the hand on the right is somebody else's
    other = after.copy()
    other[170:230, 150:195] = (185, 136, 106)
    before2 = before.copy()
    before2[170:230, 150:195] = (185, 136, 106)
    out = appearance.match_skin(before2, other, points, repainted, figure)
    assert lab_of(out, 200, 60)[0] < lab_of(before2, 200, 60)[0] - 10
    assert np.allclose(out[170:230, 160:190], before2[170:230, 160:190], atol=1.0)


def test_nothing_moves_when_the_face_kept_its_tone_or_cannot_be_read():
    points, before, after, repainted = scene()
    assert appearance.match_skin(before, before.copy(), points, repainted) is not None
    same = appearance.match_skin(before, before.copy(), points, repainted)
    assert np.allclose(same, before)
    dark = after.copy()
    dark[44:156, 44:156] = (5, 5, 5)  # a face in black: no skin to read
    assert appearance.match_skin(before, dark, points, repainted) is dark
