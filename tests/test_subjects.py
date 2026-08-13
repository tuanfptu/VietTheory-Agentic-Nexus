import pytest

from viettheory.subjects import SUBJECTS, ExtractionMode, get_subject


def test_registry_contains_exactly_the_five_supported_subjects() -> None:
    assert {subject.code for subject in SUBJECTS} == {
        "MLN111",
        "MLN122",
        "MLN131",
        "HCM202",
        "VNR202",
    }
    assert len(SUBJECTS) == 5


def test_registry_pins_native_and_ocr_boundaries() -> None:
    assert get_subject("mln111").extraction_mode is ExtractionMode.NATIVE
    assert get_subject("MLN122").extraction_mode is ExtractionMode.NATIVE
    assert get_subject("HCM202").extraction_mode is ExtractionMode.OCR
    assert get_subject("MLN131").extraction_mode is ExtractionMode.OCR
    assert get_subject("VNR202").extraction_mode is ExtractionMode.OCR


def test_unknown_subject_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported subject code"):
        get_subject("UNKNOWN")


def test_registry_codes_are_stable_identifiers() -> None:
    assert all(subject.code.isascii() and subject.code.isalnum() for subject in SUBJECTS)
