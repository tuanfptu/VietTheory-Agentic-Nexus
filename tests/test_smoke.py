"""Foundation smoke tests."""

from viettheory import __version__


def test_package_imports() -> None:
    """The package is importable and exposes a semantic version."""
    parts = __version__.split(".")

    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
