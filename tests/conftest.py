import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "bench_smoke: smoke benchmark tests (may be slow on CPU)")
    config.addinivalue_line("markers", "bench_full: full benchmark tests (very slow on CPU)")
