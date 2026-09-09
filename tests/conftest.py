"""Explicit opt-in fixture routing; source-only tests never request this fixture."""
import pytest

from scripts.historical_product_fixtures import (
    HistoricalProductUnavailable,
    load_product,
)


@pytest.fixture
def historical_product(request, monkeypatch):
    try:
        product = load_product(request.node.nodeid)
    except HistoricalProductUnavailable as exc:
        pytest.skip(str(exc))
    # Only module path constants listed in the reviewed test contract are routed.
    # Numerical assertions and their exceptions are not intercepted.
    for name, value in product.bindings.items():
        monkeypatch.setattr(request.module, name, value)
    return product
