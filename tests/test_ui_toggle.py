"""tests/test_ui_toggle.py - the address-heuristic on/off switch in the web UI."""

import io

import pytest

pytest.importorskip("flask")
fitz = pytest.importorskip("fitz")

import ui.server as srv  # noqa: E402

ADDRESS_LINE = "No. 5 Jalan Mawar Taman Indah 40000 Shah Alam Selangor Darul Ehsan"


class _FakePredictor:
    """A model that predicts nothing, so only the heuristic can create ADDRESS."""

    def predict_document_with_confidence(self, document):
        return (
            [["O"] * len(p.tokens) for p in document.pages],
            [[0.99] * len(p.tokens) for p in document.pages],
        )


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("PDFSAN_NO_ADDRESS_HEURISTIC", raising=False)
    monkeypatch.setattr(srv, "_get_predictor", lambda path: _FakePredictor())
    return srv.app.test_client()


@pytest.fixture()
def pdf_bytes():
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), ADDRESS_LINE, fontsize=10)
    return doc.tobytes()


def _post(client, route, pdf_bytes, **fields):
    data = {"file": (io.BytesIO(pdf_bytes), "t.pdf"), **fields}
    return client.post(route, data=data, content_type="multipart/form-data").get_json()


def _upload_addresses(j):
    return sum(1 for e in j["entities"] if e["type"] == "ADDRESS")


def _dump_address_tokens(j):
    return sum(1 for p in j["pages"] for t in p["tokens"] if t["label"].endswith("ADDRESS"))


def test_upload_slider_on_and_off(client, pdf_bytes):
    on = _post(client, "/api/upload", pdf_bytes, address_heuristic="true")
    off = _post(client, "/api/upload", pdf_bytes, address_heuristic="false")
    assert _upload_addresses(on) >= 1 and on["address_heuristic"] is True
    assert _upload_addresses(off) == 0 and off["address_heuristic"] is False


def test_iob2_slider_on_and_off_with_and_without_model(client, pdf_bytes):
    for no_model in ("false", "true"):
        on = _post(client, "/api/iob2-dump", pdf_bytes, address_heuristic="true", no_model=no_model)
        off = _post(client, "/api/iob2-dump", pdf_bytes, address_heuristic="false", no_model=no_model)
        assert _dump_address_tokens(on) >= 1 and on["address_heuristic"] is True
        assert _dump_address_tokens(off) == 0 and off["address_heuristic"] is False


def test_missing_field_defaults_to_on(client, pdf_bytes):
    assert _upload_addresses(_post(client, "/api/upload", pdf_bytes)) >= 1


def test_env_variable_is_the_fallback_and_the_request_overrides_it(client, pdf_bytes, monkeypatch):
    monkeypatch.setenv("PDFSAN_NO_ADDRESS_HEURISTIC", "1")
    assert _upload_addresses(_post(client, "/api/upload", pdf_bytes)) == 0
    assert _upload_addresses(_post(client, "/api/upload", pdf_bytes, address_heuristic="true")) >= 1


def test_one_request_does_not_change_the_next(client, pdf_bytes):
    _post(client, "/api/upload", pdf_bytes, address_heuristic="false")
    assert _upload_addresses(_post(client, "/api/upload", pdf_bytes)) >= 1


@pytest.mark.parametrize("route", ["/", "/iob2"])
def test_both_pages_contain_the_slider(client, route):
    assert "heuristic-toggle" in client.get(route).get_data(as_text=True)
