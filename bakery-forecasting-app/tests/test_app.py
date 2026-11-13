"""Smoke tests for the rendered Streamlit interface and its controls."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_loads_and_interactive_filters_rerun() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Forecast explorer",
        "Model evidence",
        "Method & limitations",
    ]
    assert len(app.selectbox) == 1
    assert len(app.select_slider) == 1
    assert len(app.toggle) == 1

    app.selectbox[0].set_value(6)
    app.select_slider[0].set_value((3, 5))
    app.toggle[0].set_value(False)
    app.run(timeout=30)

    assert not app.exception
    assert app.selectbox[0].value == 6
    assert app.select_slider[0].value == (3, 5)
    assert app.toggle[0].value is False
    assert len(app.metric) >= 7
    assert len(app.dataframe) >= 4
