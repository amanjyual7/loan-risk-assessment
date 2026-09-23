"""Loan Risk Assessment — Streamlit entry point.

    streamlit run app/streamlit_app.py

Routing is a small state machine in st.session_state, so entered data
survives every rerun. The model and the pincode reference are loaded once
and injected into the pure scoring function; nothing is trained at startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from app import components as ui  # noqa: E402
from app.mapping import payload_to_forms  # noqa: E402
from app.views import applicant as applicant_view  # noqa: E402
from app.views import credit as credit_view  # noqa: E402
from app.views import result as result_view  # noqa: E402
from engine.config import load_config, resolve  # noqa: E402
from engine.model import PDModel  # noqa: E402
from engine.pincode import PincodeReference  # noqa: E402
from engine.profiles import load_profiles  # noqa: E402
from engine.store import AssessmentStore  # noqa: E402

st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = {"applicant": "Applicant details", "credit": "Credit data",
         "result": "Result"}


# --------------------------------------------------------------- resources
@st.cache_resource
def get_config() -> dict:
    return load_config()


@st.cache_resource
def get_model(_cfg: dict) -> PDModel:
    """cache_resource, not cache_data: a fitted estimator is a live object
    and should be held once, not pickled on every rerun."""
    return PDModel.from_config(_cfg)


@st.cache_resource
def get_reference(_cfg: dict) -> PincodeReference:
    ref = PincodeReference.from_sqlite(
        resolve(_cfg, "pincode_reference"), _cfg["pincode"]["tier_order"]
    )
    ref.assert_unique_index()
    return ref


@st.cache_data
def get_demo_profiles(path: str) -> dict:
    return load_profiles(path)


@st.cache_resource
def get_store(path: str) -> AssessmentStore:
    return AssessmentStore(path)


# -------------------------------------------------------------------- app
def main() -> None:
    cfg = get_config()

    try:
        model = get_model(cfg)
        reference = get_reference(cfg)
    except Exception as exc:
        st.error(f"Cannot start: {exc}")
        st.stop()

    store = get_store(str(resolve(cfg, "assessments_db")))
    demos = get_demo_profiles(str(resolve(cfg, "demo_profiles")))

    st.session_state.setdefault("step", "applicant")
    ui.inject_styles()

    st.title("Loan Risk Assessment")
    st.caption(
        "An assessment tool, not a decision engine. It returns a risk band, a "
        "calibrated probability of default and a readable account of what "
        "drove the estimate. All applicant data is synthetic and typed by "
        "hand — there is no bureau or Account Aggregator integration."
    )

    if model.is_placeholder:
        st.warning(
            "**Placeholder model loaded.** The committed artefact is a "
            "hand-specified scorecard, not a trained model, so the "
            "probabilities below are illustrative. Train the real one with "
            "`notebooks/05_train.py`.",
            icon="⚠️",
        )

    _sidebar(cfg, model, demos, store)

    step = st.session_state["step"]
    ui.step_tabs(STEPS, step)

    # The band scale, shown before any band is assigned. A reader who sees
    # "Band D · High" with no scale beside it has no way to judge whether
    # that is unusual.
    ui.band_strip(cfg, current=None)

    if step == "applicant":
        applicant_view.render(reference, cfg)
    elif step == "credit":
        credit_view.render(cfg)
    else:
        result_view.render(model, reference, cfg, store)


def _sidebar(cfg: dict, model: PDModel, demos: dict, store: AssessmentStore) -> None:
    with st.sidebar:
        st.subheader("Demo profiles")
        st.caption(
            "Synthetic. Load one to see the app work without typing anything."
        )
        for key, profile in demos.items():
            if st.button(profile["label"], use_container_width=True, key=f"demo_{key}"):
                _load_payload(profile["applicant"])
                st.session_state["step"] = "result"
                st.rerun()
            st.caption(profile["blurb"])

        st.divider()
        st.subheader("Saved assessments")
        rows = store.list_recent(10)
        if not rows:
            st.caption("Nothing saved yet.")
        for row in rows:
            label = (
                f"{row['applicant_name'] or 'unnamed'} · "
                f"band {row['band'] or '—'} · {row['created_at'][:10]}"
            )
            if st.button(label, use_container_width=True, key=f"load_{row['id']}"):
                saved = store.load(row["id"])
                if saved:
                    _load_payload(saved["applicant"])
                    st.session_state["step"] = "result"
                    st.rerun()

        st.divider()
        st.subheader("Versions")
        st.caption(
            f"config `{cfg['config_version']}`  \n"
            f"model `{model.version}`  \n"
            f"dataset: {model.dataset}  \n"
            f"trained: {model.trained_at}"
        )
        if st.button("Reset form", use_container_width=True):
            for key in ("applicant_form", "credit_form", "trade_lines_df",
                        "resolved_location"):
                st.session_state.pop(key, None)
            st.session_state["step"] = "applicant"
            st.rerun()


def _load_payload(payload: dict) -> None:
    applicant_form, credit_form, grid = payload_to_forms(payload)
    st.session_state["applicant_form"] = applicant_form
    st.session_state["credit_form"] = credit_form
    st.session_state["trade_lines_df"] = grid
    # Widget keys must be cleared or Streamlit keeps the old values.
    for key in ("pincode_input", "manual_state_select", "trade_lines_editor"):
        st.session_state.pop(key, None)


if __name__ == "__main__":
    main()
