"""Screen 1 — applicant details.

Note on the form boundary. The spec asks for two things that pull against
each other: wrap the page in st.form so it does not rerun on every
keystroke, AND fire the pincode lookup on entry rather than on submit. A
widget inside st.form cannot trigger a rerun, so the pincode input sits
just ABOVE the form and everything else sits inside it. That keeps the
inline lookup while still avoiding a rerun per keystroke on the other
twenty fields.
"""

from __future__ import annotations

import streamlit as st

from engine.pincode import PincodeReference, is_well_formed

PURPOSES = [
    "education", "debt consolidation", "home improvement", "medical",
    "car", "major purchase", "small business", "other",
]
EMPLOYMENT = ["salaried", "self-employed", "business", "student"]


def _state(key: str, default):
    return st.session_state.setdefault(key, default)


def render(reference: PincodeReference, cfg: dict) -> None:
    st.subheader("1 · Applicant details")
    st.caption(
        "All data is typed by hand or loaded from a synthetic demo profile. "
        "No bureau or Account Aggregator integration exists in this app."
    )

    data = st.session_state.setdefault("applicant_form", {})

    # ---- pincode: outside the form so the lookup renders inline ----------
    st.markdown("**Location**")
    pin_col, info_col = st.columns([1, 2])
    with pin_col:
        pincode = st.text_input(
            "Pincode", value=data.get("pincode", ""), max_chars=6,
            key="pincode_input",
            help="The only location field you enter. State, city, region and "
                 "tier are derived from it.",
        )
    data["pincode"] = pincode

    location = None
    with info_col:
        if not pincode:
            st.caption("Enter a 6-digit pincode to resolve state, city and tier.")
        elif not is_well_formed(pincode):
            st.error(
                "Not a valid Indian pincode. Six digits, first digit 1–8.",
                icon="⚠️",
            )
        else:
            location = reference.lookup(
                pincode,
                default_tier=cfg["pincode"]["default_tier_on_miss"],
                miss_label=cfg["pincode"]["miss_label"],
                manual_state=data.get("manual_state"),
            )
            if location.assumed:
                st.warning(
                    f"Pincode not in the reference data. Tier defaulted to "
                    f"**{location.tier}** — {location.assumption_note}. This "
                    "assumption is shown on the result screen.",
                    icon="ℹ️",
                )
                data["manual_state"] = st.selectbox(
                    "Pick the state manually",
                    options=[""] + reference.states(),
                    index=0,
                    key="manual_state_select",
                ) or None
            else:
                # Read-only text, not a dropdown: the user should see what
                # matched and be able to catch a typo, not override it.
                st.markdown(
                    f"State **{location.state}** · City **{location.city}** · "
                    f"Region **{location.region}** · Tier **{location.tier}**"
                )
    st.session_state["resolved_location"] = location

    # ---- everything else inside the form ---------------------------------
    with st.form("screen1", clear_on_submit=False):
        st.markdown("**Identity**")
        c1, c2, c3 = st.columns(3)
        data["full_name"] = c1.text_input("Full name", value=data.get("full_name", ""))
        data["age"] = c2.number_input("Age", 18, 80, int(data.get("age", 32)))
        data["residence_type"] = c3.selectbox(
            "Residence type", ["owned", "rented", "parental"],
            index=["owned", "rented", "parental"].index(
                data.get("residence_type", "rented")),
        )
        data["years_at_address"] = c1.number_input(
            "Years at current address", 0.0, 80.0,
            float(data.get("years_at_address", 2.0)), step=0.5,
        )

        st.markdown("**Household**")
        h1, h2 = st.columns(2)
        data["dependent_children"] = h1.number_input(
            "Dependent children", 0, 12, int(data.get("dependent_children", 0)),
            help="Counted separately from dependent adults: a child and an "
                 "elderly parent do not carry the same cost.",
        )
        data["dependent_adults"] = h2.number_input(
            "Other dependent adults", 0, 12, int(data.get("dependent_adults", 0)),
        )

        st.markdown("**Employment**")
        e1, e2, e3 = st.columns(3)
        data["employment_type"] = e1.selectbox(
            "Employment type", EMPLOYMENT,
            index=EMPLOYMENT.index(data.get("employment_type", "salaried")),
        )
        data["employer_name"] = e2.text_input(
            "Employer or business name", value=data.get("employer_name", ""))
        data["industry"] = e3.text_input("Industry", value=data.get("industry", ""))
        data["years_current_job"] = e1.number_input(
            "Years in current job", 0.0, 60.0,
            float(data.get("years_current_job", 2.0)), step=0.5)
        data["total_experience_years"] = e2.number_input(
            "Total work experience (years)", 0.0, 60.0,
            float(data.get("total_experience_years", 5.0)), step=0.5)

        st.markdown("**Income**")
        i1, i2, i3 = st.columns(3)
        data["gross_monthly"] = i1.number_input(
            "Gross monthly income (₹)", 0.0, 1e8,
            float(data.get("gross_monthly", 60000.0)), step=1000.0)
        data["net_monthly"] = i2.number_input(
            "Net take-home (₹)", 0.0, 1e8,
            float(data.get("net_monthly", 50000.0)), step=1000.0,
            help="Asked separately rather than inferred from gross.")
        data["other_household_income"] = i3.number_input(
            "Other household income (₹)", 0.0, 1e8,
            float(data.get("other_household_income", 0.0)), step=1000.0)
        data["income_source"] = i1.text_input(
            "Income source", value=data.get("income_source", "Salary"))
        data["income_varies"] = i2.checkbox(
            "Income varies month to month", value=bool(data.get("income_varies", False)))

        st.markdown("**Loan requested**")
        l1, l2, l3 = st.columns(3)
        data["amount"] = l1.number_input(
            "Amount (₹)", 1000.0, 1e8, float(data.get("amount", 500000.0)), step=10000.0)
        data["purpose"] = l2.selectbox(
            "Purpose", PURPOSES,
            index=PURPOSES.index(data.get("purpose", "education")))
        data["tenure_months"] = l3.number_input(
            "Tenure (months)", int(cfg["pricing"]["min_tenure_months"]),
            int(cfg["pricing"]["max_tenure_months"]),
            int(data.get("tenure_months", 60)))
        data["collateral_offered"] = l1.text_input(
            "Collateral offered", value=data.get("collateral_offered", "none"))

        st.markdown("**Co-applicant**")
        st.caption(
            "Optional, but on education loans the co-applicant usually "
            "carries the repayment capacity."
        )
        data["has_co_applicant"] = st.checkbox(
            "Add a co-applicant", value=bool(data.get("has_co_applicant", False)))
        k1, k2, k3 = st.columns(3)
        data["co_name"] = k1.text_input("Co-applicant name", value=data.get("co_name", ""))
        data["co_relationship"] = k2.text_input(
            "Relationship", value=data.get("co_relationship", ""))
        data["co_employment_type"] = k3.selectbox(
            "Co-applicant employment", EMPLOYMENT,
            index=EMPLOYMENT.index(data.get("co_employment_type", "salaried")))
        data["co_net_monthly_income"] = k1.number_input(
            "Co-applicant net monthly income (₹)", 0.0, 1e8,
            float(data.get("co_net_monthly_income", 0.0)), step=1000.0)
        data["co_age"] = k2.number_input(
            "Co-applicant age", 18, 90, int(data.get("co_age", 45)))

        submitted = st.form_submit_button("Continue to credit data →",
                                          type="primary")

    if submitted:
        st.session_state["applicant_form"] = data
        st.session_state["step"] = "credit"
        st.rerun()
