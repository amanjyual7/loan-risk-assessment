"""The leakage exclusion list.

Kept in one importable place rather than inline in a notebook, because it is
both a modelling artefact and a README deliverable: `python
notebooks/leakage.py` prints the markdown table that goes in the model card.

Three kinds of exclusion, and the third is the one people miss.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. POST-ORIGINATION — only exists after the loan was funded. Including any
#    of these is straightforward target leakage: they are consequences of
#    repayment, not predictors of it.
# ---------------------------------------------------------------------------
POST_ORIGINATION = {
    "total_pymnt": "total paid to date — a defaulted loan has paid less",
    "total_pymnt_inv": "same, investor share",
    "total_rec_prncp": "principal received — near-perfectly separates outcomes",
    "total_rec_int": "interest received",
    "total_rec_late_fee": "late fees charged, i.e. the delinquency itself",
    "recoveries": "post charge-off recovery — nonzero ONLY for defaults",
    "collection_recovery_fee": "same",
    "last_pymnt_d": "date of last payment — early date means it stopped paying",
    "last_pymnt_amnt": "amount of last payment",
    "next_pymnt_d": "only populated for live loans",
    "last_credit_pull_d": "bureau pull date, moves with collections activity",
    "last_fico_range_low": "bureau score AFTER the loan performed",
    "last_fico_range_high": "same",
    "out_prncp": "outstanding principal — a function of how much was repaid",
    "out_prncp_inv": "same, investor share",
    "collections_12_mths_ex_med": "collections activity during the loan",
    "debt_settlement_flag": "settlement is an outcome, not a predictor",
    "settlement_status": "same",
    "settlement_date": "same",
    "settlement_amount": "same",
    "settlement_percentage": "same",
    "settlement_term": "same",
    "hardship_flag": "hardship plans are granted after distress appears",
    "hardship_status": "same",
    "hardship_type": "same",
    "deferral_term": "same",
    "payment_plan_start_date": "same",
    "pymnt_plan": "same",
}

# ---------------------------------------------------------------------------
# 2. THE LENDER'S OWN RISK ASSESSMENT — not leakage in the temporal sense,
#    but circular. Grade, sub-grade and interest rate ARE Lending Club's
#    model output. Predicting default from them measures how well we can
#    reproduce their scorecard, which is not the question asked.
# ---------------------------------------------------------------------------
LENDER_ASSESSMENT = {
    "grade": "Lending Club's own risk grade — circular",
    "sub_grade": "finer-grained version of the same",
    "int_rate": "priced directly off grade, so it encodes the grade",
    "installment": "a deterministic function of amount, term and int_rate",
}

# ---------------------------------------------------------------------------
# 3. IDENTIFIERS, FREE TEXT AND POLICY ARTEFACTS — no predictive content, or
#    predictive only through something we should not be using.
# ---------------------------------------------------------------------------
NON_FEATURES = {
    "id": "identifier",
    "member_id": "identifier",
    "url": "identifier",
    "desc": "borrower free text — invites overfitting and is unavailable here",
    "title": "free text duplicate of purpose",
    "emp_title": "free-text employer; high cardinality and a proxy for "
                 "protected attributes",
    "zip_code": "geography — excluded by design, see GEOGRAPHIC below",
    "addr_state": "geography — excluded by design",
    "policy_code": "constant within the public data",
    "application_type": "joint applications have a different schema entirely",
    "verification_status": "describes Lending Club's own verification process",
    "initial_list_status": "a funding mechanic, not a borrower attribute",
    "funded_amnt": "equals loan_amnt for funded loans; keeps only the target's cohort",
    "funded_amnt_inv": "same",
}

# ---------------------------------------------------------------------------
# 4. GEOGRAPHIC — excluded as a deliberate design decision rather than for
#    statistical reasons. Geographic risk pricing is the structural form of
#    redlining. In the Indian context a pincode proxies caste and religion
#    closely enough that the model would learn them from location alone.
#    Location is used for expense estimation and display only.
# ---------------------------------------------------------------------------
GEOGRAPHIC = {
    "zip_code": "redlining risk; used for expense estimation only in the app",
    "addr_state": "same",
}

ALL_EXCLUSIONS: dict[str, dict[str, str]] = {
    "Post-origination (target leakage)": POST_ORIGINATION,
    "Lender's own assessment (circular)": LENDER_ASSESSMENT,
    "Identifiers and free text": NON_FEATURES,
    "Geographic (design decision)": GEOGRAPHIC,
}


def excluded_columns() -> set[str]:
    out: set[str] = set()
    for group in ALL_EXCLUSIONS.values():
        out |= set(group)
    return out


def markdown_table() -> str:
    lines = []
    for heading, group in ALL_EXCLUSIONS.items():
        lines.append(f"\n**{heading}**\n")
        lines.append("| Column | Why it is excluded |")
        lines.append("| --- | --- |")
        for column, reason in sorted(group.items()):
            lines.append(f"| `{column}` | {reason.replace(chr(10), ' ')} |")
    return "\n".join(lines)


def audit(columns) -> dict[str, list[str]]:
    """Check a dataframe's columns against the list.

    Reports both directions: excluded columns still present (a bug), and
    excluded columns absent from the data (harmless, but worth knowing the
    list has drifted from the dataset version).
    """
    present = set(columns)
    excluded = excluded_columns()
    return {
        "still_present": sorted(present & excluded),
        "not_in_data": sorted(excluded - present),
        "kept": sorted(present - excluded),
    }


if __name__ == "__main__":
    print(markdown_table())
    print(f"\n{len(excluded_columns())} columns excluded in total.")
