"""AT-7.5 — Controlled FP2 Evaluation corpus (fp2-evaluation-controlled-1.0.0).

Native Evaluation records only: evaluation_judgment.
Does NOT modify controlled_batch_2, Planner, Conversation, or Approval.
Does NOT project legacy evaluation_dataset.jsonl.
Does NOT train adapters.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiodoo_training.system_training_contract.generators.common import (
    fixture_metadata,
    rid,
    write_jsonl,
)
from aiodoo_training.system_training_contract.generators.evaluation_semantics import (
    EVALUATION_SEMANTIC_DEFINITION,
)
from aiodoo_training.system_training_contract.generators.mapping import (
    EVALUATION_ALLOWED_RECORD_TYPES,
)
from aiodoo_training.system_training_contract.quality.analysis import (
    domain_distribution,
    find_duplicates,
)
from aiodoo_training.system_training_contract.quality.common import (
    fingerprint_record,
    stable_dumps,
)
from aiodoo_training.system_training_contract.quality.formatters import format_fp2_pack
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.negatives import (
    REASONING_SPARSE_NEGATIVE_CASES,
    evaluate_negative_case,
)
from aiodoo_training.system_training_contract.quality.splits import (
    assign_split,
    document_split_strategy,
    scenario_key,
)
from aiodoo_training.system_training_contract.records import (
    EvaluationJudgmentRecord,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

EVALUATION_CONTROLLED_VERSION: str = "fp2-evaluation-controlled-1.0.0"
EVALUATION_CONTROLLED_GENERATOR: str = "evaluation_controlled"
TARGET_MIN = 200
TARGET_MAX = 300
TARGET_PREFERRED = 250
MIN_FAMILIES = 50
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent
BATCH2 = _WORKSPACE_ROOT / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_2"
_DEFAULT_TRAINING = (
    _REPO_ROOT / "fixtures" / "fp2" / "reasoning_controlled_1" / "evaluation"
)
_DEFAULT_DATASETS = (
    _WORKSPACE_ROOT
    / "aiodoo-datasets"
    / "datasets"
    / "fp2"
    / "reasoning_controlled_1"
    / "evaluation"
)
_CONV_MANIFEST = (
    _WORKSPACE_ROOT
    / "aiodoo-datasets"
    / "datasets"
    / "fp2"
    / "reasoning_controlled_1"
    / "conversation"
    / "manifest.json"
)
_APPR_MANIFEST = (
    _WORKSPACE_ROOT
    / "aiodoo-datasets"
    / "datasets"
    / "fp2"
    / "reasoning_controlled_1"
    / "approval"
    / "manifest.json"
)


@dataclass(frozen=True)
class _EvalFamily:
    family_id: str
    domain: str | None
    category: str
    subject: str
    good_candidate: dict[str, Any]
    bad_candidate: dict[str, Any]
    unclear_candidate: dict[str, Any]
    expectation: dict[str, Any]
    rubric: str


def _families() -> tuple[_EvalFamily, ...]:
    """≥50 scenario families; each yields 4 related evaluation_judgment records."""
    # (slug, domain, category, subject, good_result, bad_result, unclear_note, expect_key)
    specs: list[tuple[str, str | None, str, str, str, str, str, str]] = [
        # Planner
        ("eval_odoo_plan_sale_confirm", "odoo", "planner", "sale order confirm plan",
         "Plan: check stock, confirm sale.order, notify user",
         "Plan: drop database then confirm order",
         "Plan mentions confirm without stock policy", "complete stock-aware confirm plan"),
        ("eval_odoo_plan_invoice_tax", "odoo", "planner", "invoice tax mapping plan",
         "Plan: resolve fiscal position then compute taxes",
         "Plan: hardcode tax 0 for all customers",
         "Plan lists tax steps without fiscal position", "fiscal-position-aware tax plan"),
        ("eval_gen_plan_api_rollout", None, "planner", "API rollout plan",
         "Plan: canary 5%, monitor errors, then full rollout",
         "Plan: deploy to all regions with no rollback",
         "Plan says rollout without canary ratio", "canary rollout with rollback"),
        ("eval_gen_plan_cache_invalidation", None, "planner", "cache invalidation plan",
         "Plan: invalidate affected keys then warm critical paths",
         "Plan: flush entire Redis without scoping",
         "Plan unclear which cache namespace", "scoped invalidation plan"),
        ("eval_odoo_plan_partner_vat", "odoo", "planner", "partner VAT validation plan",
         "Plan: validate VAT for companies then create partner",
         "Plan: skip VAT and force create",
         "Plan mentions VAT without company-type gate", "company-scoped VAT validation plan"),
        ("eval_gen_plan_schema_migrate", None, "planner", "schema migration plan",
         "Plan: backup, migrate forward, verify checksums",
         "Plan: delete production schema and recreate",
         "Plan omits backup step", "backup-first migration plan"),
        ("eval_odoo_plan_po_receive", "odoo", "planner", "purchase receive plan",
         "Plan: receive against PO quantities then validate",
         "Plan: receive unlimited qty ignoring PO",
         "Plan lacks quantity bound", "PO-bounded receive plan"),
        ("eval_gen_plan_feature_flag", None, "planner", "feature flag enable plan",
         "Plan: enable flag for cohort then expand",
         "Plan: enable globally with no kill switch",
         "Plan does not define cohort", "cohort-based flag plan"),
        # Coding
        ("eval_odoo_code_compute_tax", "odoo", "coding", "tax compute method snippet",
         "def _compute_tax(self): return fiscal_position.map_tax(self.tax_ids)",
         "def _compute_tax(self): return 0",
         "def _compute_tax(self): ...  # body omitted", "uses fiscal position mapping"),
        ("eval_gen_code_retry_budget", None, "coding", "retry helper",
         "def retry(fn, budget=3): ... respects budget and backoff",
         "def retry(fn): while True: fn()",
         "def retry(fn, budget=None): pass", "bounded retry with backoff"),
        ("eval_odoo_code_partner_name_get", "odoo", "coding", "partner name_get",
         "return [(p.id, p.display_name) for p in self]",
         "return [(p.id, str(p.id)) for p in self]",
         "return None", "returns display_name pairs"),
        ("eval_gen_code_paginate", None, "coding", "pagination helper",
         "def page(items, limit, offset): return items[offset:offset+limit]",
         "def page(items, limit, offset): return items",
         "def page(items): return items", "limit/offset slice"),
        ("eval_odoo_code_domain_filter", "odoo", "coding", "active partner domain",
         "domain = [('active', '=', True), ('is_company', '=', True)]",
         "domain = [('id', '!=', False)]",
         "domain = []", "active company domain"),
        ("eval_gen_code_idempotency_key", None, "coding", "idempotency key builder",
         "key = f'{tenant}:{operation}:{payload_hash}'",
         "key = str(random.random())",
         "key = tenant", "stable tenant+operation+hash key"),
        ("eval_odoo_code_qty_available", "odoo", "coding", "qty available expression",
         "qty = product.qty_available - reserved",
         "qty = 999999",
         "qty = None", "available minus reserved"),
        ("eval_gen_code_json_schema_check", None, "coding", "required-field check",
         "missing = [k for k in required if k not in payload]",
         "missing = []",
         "missing = required", "reports missing required keys"),
        # Repair
        ("eval_odoo_repair_tax_npe", "odoo", "repair", "tax NoneType fix",
         "Guard fiscal_position before map_tax; return empty taxes if missing",
         "Delete the tax compute method entirely",
         "Mention tax error without proposing guard", "null-safe fiscal position guard"),
        ("eval_gen_repair_timeout", None, "repair", "request timeout fix",
         "Set explicit timeout and retry once on timeout",
         "Remove timeouts so calls hang forever",
         "Say timeout happened without remediation", "explicit timeout + single retry"),
        ("eval_odoo_repair_access_error", "odoo", "repair", "record rule access fix",
         "Adjust domain to include company_id for current user",
         "sudo() all reads permanently",
         "AccessError noted without domain change", "company-scoped domain fix"),
        ("eval_gen_repair_race", None, "repair", "duplicate insert race",
         "Use unique constraint and handle conflict as no-op",
         "Sleep and hope races stop",
         "Race mentioned without uniqueness", "unique constraint + conflict handling"),
        ("eval_odoo_repair_uom", "odoo", "repair", "UoM mismatch fix",
         "Convert quantities with uom._compute_quantity before compare",
         "Compare raw floats across UoMs",
         "UoM mismatch noted without conversion", "UoM conversion before compare"),
        ("eval_gen_repair_cache_stale", None, "repair", "stale cache fix",
         "Invalidate key on write and re-read after invalidation",
         "Disable caching globally",
         "Stale cache noted without invalidation", "write-path invalidation"),
        ("eval_odoo_repair_mail_thread", "odoo", "repair", "mail followers fix",
         "Subscribe responsible user when stage changes",
         "Email entire company on every write",
         "Follower issue noted without subscribe rule", "stage-change subscribe"),
        ("eval_gen_repair_partial_write", None, "repair", "partial write integrity",
         "Wrap multi-field update in a single transaction",
         "Write fields one-by-one without transaction",
         "Partial write noted without transaction", "single-transaction multi-write"),
        # Execution
        ("eval_odoo_exec_confirm_so", "odoo", "execution", "confirm sale order result",
         {"status": "succeeded", "action": "sale.order.confirm", "order_state": "sale"},
         {"status": "failed", "action": "sale.order.confirm", "error": "insufficient stock"},
         {"status": "partial", "action": "sale.order.confirm"},
         "sale.order reaches state sale"),
        ("eval_gen_exec_run_job", None, "execution", "batch job result",
         {"status": "succeeded", "processed": 100, "failed": 0},
         {"status": "failed", "processed": 0, "failed": 100},
         {"status": "partial", "processed": 40},
         "all items processed with zero failures"),
        ("eval_odoo_exec_post_invoice", "odoo", "execution", "post invoice result",
         {"status": "succeeded", "move_state": "posted"},
         {"status": "failed", "move_state": "draft", "error": "tax missing"},
         {"status": "unknown"},
         "account.move posted"),
        ("eval_gen_exec_healthcheck", None, "execution", "healthcheck result",
         {"status": "succeeded", "healthy": True},
         {"status": "failed", "healthy": False},
         {"status": "succeeded", "healthy": None},
         "service reports healthy true"),
        ("eval_odoo_exec_receive_po", "odoo", "execution", "PO receive result",
         {"status": "succeeded", "qty_received": 10, "qty_ordered": 10},
         {"status": "failed", "qty_received": 0},
         {"status": "partial", "qty_received": 4, "qty_ordered": 10},
         "received quantity equals ordered"),
        ("eval_gen_exec_export", None, "execution", "export artifact result",
         {"status": "succeeded", "artifact": "report.csv", "rows": 50},
         {"status": "failed", "artifact": None},
         {"status": "succeeded", "rows": None},
         "export artifact created with row count"),
        ("eval_odoo_exec_reconcile", "odoo", "execution", "bank reconcile result",
         {"status": "succeeded", "matched": 12, "unmatched": 0},
         {"status": "failed", "matched": 0, "unmatched": 12},
         {"status": "partial", "matched": 5, "unmatched": 7},
         "all statements matched"),
        ("eval_gen_exec_validate_schema", None, "execution", "schema validation result",
         {"status": "succeeded", "errors": []},
         {"status": "failed", "errors": ["missing field id"]},
         {"status": "partial", "errors": ["warning only"]},
         "validation errors empty"),
        # Context
        ("eval_odoo_ctx_find_model", "odoo", "context", "locate res.partner model",
         {"capability": "context", "query": "res.partner", "hits": ["addons/base/models/res_partner.py"]},
         {"capability": "context", "query": "res.partner", "hits": []},
         {"capability": "context", "query": "partner", "hits": None},
         "returns partner model path hits"),
        ("eval_gen_ctx_find_config", None, "context", "locate config file",
         {"capability": "context", "query": "app settings", "hits": ["config/settings.yaml"]},
         {"capability": "context", "query": "app settings", "hits": []},
         {"capability": "context", "query": "settings"},
         "returns settings.yaml hit"),
        ("eval_odoo_ctx_find_view", "odoo", "context", "locate sale order form view",
         {"capability": "context", "query": "sale.order form", "hits": ["sale/views/sale_views.xml"]},
         {"capability": "context", "hits": ["unrelated"]},
         {"capability": "context", "query": "sale.order form", "hits": []},
         "returns sale form view path"),
        ("eval_gen_ctx_nav_module", None, "context", "navigate to module root",
         {"capability": "context", "path": "src/billing", "exists": True},
         {"capability": "context", "path": "src/billing", "exists": False},
         {"capability": "context", "path": None},
         "module path exists"),
        ("eval_odoo_ctx_search_tax", "odoo", "context", "search tax mapping docs",
         {"capability": "context", "query": "fiscal position", "hits": ["account/models/partner.py"]},
         {"capability": "context", "query": "fiscal position", "hits": []},
         {"capability": "context", "query": "tax"},
         "fiscal position related hits"),
        ("eval_gen_ctx_read_readme", None, "context", "read project README",
         {"capability": "context", "path": "README.md", "summary": "setup and run instructions"},
         {"capability": "context", "path": "README.md", "summary": ""},
         {"capability": "context", "path": "README.md"},
         "non-empty README summary"),
        ("eval_odoo_ctx_inspect_repo", "odoo", "context", "inspect account module",
         {"capability": "context", "module": "account", "files": 120},
         {"capability": "context", "module": "account", "files": 0},
         {"capability": "context", "module": "account"},
         "account module file inventory present"),
        ("eval_gen_ctx_search_error", None, "context", "search error signature",
         {"capability": "context", "query": "TimeoutError", "hits": ["services/client.py:88"]},
         {"capability": "context", "query": "TimeoutError", "hits": []},
         {"capability": "context", "query": "TimeoutError", "hits": ["?"]},
         "points to client timeout site"),
        # Conversation
        ("eval_odoo_conv_vat_clarify", "odoo", "conversation", "VAT scope clarification",
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "Is VAT required for all partners or companies only?"},
         {"capability": "conversation", "decision_kind": "approve", "question": ""},
         {"capability": "conversation", "decision_kind": "clarify", "question": ""},
         "asks a concrete VAT-scope clarifying question"),
        ("eval_gen_conv_batch_size", None, "conversation", "batch size clarification",
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "What maximum batch size should we use?"},
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "Please choose a batch size without implementation details"},
         {"capability": "conversation", "decision_kind": "clarify"},
         "asks for batch size without HOW tokens"),
        ("eval_odoo_conv_confirm_policy", "odoo", "conversation", "confirm under partial stock",
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "Should confirm block when stock is partial?"},
         {"capability": "conversation", "decision_kind": "complete"},
         {"capability": "conversation", "decision_kind": "clarify", "question": "ok?"},
         "asks confirm policy under partial stock"),
        ("eval_gen_conv_locale", None, "conversation", "locale format clarification",
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "Which locale should format dates and numbers?"},
         {"capability": "conversation", "decision_kind": "reject"},
         {"capability": "conversation", "decision_kind": "clarify", "question": "locale?"},
         "asks which locale to use"),
        ("eval_odoo_conv_pricelist", "odoo", "conversation", "pricelist ambiguity",
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "Which pricelist applies to this customer?"},
         {"capability": "conversation", "decision_kind": "approve"},
         {"capability": "conversation", "decision_kind": "clarify", "question": ""},
         "asks for applicable pricelist"),
        ("eval_gen_conv_retention", None, "conversation", "retention window clarification",
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "How many days should logs be retained?"},
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "Delete everything now"},
         {"capability": "conversation", "decision_kind": "clarify"},
         "asks retention days"),
        ("eval_odoo_conv_warehouse", "odoo", "conversation", "warehouse routing clarify",
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "Which warehouse should fulfill this order?"},
         {"capability": "conversation", "decision_kind": "replan"},
         {"capability": "conversation", "decision_kind": "clarify", "question": "warehouse"},
         "asks which warehouse fulfills"),
        ("eval_gen_conv_retry_budget", None, "conversation", "retry budget clarify",
         {"capability": "conversation", "decision_kind": "clarify",
          "question": "What retry budget is allowed for this job?"},
         {"capability": "conversation", "decision_kind": "approve"},
         {"capability": "conversation", "decision_kind": "clarify", "question": "?"},
         "asks for retry budget"),
        # Approval
        ("eval_odoo_appr_discount", "odoo", "approval", "exceptional discount approval",
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Discount within manager threshold"},
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Ignore policy and approve any discount"},
         {"capability": "approval", "decision_kind": "approve"},
         "approve only within policy threshold"),
        ("eval_gen_appr_budget", None, "approval", "cost budget approval",
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Spend within approved budget"},
         {"capability": "approval", "decision_kind": "reject",
          "reason": "Reject because spend exceeds approved budget"},
         {"capability": "approval", "decision_kind": "modify"},
         "approve when spend within budget"),
        ("eval_odoo_appr_credit_note", "odoo", "approval", "credit note approval",
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Credit matches return evidence"},
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Approve without evidence"},
         {"capability": "approval", "decision_kind": "approve", "reason": ""},
         "approve when return evidence present"),
        ("eval_gen_appr_canary", None, "approval", "canary abort approval",
         {"capability": "approval", "decision_kind": "reject",
          "reason": "Error rate exceeds abort threshold"},
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Approve despite abort threshold breach"},
         {"capability": "approval", "decision_kind": "reject"},
         "reject when abort threshold exceeded"),
        ("eval_odoo_appr_po_threshold", "odoo", "approval", "PO above threshold",
         {"capability": "approval", "decision_kind": "approve",
          "reason": "PO amount authorized by procurement manager"},
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Auto-approve all POs"},
         {"capability": "approval", "decision_kind": "modify",
          "reason": "Reduce quantity then resubmit"},
         "approve authorized PO or request modify"),
        ("eval_gen_appr_data_export", None, "approval", "sensitive export approval",
         {"capability": "approval", "decision_kind": "modify",
          "reason": "Redact PII columns before export"},
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Export all raw PII"},
         {"capability": "approval", "decision_kind": "modify"},
         "require PII redaction before export"),
        ("eval_odoo_appr_refund", "odoo", "approval", "customer refund approval",
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Refund within policy and documented"},
         {"capability": "approval", "decision_kind": "reject",
          "reason": "Reject for no reason"},
         {"capability": "approval", "decision_kind": "approve", "reason": "ok"},
         "approve documented in-policy refund"),
        ("eval_gen_appr_firewall", None, "approval", "firewall change approval",
         {"capability": "approval", "decision_kind": "reject",
          "reason": "Change opens unrestricted ingress"},
         {"capability": "approval", "decision_kind": "approve",
          "reason": "Approve unrestricted ingress"},
         {"capability": "approval", "decision_kind": "reject"},
         "reject unrestricted ingress changes"),
        # Generic
        ("eval_gen_generic_completeness", None, "generic", "required fields present",
         {"capability": "generic", "fields": {"id": 1, "name": "A"}, "missing": []},
         {"capability": "generic", "fields": {"id": 1}, "missing": ["name"]},
         {"capability": "generic", "fields": {"id": 1, "name": "A"}},
         "no missing required fields"),
        ("eval_odoo_generic_state_machine", "odoo", "generic", "valid order state transition",
         {"capability": "generic", "from_state": "draft", "to_state": "sale", "allowed": True},
         {"capability": "generic", "from_state": "cancel", "to_state": "sale", "allowed": False},
         {"capability": "generic", "from_state": "draft", "to_state": "sale"},
         "transition marked allowed"),
        ("eval_gen_generic_threshold", None, "generic", "metric under threshold",
         {"capability": "generic", "metric": 0.02, "threshold": 0.05, "ok": True},
         {"capability": "generic", "metric": 0.09, "threshold": 0.05, "ok": False},
         {"capability": "generic", "metric": 0.02, "threshold": 0.05},
         "metric below threshold"),
        ("eval_odoo_generic_company_scope", "odoo", "generic", "company_id consistency",
         {"capability": "generic", "record_company": 1, "user_company": 1, "ok": True},
         {"capability": "generic", "record_company": 2, "user_company": 1, "ok": False},
         {"capability": "generic", "record_company": 1},
         "record company matches user company"),
        ("eval_gen_generic_nonempty_summary", None, "generic", "non-empty summary",
         {"capability": "generic", "summary": "Completed requested change"},
         {"capability": "generic", "summary": ""},
         {"capability": "generic"},
         "summary is non-empty"),
        ("eval_odoo_generic_currency", "odoo", "generic", "currency consistency",
         {"capability": "generic", "order_currency": "USD", "line_currency": "USD", "ok": True},
         {"capability": "generic", "order_currency": "USD", "line_currency": "EUR", "ok": False},
         {"capability": "generic", "order_currency": "USD"},
         "line currency matches order"),
        ("eval_gen_generic_list_length", None, "generic", "expected item count",
         {"capability": "generic", "count": 3, "expected": 3},
         {"capability": "generic", "count": 1, "expected": 3},
         {"capability": "generic", "count": 3},
         "count equals expected"),
    ]

    out: list[_EvalFamily] = []
    for slug, domain, category, subject, good, bad, unclear, expect_key in specs:
        if category in {"execution", "context", "conversation", "approval", "generic"} and isinstance(good, dict):
            good_c = dict(good)
            bad_c = dict(bad) if isinstance(bad, dict) else {"capability": category, "result": bad}
            unclear_c = dict(unclear) if isinstance(unclear, dict) else {"capability": category, "result": unclear}
        else:
            # planner/coding/repair use string results
            good_c = {"capability": category, "result": good}
            bad_c = {"capability": category, "result": bad}
            unclear_c = {"capability": category, "result": unclear}
        expectation = {
            "capability": category,
            "expected": expect_key,
            "subject": subject,
        }
        rubric = (
            f"Judge whether the {category} candidate satisfies the expectation "
            f"for: {subject}."
        )
        out.append(
            _EvalFamily(
                family_id=slug,
                domain=domain,
                category=category,
                subject=subject,
                good_candidate=good_c,
                bad_candidate=bad_c,
                unclear_candidate=unclear_c,
                expectation=expectation,
                rubric=rubric,
            )
        )
    return tuple(out)


def _extra(family: _EvalFamily, role: str, field_pattern: str) -> dict[str, Any]:
    return {
        "evaluation_controlled_version": EVALUATION_CONTROLLED_VERSION,
        "scenario_family": family.family_id,
        "evaluation_role": role,
        "candidate_category": family.category,
        "field_pattern": field_pattern,
        "fp2_native": True,
        "legacy": False,
        "controlled_evaluation": True,
        "legacy_projection": False,
    }


def _make(
    *,
    index: int,
    family: _EvalFamily,
    role: str,
    field_pattern: str,
    candidate: dict[str, Any],
    expectation: dict[str, Any] | None,
    rubric: str | None,
    verdict: str,
    score: float | None,
    explanation: str | None,
) -> dict[str, Any]:
    rec = EvaluationJudgmentRecord(
        record_type="evaluation_judgment",
        record_id=rid("eval-ej", index),
        provider_capability="evaluation",
        domain_specialization=family.domain,
        candidate=candidate,
        expectation=expectation,
        rubric=rubric,
        verdict=verdict,
        score=score,
        explanation=explanation,
        metadata=fixture_metadata(
            generator=EVALUATION_CONTROLLED_GENERATOR,
            index=index,
            provider_capability="evaluation",
            domain_specialization=family.domain,
            extra=_extra(family, role, field_pattern),
        ),
    )
    return rec.to_dict()


def generate_evaluation_controlled_records() -> list[dict[str, Any]]:
    """Generate controlled Evaluation corpus (~250), deterministic, family-linked."""
    records: list[dict[str, Any]] = []
    idx = 0
    families = _families()
    if len(families) < MIN_FAMILIES:
        raise RuntimeError(f"evaluation families {len(families)} < {MIN_FAMILIES}")

    for family in families:
        # 1) pass — candidate + expectation + rubric + score + explanation
        idx += 1
        records.append(
            _make(
                index=idx,
                family=family,
                role="pass_full",
                field_pattern="candidate_expectation_rubric",
                candidate=family.good_candidate,
                expectation=family.expectation,
                rubric=family.rubric,
                verdict="pass",
                score=1.0 if idx % 2 == 0 else 0.92,
                explanation=f"Candidate matches expectation for {family.subject}",
            )
        )
        # 2) fail — candidate + expectation (+ score/explanation); no rubric
        idx += 1
        records.append(
            _make(
                index=idx,
                family=family,
                role="fail_expectation",
                field_pattern="candidate_expectation",
                candidate=family.bad_candidate,
                expectation=family.expectation,
                rubric=None,
                verdict="fail",
                score=0.0 if idx % 3 != 0 else 0.15,
                explanation=f"Candidate does not satisfy expectation for {family.subject}",
            )
        )
        # 3) inconclusive — candidate + rubric; no expectation; usually no score
        idx += 1
        records.append(
            _make(
                index=idx,
                family=family,
                role="inconclusive_rubric",
                field_pattern="candidate_rubric",
                candidate=family.unclear_candidate,
                expectation=None,
                rubric=family.rubric,
                verdict="inconclusive",
                score=None if idx % 4 != 0 else 0.4,
                explanation=(
                    None
                    if idx % 5 == 0
                    else f"Insufficient evidence to judge {family.subject}"
                ),
            )
        )
        # 4) candidate only — alternating pass/fail; explanation optional
        idx += 1
        only_pass = idx % 2 == 0
        records.append(
            _make(
                index=idx,
                family=family,
                role="candidate_only",
                field_pattern="candidate_only",
                candidate=family.good_candidate if only_pass else family.bad_candidate,
                expectation=None,
                rubric=None,
                verdict="pass" if only_pass else "fail",
                score=None if idx % 3 == 0 else (0.85 if only_pass else 0.2),
                explanation=(
                    None
                    if idx % 4 == 0
                    else (
                        f"Candidate-only judgment for {family.subject}: "
                        + ("acceptable" if only_pass else "not acceptable")
                    )
                ),
            )
        )

    seen_fp: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for rec in records:
        validate_record_mapping(rec)
        assert rec["provider_capability"] == "evaluation"
        assert rec["record_type"] == "evaluation_judgment"
        assert rec["record_type"] in EVALUATION_ALLOWED_RECORD_TYPES
        assert rec["training_contract_version"] == SYSTEM_TRAINING_CONTRACT_VERSION
        assert rec["metadata"].get("scenario_family")
        assert rec["metadata"].get("legacy") is False
        assert rec["metadata"].get("legacy_projection") is False
        assert not scan_forbidden_how(rec)
        assert not scan_taxonomy(rec)
        fp = fingerprint_record(rec)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        deduped.append(rec)

    if not (TARGET_MIN <= len(deduped) <= TARGET_MAX):
        raise RuntimeError(
            f"controlled Evaluation count {len(deduped)} outside [{TARGET_MIN}, {TARGET_MAX}]"
        )
    return deduped


def evaluation_family_count() -> int:
    return len(_families())


def normalized_fingerprint(record: Mapping[str, Any]) -> str:
    blob = stable_dumps(
        {
            "record_type": record.get("record_type"),
            "provider": record.get("provider_capability"),
            "domain": record.get("domain_specialization") or "generic",
            "input": record.get("input"),
            "expected_output": record.get("expected_output"),
        }
    ).lower()
    blob = re.sub(r"[a-z0-9_./\\-]+\.(py|xml|csv|md|html|json)", "<PATH>", blob)
    blob = re.sub(r"\b\d+\b", "<N>", blob)
    blob = re.sub(r"\s+", " ", blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def find_normalized_duplicates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for r in records:
        buckets[normalized_fingerprint(r)].append(str(r.get("record_id") or "?"))
    dups = {fp: ids for fp, ids in buckets.items() if len(ids) > 1}
    return {
        "unique_normalized_fingerprints": len(buckets),
        "normalized_duplicate_groups": len(dups),
        "normalized_duplicate_record_ids": dups,
    }


def _optional_field_pattern(record: Mapping[str, Any]) -> str:
    inp = record.get("input") if isinstance(record.get("input"), Mapping) else {}
    has_exp = "expectation" in inp
    has_rub = "rubric" in inp
    if has_exp and has_rub:
        return "candidate_expectation_rubric"
    if has_exp:
        return "candidate_expectation"
    if has_rub:
        return "candidate_rubric"
    return "candidate_only"


def analyze_evaluation_controlled(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(r["record_type"]) for r in records)
    families = Counter(
        str((r.get("metadata") or {}).get("scenario_family") or "?") for r in records
    )
    categories = Counter(
        str((r.get("metadata") or {}).get("candidate_category") or "?") for r in records
    )
    domain = domain_distribution(list(records))
    exact = find_duplicates(list(records))
    normalized = find_normalized_duplicates(records)

    verdicts = Counter(
        str((r.get("expected_output") or {}).get("verdict") or "") for r in records
    )
    with_score = sum(1 for r in records if "score" in (r.get("expected_output") or {}))
    without_score = len(records) - with_score
    with_expl = sum(
        1 for r in records if (r.get("expected_output") or {}).get("explanation") is not None
    )
    without_expl = len(records) - with_expl
    field_patterns = Counter(_optional_field_pattern(r) for r in records)

    split_counts: Counter[str] = Counter()
    family_splits: dict[str, set[str]] = defaultdict(set)
    split_rows: list[dict[str, Any]] = []
    for r in records:
        split = assign_split(r).value
        fam = str((r.get("metadata") or {}).get("scenario_family") or "")
        split_counts[split] += 1
        family_splits[fam].add(split)
        split_rows.append(
            {
                "record_id": r.get("record_id"),
                "split": split,
                "scenario_family": fam,
                "scenario_key": scenario_key(r),
                "record_type": r.get("record_type"),
            }
        )
    leakage = {
        fam: sorted(splits)
        for fam, splits in sorted(family_splits.items())
        if len(splits) > 1
    }

    how_hits = sum(1 for r in records if scan_forbidden_how(r))
    tax_hits = sum(1 for r in records if scan_taxonomy(r))
    legacy_hits = sum(
        1
        for r in records
        if r.get("metadata", {}).get("legacy")
        or r.get("metadata", {}).get("legacy_projection")
        or "evaluation_dataset.jsonl" in json.dumps(r)
        or r.get("provider_capability") != "evaluation"
    )
    neg_contam = sum(
        1
        for r in records
        if str((r.get("metadata") or {}).get("quality_corpus") or "").startswith("negative")
    )
    neg_results = [evaluate_negative_case(c) for c in REASONING_SPARSE_NEGATIVE_CASES]
    neg_ok = all(r["matched"] for r in neg_results)

    pack = format_fp2_pack(list(records), pack="reasoning")
    pack_ok = len(pack) == len(records) and all(
        ex.dataset_type.value == "evaluation" for ex in pack
    )

    # Semantic audit (lightweight, deterministic)
    semantic_issues: list[str] = []
    for r in records:
        inp = r.get("input") or {}
        out = r.get("expected_output") or {}
        cand = inp.get("candidate")
        if not isinstance(cand, Mapping) or not cand:
            semantic_issues.append(f"{r.get('record_id')}:empty_candidate")
        verdict = str(out.get("verdict") or "")
        if verdict not in {"pass", "fail", "inconclusive"}:
            semantic_issues.append(f"{r.get('record_id')}:bad_verdict")
        if "score" in out:
            try:
                s = float(out["score"])
                if s < 0.0 or s > 1.0:
                    semantic_issues.append(f"{r.get('record_id')}:score_range")
            except (TypeError, ValueError):
                semantic_issues.append(f"{r.get('record_id')}:score_type")
        # Judgment records must not invent Engineering WHAT capability_id on root
        if "capability_id" in out or "capability_id" in inp:
            semantic_issues.append(f"{r.get('record_id')}:engineering_what_leak")

    hard_fail = False
    reasons: list[str] = []
    if not (TARGET_MIN <= len(records) <= TARGET_MAX):
        hard_fail = True
        reasons.append("count_out_of_band")
    if len(families) < MIN_FAMILIES:
        hard_fail = True
        reasons.append("insufficient_families")
    if exact["duplicate_groups"] != 0:
        hard_fail = True
        reasons.append("exact_duplicates")
    if how_hits or tax_hits or legacy_hits or neg_contam:
        hard_fail = True
        reasons.append("safety_contamination")
    if leakage:
        hard_fail = True
        reasons.append("family_leakage")
    if not pack_ok:
        hard_fail = True
        reasons.append("pack_invalid")
    if split_counts.get("validation", 0) == 0 or split_counts.get("test", 0) == 0:
        hard_fail = True
        reasons.append("missing_val_or_test")
    if domain["odoo"] < 1 or domain["generic"] < 1:
        hard_fail = True
        reasons.append("domain_imbalance")
    if normalized["normalized_duplicate_groups"] > 5:
        hard_fail = True
        reasons.append("normalized_duplicates_severe")
    if not neg_ok:
        hard_fail = True
        reasons.append("negatives_mismatched")
    if any(verdicts.get(v, 0) == 0 for v in ("pass", "fail", "inconclusive")):
        hard_fail = True
        reasons.append("missing_verdict_class")
    if any(
        field_patterns.get(p, 0) == 0
        for p in (
            "candidate_only",
            "candidate_expectation",
            "candidate_rubric",
            "candidate_expectation_rubric",
        )
    ):
        hard_fail = True
        reasons.append("missing_optional_field_pattern")
    if with_score == 0 or without_score == 0:
        hard_fail = True
        reasons.append("score_presence_imbalance")
    if with_expl == 0 or without_expl == 0:
        hard_fail = True
        reasons.append("explanation_presence_imbalance")
    if semantic_issues:
        hard_fail = True
        reasons.append("semantic_audit_failed")

    verdict = "EVALUATION_CORPUS_BLOCKED" if hard_fail else "EVALUATION_CORPUS_READY"

    scorecard = {
        "native_records": len(records),
        "odoo": domain["odoo"],
        "generic": domain["generic"],
        "scenario_families": len(families),
        "largest_family_concentration": max(families.values()) if families else 0,
        "duplicate_groups": exact["duplicate_groups"],
        "normalized_duplicate_groups": normalized["normalized_duplicate_groups"],
        "forbidden_how": how_hits,
        "taxonomy_violations": tax_hits,
        "negative_contamination": neg_contam,
        "legacy_contamination": legacy_hits,
        "train": split_counts.get("train", 0),
        "validation": split_counts.get("validation", 0),
        "test": split_counts.get("test", 0),
        "family_leakage": len(leakage),
        "pack_validity": pack_ok,
        "provider_dataset_equivalence": pack_ok,
        "by_record_type": dict(by_type),
        "verdicts": dict(verdicts),
        "with_score": with_score,
        "without_score": without_score,
        "with_explanation": with_expl,
        "without_explanation": without_expl,
        "field_patterns": dict(field_patterns),
        "candidate_categories": dict(categories),
        "negatives_ok": neg_ok,
        "semantic_issue_count": len(semantic_issues),
    }

    return {
        "version": EVALUATION_CONTROLLED_VERSION,
        "provider_capability": "evaluation",
        "verdict": verdict,
        "fail_reasons": reasons,
        "scorecard": scorecard,
        "by_record_type": dict(by_type),
        "by_family": dict(families),
        "domain": domain,
        "duplicates": exact,
        "normalized_duplicates": normalized,
        "splits": dict(split_counts),
        "split_rows": split_rows,
        "leakage": leakage,
        "split_strategy": document_split_strategy(),
        "pack_count": len(pack),
        "pack_examples": pack,
        "semantic_issues": semantic_issues[:20],
        "legacy_projection": "NOT PERFORMED",
    }


def _batch2_checksum() -> str:
    return json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]


def _tree_checksum(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix().encode()
            h.update(rel)
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def emit_evaluation_controlled_corpus(
    *,
    training_root: Path | None = None,
    datasets_root: Path | None = None,
) -> dict[str, Any]:
    """Write Evaluation controlled corpus beside preserved AT-7.4 semantics artifacts."""
    before = _batch2_checksum()
    assert before == BATCH2_CHECKSUM

    conv_before = json.loads(_CONV_MANIFEST.read_text(encoding="utf-8"))
    appr_before = json.loads(_APPR_MANIFEST.read_text(encoding="utf-8"))
    assert conv_before["total_records"] == 232
    assert appr_before["total_records"] == 162

    records = generate_evaluation_controlled_records()
    analysis = analyze_evaluation_controlled(records)
    pack = analysis.pop("pack_examples")
    split_rows = analysis.pop("split_rows")

    train_out = Path(training_root or _DEFAULT_TRAINING)
    data_out = Path(datasets_root or _DEFAULT_DATASETS)
    written: dict[str, str] = {}

    for root in (train_out, data_out):
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve()
        assert "controlled_batch_2" not in str(resolved)
        assert "conversation" not in resolved.name or resolved.name == "evaluation"
        # Preserve AT-7.4 semantics report if present
        semantics_path = root / "semantics_report.json"
        if not semantics_path.exists():
            semantics_path.write_text(
                json.dumps(EVALUATION_SEMANTIC_DEFINITION, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            written[str(semantics_path)] = "semantics_report_seeded"

        write_jsonl(root / "evaluation_native.jsonl", records)
        written[str(root / "evaluation_native.jsonl")] = f"{len(records)} records"
        write_jsonl(root / "evaluation_judgment.jsonl", records)
        written[str(root / "evaluation_judgment.jsonl")] = f"{len(records)} records"

        pack_rows = [
            {
                "example_id": ex.example_id,
                "dataset_type": ex.dataset_type.value,
                "messages": [dict(m) for m in ex.messages],
                "metadata": dict(ex.metadata),
            }
            for ex in pack
        ]
        write_jsonl(root / "pack_evaluation.jsonl", pack_rows)
        written[str(root / "pack_evaluation.jsonl")] = f"{len(pack_rows)} examples"
        write_jsonl(root / "splits.jsonl", split_rows)
        written[str(root / "splits.jsonl")] = f"{len(split_rows)} rows"

        scorecard = analysis["scorecard"]
        manifest = {
            "version": EVALUATION_CONTROLLED_VERSION,
            "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
            "provider_capability": "evaluation",
            "product_plane": "reasoning",
            "generator": EVALUATION_CONTROLLED_GENERATOR,
            "record_type": "evaluation_judgment",
            "total_records": len(records),
            "by_record_type": {"evaluation_judgment": len(records)},
            "scorecard": scorecard,
            "verdict": analysis["verdict"],
            "legacy_projection": False,
            "legacy_projection_status": "NOT PERFORMED",
            "controlled_batch_2_modified": False,
            "conversation_corpus_modified": False,
            "approval_corpus_modified": False,
            "split_version": "fp2-split-1.0.0",
            "at74_mapping": "EVALUATION_MAPPING_READY",
            "notes": [
                "AT-7.5 controlled Evaluation corpus",
                "evaluation_judgment only; no Continuity/observation families",
                "Preserves AT-7.4 semantics_report.json",
                "Not for certification / not a production adapter pack",
            ],
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written[str(root / "manifest.json")] = "manifest"

        report = {k: v for k, v in analysis.items()}
        (root / "quality_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        written[str(root / "quality_report.json")] = "quality_report"

        gen_meta = {
            "version": EVALUATION_CONTROLLED_VERSION,
            "provider_capability": "evaluation",
            "family_count": evaluation_family_count(),
            "actual_count": len(records),
            "scenario_family_mechanism": (
                "metadata.scenario_family groups related variants; "
                "assign_split uses family: key (fp2-split-1.0.0)"
            ),
            "record_types": ["evaluation_judgment"],
            "legacy_projection": "NOT PERFORMED",
        }
        (root / "generation_metadata.json").write_text(
            json.dumps(gen_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written[str(root / "generation_metadata.json")] = "generation_metadata"

    after = _batch2_checksum()
    assert after == before == BATCH2_CHECKSUM

    conv_after = json.loads(_CONV_MANIFEST.read_text(encoding="utf-8"))
    appr_after = json.loads(_APPR_MANIFEST.read_text(encoding="utf-8"))
    assert conv_after == conv_before
    assert appr_after == appr_before

    return {
        "version": EVALUATION_CONTROLLED_VERSION,
        "verdict": analysis["verdict"],
        "count": len(records),
        "scorecard": analysis["scorecard"],
        "fail_reasons": analysis["fail_reasons"],
        "splits": analysis["splits"],
        "legacy_projection": "NOT PERFORMED",
        "batch2_checksum_before": before,
        "batch2_checksum_after": after,
        "batch2_immutable": True,
        "conversation_preserved": True,
        "approval_preserved": True,
        "corpus_checksum_training": _tree_checksum(train_out),
        "corpus_checksum_datasets": _tree_checksum(data_out),
        "written": written,
    }


if __name__ == "__main__":
    result = emit_evaluation_controlled_corpus()
    print(json.dumps({k: v for k, v in result.items() if k != "written"}, indent=2, default=str))
