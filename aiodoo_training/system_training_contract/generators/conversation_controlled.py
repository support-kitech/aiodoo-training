"""AT-7.3 — Controlled FP2 Conversation corpus (fp2-reasoning-sparse-1.0.0).

Native Conversation records only: decision_context + loop_decision(clarify).
Does NOT modify controlled_batch_2, Planner, or mapping.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aiodoo_training.system_training_contract.generators.common import (
    fixture_metadata,
    rid,
)
from aiodoo_training.system_training_contract.quality.common import fingerprint_record
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.records import (
    DecisionContextRecord,
    LoopDecisionRecord,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

REASONING_SPARSE_VERSION: str = "fp2-reasoning-sparse-1.0.0"
CONVERSATION_CONTROLLED_GENERATOR: str = "conversation_controlled"
TARGET_MIN = 150
TARGET_MAX = 250
TARGET_PREFERRED = 220
MIN_FAMILIES = 50


@dataclass(frozen=True)
class _ConvFamily:
    family_id: str
    domain: str | None
    objective: str
    ambiguity: str
    question: str
    followup_question: str
    history_note: str


def _families() -> tuple[_ConvFamily, ...]:
    """≥50 scenario families; each yields ~4 related conversation records."""
    # (slug, domain, objective, ambiguity, question, followup, history_note)
    specs: list[tuple[str, str | None, str, str, str, str, str]] = [
        # --- Odoo ---
        (
            "conv_odoo_partner_vat_scope",
            "odoo",
            "Clarify whether partner VAT is required on create",
            "ambiguous_vat_requirement",
            "Ask whether VAT is mandatory for all partners or company-type only",
            "Ask which countries require VAT validation on create",
            "User said add VAT but did not specify when it is required",
        ),
        (
            "conv_odoo_sale_confirm_policy",
            "odoo",
            "Clarify sale.order confirmation policy under partial stock",
            "ambiguous_confirm_policy",
            "Ask whether confirm should block when stock is partial",
            "Ask if backorders are acceptable after confirm",
            "Confirm requested without stock-policy detail",
        ),
        (
            "conv_odoo_invoice_tax_mapping",
            "odoo",
            "Clarify tax mapping for customer invoices",
            "ambiguous_tax_mapping",
            "Ask which fiscal position applies to this customer",
            "Ask whether tax is price-included or excluded",
            "Invoice tax outcome requested without fiscal context",
        ),
        (
            "conv_odoo_stock_route",
            "odoo",
            "Clarify preferred stock route for replenishment",
            "ambiguous_route_choice",
            "Ask whether to use buy or manufacture route",
            "Ask which warehouse should own the replenishment",
            "Replenishment requested without route preference",
        ),
        (
            "conv_odoo_hr_leave_type",
            "odoo",
            "Clarify leave type for employee time-off request",
            "ambiguous_leave_type",
            "Ask which leave type applies to this request",
            "Ask whether unpaid leave is allowed for the period",
            "Time-off requested without leave-type selection",
        ),
        (
            "conv_odoo_project_task_owner",
            "odoo",
            "Clarify project.task assignee when multiple users match",
            "ambiguous_assignee",
            "Ask which user should own the task",
            "Ask whether assignee may be a project team rather than one person",
            "Task create blocked on unclear assignee",
        ),
        (
            "conv_odoo_account_journal",
            "odoo",
            "Clarify target journal for miscellaneous entry",
            "ambiguous_journal",
            "Ask which journal should receive the misc entry",
            "Ask whether entry is bank, cash, or general",
            "Misc move requested without journal",
        ),
        (
            "conv_odoo_purchase_rfq_vendor",
            "odoo",
            "Clarify vendor selection for RFQ",
            "ambiguous_vendor",
            "Ask which vendor should receive the RFQ",
            "Ask whether multi-vendor RFQ comparison is required",
            "RFQ drafted without vendor preference",
        ),
        (
            "conv_odoo_crm_stage",
            "odoo",
            "Clarify CRM opportunity stage transition",
            "ambiguous_stage",
            "Ask which stage the opportunity should move to",
            "Ask whether stage change requires won/lost reason",
            "Opportunity update lacks stage intent",
        ),
        (
            "conv_odoo_website_publish",
            "odoo",
            "Clarify website page publish audience",
            "ambiguous_publish_scope",
            "Ask whether page is public or portal-only",
            "Ask which website / language variant to publish",
            "Publish requested without audience scope",
        ),
        (
            "conv_odoo_pos_payment",
            "odoo",
            "Clarify POS payment method for refund",
            "ambiguous_payment_method",
            "Ask which payment method should receive the refund",
            "Ask whether refund must match original tender",
            "Refund blocked pending payment-method clarity",
        ),
        (
            "conv_odoo_mail_template",
            "odoo",
            "Clarify which mail template to send after confirm",
            "ambiguous_template",
            "Ask which mail template applies after confirmation",
            "Ask whether template language follows partner or company",
            "Post-confirm email requested without template id",
        ),
        (
            "conv_odoo_access_group",
            "odoo",
            "Clarify security group for new portal users",
            "ambiguous_access_group",
            "Ask which security group new portal users should join",
            "Ask whether sales or accounting portal rights are needed",
            "Portal user create blocked on group ambiguity",
        ),
        (
            "conv_odoo_uom_conversion",
            "odoo",
            "Clarify UoM conversion for product packing",
            "ambiguous_uom",
            "Ask which unit of measure is authoritative for packing",
            "Ask for conversion factor between sale and inventory UoM",
            "Packing change requested without UoM clarity",
        ),
        (
            "conv_odoo_pricelist",
            "odoo",
            "Clarify pricelist for quotation",
            "ambiguous_pricelist",
            "Ask which pricelist applies to this quotation",
            "Ask whether currency follows pricelist or company",
            "Quotation pricing blocked on missing pricelist",
        ),
        (
            "conv_odoo_delivery_carrier",
            "odoo",
            "Clarify delivery carrier for outbound shipment",
            "ambiguous_carrier",
            "Ask which delivery carrier to assign",
            "Ask whether carrier rates must be refreshed before ship",
            "Shipment ready but carrier unspecified",
        ),
        (
            "conv_odoo_analytic_account",
            "odoo",
            "Clarify analytic account for cost allocation",
            "ambiguous_analytic",
            "Ask which analytic account should receive the cost",
            "Ask whether distribution across multiple analytics is required",
            "Cost posting blocked without analytic target",
        ),
        (
            "conv_odoo_bom_variant",
            "odoo",
            "Clarify BOM variant for manufacturing order",
            "ambiguous_bom",
            "Ask which BOM variant applies to this MO",
            "Ask whether optional components should be included",
            "MO create blocked on BOM ambiguity",
        ),
        (
            "conv_odoo_bank_reconcile",
            "odoo",
            "Clarify bank statement line matching rule",
            "ambiguous_reconcile_rule",
            "Ask whether to match by amount, reference, or partner",
            "Ask if partial reconciliation is allowed",
            "Reconcile requested without matching preference",
        ),
        (
            "conv_odoo_helpdesk_sla",
            "odoo",
            "Clarify helpdesk SLA team for ticket",
            "ambiguous_sla_team",
            "Ask which helpdesk team owns the SLA",
            "Ask whether VIP SLA overrides default team SLA",
            "Ticket triage blocked on SLA team ambiguity",
        ),
        (
            "conv_odoo_product_attribute",
            "odoo",
            "Clarify product attribute values for variant create",
            "ambiguous_attributes",
            "Ask which attribute values define the new variant",
            "Ask whether all combinations should be generated",
            "Variant create missing attribute selections",
        ),
        (
            "conv_odoo_payment_term",
            "odoo",
            "Clarify payment terms on customer invoice",
            "ambiguous_payment_term",
            "Ask which payment term applies to this invoice",
            "Ask whether early-payment discount is intended",
            "Invoice draft lacks payment-term choice",
        ),
        (
            "conv_odoo_warehouse_dest",
            "odoo",
            "Clarify destination warehouse for internal transfer",
            "ambiguous_warehouse",
            "Ask which warehouse is the transfer destination",
            "Ask whether transit location is required",
            "Internal transfer blocked without destination",
        ),
        (
            "conv_odoo_sequence_reset",
            "odoo",
            "Clarify fiscal-year sequence reset policy",
            "ambiguous_sequence_policy",
            "Ask whether sequences reset each fiscal year",
            "Ask which document types participate in reset",
            "Sequence change requested without policy detail",
        ),
        (
            "conv_odoo_loyalty_program",
            "odoo",
            "Clarify loyalty program for POS basket",
            "ambiguous_loyalty",
            "Ask which loyalty program applies to this basket",
            "Ask whether points may be redeemed and earned together",
            "Loyalty apply blocked on program ambiguity",
        ),
        (
            "conv_odoo_sign_template",
            "odoo",
            "Clarify Sign template for contract send",
            "ambiguous_sign_template",
            "Ask which Sign template to use for the contract",
            "Ask which partner contact must sign first",
            "Contract send blocked without Sign template",
        ),
        (
            "conv_odoo_fleet_vehicle",
            "odoo",
            "Clarify fleet vehicle for service log",
            "ambiguous_vehicle",
            "Ask which fleet vehicle the service log belongs to",
            "Ask whether odometer reading is required before save",
            "Service log draft missing vehicle identity",
        ),
        (
            "conv_odoo_expense_category",
            "odoo",
            "Clarify expense category for employee claim",
            "ambiguous_expense_category",
            "Ask which expense category applies",
            "Ask whether receipt attachment is mandatory",
            "Expense claim blocked on category ambiguity",
        ),
        # --- Generic ---
        (
            "conv_gen_api_auth_scheme",
            None,
            "Clarify API authentication scheme for new endpoint",
            "ambiguous_auth_scheme",
            "Ask whether the endpoint uses API key, OAuth, or session auth",
            "Ask which scopes are required for callers",
            "Endpoint design blocked on auth ambiguity",
        ),
        (
            "conv_gen_retry_budget",
            None,
            "Clarify retry budget after transient failure",
            "ambiguous_retry_budget",
            "Ask how many retries are allowed before escalate",
            "Ask whether backoff is fixed or exponential",
            "Retry path blocked without budget",
        ),
        (
            "conv_gen_config_source",
            None,
            "Clarify configuration source of truth",
            "ambiguous_config_source",
            "Ask whether config comes from env, file, or remote store",
            "Ask which environment profile is active",
            "Config change requested without source preference",
        ),
        (
            "conv_gen_error_surface",
            None,
            "Clarify user-facing error message policy",
            "ambiguous_error_surface",
            "Ask whether errors expose codes, messages, or both",
            "Ask if internal traces may appear in client responses",
            "Error handling blocked on surface policy",
        ),
        (
            "conv_gen_pagination",
            None,
            "Clarify pagination contract for list API",
            "ambiguous_pagination",
            "Ask whether to use cursor or offset pagination",
            "Ask for default and maximum page size",
            "List API design missing pagination choice",
        ),
        (
            "conv_gen_idempotency",
            None,
            "Clarify idempotency key requirements",
            "ambiguous_idempotency",
            "Ask whether clients must send idempotency keys",
            "Ask how long idempotency records are retained",
            "Write API blocked on idempotency ambiguity",
        ),
        (
            "conv_gen_feature_flag",
            None,
            "Clarify feature-flag rollout audience",
            "ambiguous_flag_audience",
            "Ask which audience the feature flag targets",
            "Ask whether rollout is percentage or allow-list",
            "Flag enablement blocked without audience",
        ),
        (
            "conv_gen_schema_version",
            None,
            "Clarify schema version for payload migration",
            "ambiguous_schema_version",
            "Ask which schema version clients must speak",
            "Ask whether dual-read of old and new versions is required",
            "Migration blocked on schema-version ambiguity",
        ),
        (
            "conv_gen_cache_ttl",
            None,
            "Clarify cache TTL for read-through responses",
            "ambiguous_cache_ttl",
            "Ask what TTL applies to cached responses",
            "Ask whether stale-while-revalidate is acceptable",
            "Caching change requested without TTL",
        ),
        (
            "conv_gen_rate_limit",
            None,
            "Clarify rate-limit policy for public clients",
            "ambiguous_rate_limit",
            "Ask for requests-per-minute limit for public clients",
            "Ask whether burst tokens are allowed",
            "Rate-limit config blocked without policy numbers",
        ),
        (
            "conv_gen_logging_redaction",
            None,
            "Clarify which fields must be redacted in logs",
            "ambiguous_redaction",
            "Ask which PII fields must never appear in logs",
            "Ask whether hashed identifiers are allowed",
            "Logging change blocked on redaction rules",
        ),
        (
            "conv_gen_queue_priority",
            None,
            "Clarify job queue priority for backlog drain",
            "ambiguous_queue_priority",
            "Ask which job class has highest drain priority",
            "Ask whether starved low-priority jobs get promotion",
            "Backlog drain blocked without priority policy",
        ),
        (
            "conv_gen_timeout_budget",
            None,
            "Clarify end-to-end timeout budget",
            "ambiguous_timeout",
            "Ask for overall request timeout budget in seconds",
            "Ask how budget is split across upstream calls",
            "Timeouts unspecified for multi-hop call",
        ),
        (
            "conv_gen_storage_backend",
            None,
            "Clarify object storage backend for uploads",
            "ambiguous_storage",
            "Ask which object store receives uploads",
            "Ask whether objects are private or signed-URL public",
            "Upload path blocked without storage choice",
        ),
        (
            "conv_gen_locale_format",
            None,
            "Clarify locale for date and number formatting",
            "ambiguous_locale",
            "Ask which locale drives date and number formats",
            "Ask whether locale follows user or tenant default",
            "Formatting blocked on locale ambiguity",
        ),
        (
            "conv_gen_webhook_retry",
            None,
            "Clarify webhook delivery retry policy",
            "ambiguous_webhook_retry",
            "Ask how many webhook delivery attempts are allowed",
            "Ask whether failed deliveries must alert operators",
            "Webhook setup missing retry policy",
        ),
        (
            "conv_gen_search_ranking",
            None,
            "Clarify search ranking preference",
            "ambiguous_ranking",
            "Ask whether ranking prioritizes recency or relevance",
            "Ask if exact-match boosts are required",
            "Search change blocked on ranking preference",
        ),
        (
            "conv_gen_notification_channel",
            None,
            "Clarify notification channel for alerts",
            "ambiguous_channel",
            "Ask whether alerts go to email, chat, or both",
            "Ask which severity levels trigger each channel",
            "Alerting blocked without channel choice",
        ),
        (
            "conv_gen_data_retention",
            None,
            "Clarify data retention window",
            "ambiguous_retention",
            "Ask how long raw events must be retained",
            "Ask whether deletion is hard delete or soft tombstone",
            "Retention policy change missing window",
        ),
        (
            "conv_gen_batch_size",
            None,
            "Clarify batch size for bulk import",
            "ambiguous_batch_size",
            "Ask for preferred batch size for bulk import",
            "Ask whether failed rows abort the whole batch",
            "Bulk import blocked without batch policy",
        ),
        (
            "conv_gen_conflict_resolution",
            None,
            "Clarify conflict resolution for concurrent edits",
            "ambiguous_conflict",
            "Ask whether last-write-wins or merge is preferred",
            "Ask if conflicts must surface to the user",
            "Concurrent edit path blocked on conflict policy",
        ),
        (
            "conv_gen_secret_rotation",
            None,
            "Clarify secret rotation window",
            "ambiguous_rotation",
            "Ask how often secrets must rotate",
            "Ask whether dual-key overlap is required during rotation",
            "Secret rotation requested without window",
        ),
        (
            "conv_gen_healthcheck",
            None,
            "Clarify health-check readiness criteria",
            "ambiguous_readiness",
            "Ask which dependencies must pass for readiness",
            "Ask whether degraded mode still reports ready",
            "Health endpoint blocked on readiness definition",
        ),
        (
            "conv_gen_export_format",
            None,
            "Clarify export format for report download",
            "ambiguous_export_format",
            "Ask whether export should be CSV, JSON, or PDF",
            "Ask which columns are mandatory in the export",
            "Report download blocked without format",
        ),
        (
            "conv_gen_timezone",
            None,
            "Clarify timezone for scheduled jobs",
            "ambiguous_timezone",
            "Ask which timezone schedules should use",
            "Ask how DST transitions should be handled",
            "Scheduler change blocked on timezone",
        ),
        (
            "conv_gen_sla_breach",
            None,
            "Clarify SLA breach escalation contact",
            "ambiguous_sla_contact",
            "Ask who must be notified on SLA breach",
            "Ask whether customer-facing status updates are required",
            "SLA handling blocked without escalation contact",
        ),
        (
            "conv_gen_rollback_window",
            None,
            "Clarify rollback window after deploy",
            "ambiguous_rollback",
            "Ask how long post-deploy rollback remains available",
            "Ask whether data migrations are reversible in that window",
            "Deploy follow-up blocked on rollback policy",
        ),
        (
            "conv_odoo_multi_company",
            "odoo",
            "Clarify company context for multi-company record",
            "ambiguous_company",
            "Ask which company the record belongs to",
            "Ask whether inter-company rules should apply",
            "Create blocked without company context",
        ),
        (
            "conv_odoo_currency_rate",
            "odoo",
            "Clarify currency rate source for conversion",
            "ambiguous_rate_source",
            "Ask which rate provider or date to use",
            "Ask whether manual rate override is allowed",
            "Currency conversion blocked on rate source",
        ),
        (
            "conv_gen_consent_scope",
            None,
            "Clarify user consent scope before processing",
            "ambiguous_consent",
            "Ask which processing purposes the user consented to",
            "Ask whether consent may be withdrawn mid-flow",
            "Processing blocked without consent clarity",
        ),
    ]
    out: list[_ConvFamily] = []
    for slug, domain, objective, ambiguity, question, followup, history in specs:
        out.append(
            _ConvFamily(
                family_id=slug,
                domain=domain,
                objective=objective,
                ambiguity=ambiguity,
                question=question,
                followup_question=followup,
                history_note=history,
            )
        )
    return tuple(out)


def _extra(family: _ConvFamily, role: str) -> dict[str, Any]:
    return {
        "reasoning_sparse_version": REASONING_SPARSE_VERSION,
        "scenario_family": family.family_id,
        "conversation_role": role,
        "fp2_native": True,
        "legacy": False,
        "controlled_conversation": True,
    }


def _make_dc(
    *,
    index: int,
    family: _ConvFamily,
    role: str,
    cycle_index: int,
    bounded_history: tuple[Mapping[str, Any], ...] = (),
    observation_quality: str = "",
) -> dict[str, Any]:
    rec = DecisionContextRecord(
        record_type="decision_context",
        record_id=rid("conv-dc", index),
        objective=f"[{family.family_id}] {family.objective}",
        objective_state="blocked",
        cycle_index=cycle_index,
        observation_quality=observation_quality,
        blockers=(family.ambiguity,),
        possible_next_actions=("clarify", "escalate"),
        continuation_hint="clarify",
        bounded_history=bounded_history,
        provider_capability="conversation",
        domain_specialization=family.domain,
        metadata=fixture_metadata(
            generator=CONVERSATION_CONTROLLED_GENERATOR,
            index=index,
            provider_capability="conversation",
            domain_specialization=family.domain,
            extra=_extra(family, role),
        ),
    )
    return rec.to_dict()


def _make_ld(
    *,
    index: int,
    family: _ConvFamily,
    role: str,
    reason: str,
    next_goal: str,
) -> dict[str, Any]:
    rec = LoopDecisionRecord(
        record_type="loop_decision",
        record_id=rid("conv-ld", index),
        decision_kind="clarify",
        reason=reason,
        next_goal=next_goal,
        provider_capability="conversation",
        domain_specialization=family.domain,
        metadata=fixture_metadata(
            generator=CONVERSATION_CONTROLLED_GENERATOR,
            index=1000 + index,
            provider_capability="conversation",
            domain_specialization=family.domain,
            extra=_extra(family, role),
        ),
    )
    return rec.to_dict()


def generate_conversation_controlled_records() -> list[dict[str, Any]]:
    """Generate controlled Conversation corpus (~200), deterministic, family-linked."""
    records: list[dict[str, Any]] = []
    dc_i = 0
    ld_i = 0
    families = _families()
    if len(families) < MIN_FAMILIES:
        raise RuntimeError(f"conversation families {len(families)} < {MIN_FAMILIES}")

    for family in families:
        dc_i += 1
        records.append(
            _make_dc(
                index=dc_i,
                family=family,
                role="decision_context_primary",
                cycle_index=1,
            )
        )
        dc_i += 1
        records.append(
            _make_dc(
                index=dc_i,
                family=family,
                role="decision_context_followup",
                cycle_index=2,
                observation_quality="partial",
                bounded_history=(
                    {
                        "cycle_index": 1,
                        "objective_state": "blocked",
                        "note": family.history_note,
                        "scenario_family": family.family_id,
                    },
                ),
            )
        )
        ld_i += 1
        records.append(
            _make_ld(
                index=ld_i,
                family=family,
                role="loop_clarify_primary",
                reason=(
                    f"[{family.family_id}] Requirement ambiguous ({family.ambiguity}); "
                    "need user clarification before continuing"
                ),
                next_goal=family.question,
            )
        )
        ld_i += 1
        records.append(
            _make_ld(
                index=ld_i,
                family=family,
                role="loop_clarify_followup",
                reason=(
                    f"[{family.family_id}] Prior answer incomplete for {family.ambiguity}; "
                    "ask a focused follow-up"
                ),
                next_goal=family.followup_question,
            )
        )

    seen_fp: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for rec in records:
        validate_record_mapping(rec)
        assert rec["provider_capability"] == "conversation"
        assert rec["record_type"] in {"decision_context", "loop_decision"}
        assert rec["training_contract_version"] == SYSTEM_TRAINING_CONTRACT_VERSION
        assert rec["metadata"].get("scenario_family")
        assert rec["metadata"].get("legacy") is False
        if rec["record_type"] == "loop_decision":
            assert rec["expected_output"]["decision_kind"] == "clarify"
            assert str(rec["expected_output"].get("reason") or "").strip()
        assert not scan_forbidden_how(rec)
        assert not scan_taxonomy(rec)
        fp = fingerprint_record(rec)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        deduped.append(rec)

    if not (TARGET_MIN <= len(deduped) <= TARGET_MAX):
        raise RuntimeError(
            f"controlled Conversation count {len(deduped)} outside [{TARGET_MIN}, {TARGET_MAX}]"
        )
    return deduped


def conversation_family_count() -> int:
    return len(_families())
