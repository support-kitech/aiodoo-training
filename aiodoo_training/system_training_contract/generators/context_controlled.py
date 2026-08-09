"""AT-6.2.1 — Controlled FP2 Context corpus (fp2-context-controlled-1.0.0).

Scales Context-native generation beyond AT-6.2 fixtures without overwriting them.
Does NOT modify controlled_batch_2. Does NOT project legacy context_v1_0.
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
from aiodoo_training.system_training_contract.generators.mapping import (
    CONTEXT_ALLOWED_RECORD_TYPES,
)
from aiodoo_training.system_training_contract.quality.analysis import (
    domain_distribution,
    find_duplicates,
)
from aiodoo_training.system_training_contract.quality.common import (
    extract_engineering_capability,
    fingerprint_record,
    stable_dumps,
)
from aiodoo_training.system_training_contract.quality.formatters import format_fp2_pack
from aiodoo_training.system_training_contract.quality.gates import (
    scan_forbidden_how,
    scan_taxonomy,
)
from aiodoo_training.system_training_contract.quality.negatives import (
    NEGATIVE_CASES,
    evaluate_negative_case,
)
from aiodoo_training.system_training_contract.quality.splits import (
    assign_split,
    document_split_strategy,
    scenario_key,
)
from aiodoo_training.system_training_contract.records import (
    CapabilityIntentRecord,
    ObservationRecord,
    validate_record_mapping,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

CONTEXT_CONTROLLED_VERSION: str = "fp2-context-controlled-1.0.0"
CONTEXT_CONTROLLED_GENERATOR: str = "context_controlled"
CONTEXT_LOCATE_CAPS: frozenset[str] = frozenset(
    {
        "workspace.search",
        "workspace.navigate",
        "workspace.read",
        "repository.inspect",
    }
)
TARGET_MIN = 200
TARGET_MAX = 300
TARGET_PREFERRED = 250

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent
_DEFAULT_TRAINING = _REPO_ROOT / "fixtures" / "fp2" / "context_controlled_1"
_DEFAULT_DATASETS = (
    _WORKSPACE_ROOT / "aiodoo-datasets" / "datasets" / "fp2" / "context_controlled_1"
)
_AT62_FIXTURES = _REPO_ROOT / "fixtures" / "fp2" / "context"
BATCH2_CHECKSUM = "728d9bad313626b470ff155e1211f779b6330758eab57301672a617692e3f227"
BATCH2 = _WORKSPACE_ROOT / "aiodoo-datasets" / "datasets" / "fp2" / "controlled_batch_2"


@dataclass(frozen=True)
class _ObsVariant:
    role: str  # success | partial | failed | navigate_obs | read_obs | inspect_obs
    capability_id: str
    kind: str
    status: str
    summary: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class _Family:
    family_id: str
    task: str
    domain: str | None
    intent_cap: str
    intent_objective: str
    intent_args: dict[str, Any]
    observations: tuple[_ObsVariant, ...]
    followup_intents: tuple[tuple[str, str, dict[str, Any]], ...] = ()


def _art(
    path: str,
    symbol: str,
    kind: str,
    score: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "symbol": symbol,
        "artifact_kind": kind,
        "score": score,
        "ranking_reason": reason,
    }


def _families() -> tuple[_Family, ...]:
    """Deterministic scenario families — semantically distinct locate narratives."""
    # fmt: off
    specs: list[_Family] = [
        # --- model lookup ---
        _Family(
            "ctx_model_res_partner", "model_lookup", "odoo",
            "workspace.search", "Locate definition of res.partner model",
            {"query": "res.partner", "intent": "locate_model_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located res.partner model definition",
                            {"match_count": 1, "ranked_artifacts": [_art("models/partner.py", "res.partner", "model", 100, "direct_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Ambiguous partner model matches across modules",
                            {"match_count": 5, "ambiguous": True}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "No model definition for res.partner under scoped path",
                            {"match_count": 0, "query": "res.partner", "scope": "addons/unknown"}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to located partner model path", {"path": "models/partner.py"}),
                ("workspace.read", "Read located res.partner model source", {"path": "models/partner.py"}),
            ),
        ),
        _Family(
            "ctx_model_auth_totp", "model_lookup", "odoo",
            "workspace.search", "Find AuthTotpDevice class definition",
            {"query": "AuthTotpDevice", "intent": "locate_model_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located AuthTotpDevice model",
                            {"match_count": 1, "ranked_artifacts": [_art("models/auth_totp_device.py", "AuthTotpDevice", "model", 100, "direct_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "AuthTotpDevice not found in workspace",
                            {"match_count": 0, "query": "AuthTotpDevice"}),
            ),
            followup_intents=(
                ("workspace.read", "Open AuthTotpDevice source after locate", {"path": "models/auth_totp_device.py"}),
            ),
        ),
        _Family(
            "ctx_model_sale_order", "model_lookup", "odoo",
            "workspace.search", "Locate sale.order model class",
            {"query": "class SaleOrder", "intent": "locate_model_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located SaleOrder model class",
                            {"match_count": 1, "ranked_artifacts": [_art("models/sale_order.py", "SaleOrder", "model", 99, "class_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Multiple SaleOrder-like symbols found",
                            {"match_count": 4, "ambiguous": True}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to sale_order.py after locate", {"path": "models/sale_order.py"}),
            ),
        ),
        _Family(
            "ctx_model_account_move", "model_lookup", "odoo",
            "workspace.search", "Locate account.move model definition",
            {"query": "account.move", "intent": "locate_model_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located account.move model",
                            {"match_count": 1, "ranked_artifacts": [_art("models/account_move.py", "account.move", "model", 100, "direct_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "account.move absent from current workspace",
                            {"match_count": 0, "query": "account.move"}),
            ),
        ),
        _Family(
            "ctx_model_generic_widget", "model_lookup", None,
            "workspace.search", "Locate WidgetRegistry class definition",
            {"query": "class WidgetRegistry", "intent": "locate_class_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located WidgetRegistry class",
                            {"match_count": 1, "ranked_artifacts": [_art("src/widgets/registry.py", "WidgetRegistry", "class", 98, "class_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Several Widget* registry symbols matched",
                            {"match_count": 6, "ambiguous": True}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to widget registry module", {"path": "src/widgets/registry.py"}),
                ("workspace.read", "Read WidgetRegistry source after locate", {"path": "src/widgets/registry.py"}),
            ),
        ),
        # --- field lookup ---
        _Family(
            "ctx_field_partner_vat", "field_lookup", "odoo",
            "workspace.search", "Find field vat on partner model",
            {"query": "vat", "scope": "models/partner.py", "intent": "locate_field_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located vat field on partner",
                            {"match_count": 2, "ranked_artifacts": [_art("models/partner.py", "vat", "field", 95, "field_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "vat field not found in scoped partner file",
                            {"match_count": 0, "query": "vat", "scope": "models/partner.py"}),
            ),
            followup_intents=(
                ("workspace.read", "Read partner model around vat field", {"path": "models/partner.py"}),
            ),
        ),
        _Family(
            "ctx_field_order_partner_id", "field_lookup", "odoo",
            "workspace.search", "Find related field partner_id on sale.order",
            {"query": "partner_id", "scope": "models/sale_order.py", "intent": "locate_field_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located partner_id Many2one on sale.order",
                            {"match_count": 1, "ranked_artifacts": [_art("models/sale_order.py", "partner_id", "field", 97, "field_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "partner_id appears in model and view layers",
                            {"match_count": 7, "ambiguous": True}),
            ),
        ),
        _Family(
            "ctx_field_move_state", "field_lookup", "odoo",
            "workspace.search", "Locate Selection field state on account.move",
            {"query": "fields.Selection", "scope": "models/account_move.py", "intent": "locate_field_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located state Selection field on account.move",
                            {"match_count": 1, "ranked_artifacts": [_art("models/account_move.py", "state", "field", 94, "field_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "No Selection field match in account_move scope",
                            {"match_count": 0}),
            ),
        ),
        _Family(
            "ctx_field_generic_timeout", "field_lookup", None,
            "workspace.search", "Locate timeout configuration attribute",
            {"query": "DEFAULT_TIMEOUT", "intent": "locate_constant"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located DEFAULT_TIMEOUT constant",
                            {"match_count": 1, "ranked_artifacts": [_art("src/config/defaults.py", "DEFAULT_TIMEOUT", "constant", 96, "direct_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Multiple timeout constants across packages",
                            {"match_count": 9, "ambiguous": True}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to defaults module for timeout", {"path": "src/config/defaults.py"}),
            ),
        ),
        # --- method lookup ---
        _Family(
            "ctx_method_action_confirm", "method_lookup", "odoo",
            "workspace.search", "Locate sale.order action_confirm method",
            {"query": "def action_confirm", "scope": "models/sale_order.py", "intent": "locate_method_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located action_confirm on sale.order",
                            {"match_count": 1, "ranked_artifacts": [_art("models/sale_order.py", "action_confirm", "method", 98, "method_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "action_confirm overridden in multiple inheritance layers",
                            {"match_count": 3, "ambiguous": True}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "action_confirm not present in scoped sale_order file",
                            {"match_count": 0}),
            ),
            followup_intents=(
                ("workspace.read", "Read action_confirm implementation after locate", {"path": "models/sale_order.py"}),
            ),
        ),
        _Family(
            "ctx_method_create", "method_lookup", "odoo",
            "workspace.search", "Locate create override on stock.picking",
            {"query": "def create", "scope": "models/stock_picking.py", "intent": "locate_method_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located create override on stock.picking",
                            {"match_count": 1, "ranked_artifacts": [_art("models/stock_picking.py", "create", "method", 93, "method_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "create matches include helpers and overrides",
                            {"match_count": 11, "ambiguous": True, "match_quality": "fuzzy"}),
            ),
        ),
        _Family(
            "ctx_method_generic_serialize", "method_lookup", None,
            "workspace.search", "Locate serialize_response method implementation",
            {"query": "def serialize_response", "intent": "locate_method_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located serialize_response method",
                            {"match_count": 1, "ranked_artifacts": [_art("src/api/serializers.py", "serialize_response", "method", 97, "method_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "serialize_response not found",
                            {"match_count": 0, "query": "serialize_response"}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to serializers module", {"path": "src/api/serializers.py"}),
                ("workspace.read", "Read serialize_response after locate", {"path": "src/api/serializers.py"}),
            ),
        ),
        # --- view lookup ---
        _Family(
            "ctx_view_sale_form", "view_lookup", "odoo",
            "workspace.search", "Locate view architecture for sale.order form",
            {"query": "sale.order.form", "intent": "locate_view"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located sale.order form view",
                            {"match_count": 1, "ranked_artifacts": [_art("views/sale_views.xml", "sale.order.form", "view", 97, "view_id")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Form and tree views both mention sale.order",
                            {"match_count": 4, "ambiguous": True}),
            ),
            followup_intents=(
                ("workspace.read", "Read sale form view XML after locate", {"path": "views/sale_views.xml"}),
            ),
        ),
        _Family(
            "ctx_view_partner_kanban", "view_lookup", "odoo",
            "workspace.search", "Locate kanban view for res.partner",
            {"query": "res.partner.kanban", "intent": "locate_view"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located partner kanban view",
                            {"match_count": 1, "ranked_artifacts": [_art("views/partner_views.xml", "res.partner.kanban", "view", 96, "view_id")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "Partner kanban view id not found",
                            {"match_count": 0, "query": "res.partner.kanban"}),
            ),
        ),
        _Family(
            "ctx_view_generic_template", "view_lookup", None,
            "workspace.search", "Locate HTML template for dashboard page",
            {"query": "dashboard.html", "intent": "locate_template"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located dashboard HTML template",
                            {"match_count": 1, "ranked_artifacts": [_art("templates/dashboard.html", "dashboard", "template", 95, "direct_path")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Multiple dashboard templates across themes",
                            {"match_count": 5, "ambiguous": True}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to templates directory", {"path": "templates"}),
            ),
        ),
        # --- manifest / module ---
        _Family(
            "ctx_manifest_sale", "manifest_lookup", "odoo",
            "workspace.search", "Locate __manifest__ for sale module",
            {"query": "__manifest__.py", "scope": "addons/sale", "intent": "locate_manifest"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located sale module manifest",
                            {"match_count": 1, "ranked_artifacts": [_art("addons/sale/__manifest__.py", "__manifest__", "manifest", 100, "direct_path")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "No __manifest__.py under addons/sale",
                            {"match_count": 0}),
            ),
            followup_intents=(
                ("workspace.read", "Read sale __manifest__ after locate", {"path": "addons/sale/__manifest__.py"}),
                ("repository.inspect", "Inspect sale addon package layout", {"path": "addons/sale"}),
            ),
        ),
        _Family(
            "ctx_manifest_account", "manifest_lookup", "odoo",
            "workspace.search", "Locate account module manifest dependencies",
            {"query": "'depends'", "scope": "addons/account/__manifest__.py", "intent": "locate_manifest"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located depends list in account manifest",
                            {"match_count": 1, "ranked_artifacts": [_art("addons/account/__manifest__.py", "depends", "manifest", 94, "manifest_key")]}),
            ),
            followup_intents=(
                ("repository.inspect", "Inspect account addon root", {"path": "addons/account"}),
            ),
        ),
        _Family(
            "ctx_security_access_csv", "module_lookup", "odoo",
            "workspace.search", "Locate security access CSV for module",
            {"query": "ir.model.access.csv", "intent": "locate_security"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located ir.model.access.csv",
                            {"match_count": 1, "ranked_artifacts": [_art("security/ir.model.access.csv", "ir.model.access.csv", "security", 99, "direct_path")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Multiple access CSV files across addons",
                            {"match_count": 8, "ambiguous": True}),
            ),
        ),
        # --- controller ---
        _Family(
            "ctx_controller_http", "controller_lookup", None,
            "workspace.search", "Find HTTP controller route definition",
            {"query": "type='http'", "intent": "locate_controller"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located HTTP controller route",
                            {"match_count": 2, "ranked_artifacts": [_art("controllers/main.py", "http", "controller", 92, "route_marker")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "HTTP route markers appear in several controllers",
                            {"match_count": 12, "ambiguous": True, "match_quality": "fuzzy"}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to controllers package", {"path": "controllers"}),
                ("workspace.read", "Read main controller after locate", {"path": "controllers/main.py"}),
            ),
        ),
        _Family(
            "ctx_controller_json_rpc", "controller_lookup", "odoo",
            "workspace.search", "Locate JSON-RPC controller endpoint",
            {"query": "type='json'", "scope": "controllers", "intent": "locate_controller"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located JSON controller endpoint",
                            {"match_count": 1, "ranked_artifacts": [_art("controllers/portal.py", "json", "controller", 93, "route_marker")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "No JSON route under controllers scope",
                            {"match_count": 0}),
            ),
        ),
        # --- repository inspect families ---
        _Family(
            "ctx_repo_models_pkg", "repository_inspection", None,
            "repository.inspect", "Inspect repository tree for models package",
            {"path": "models", "intent": "locate_package"},
            (
                _ObsVariant("inspect_obs", "repository.inspect", "repository_status", "succeeded",
                            "Repository models package layout available",
                            {"models_present": True, "path": "models"}),
                _ObsVariant("failed", "repository.inspect", "repository_status", "failed",
                            "models package missing from repository root",
                            {"models_present": False, "path": "models"}),
            ),
        ),
        _Family(
            "ctx_repo_addons_sale", "repository_inspection", "odoo",
            "repository.inspect", "Locate repository module layout for sale",
            {"root_hint": "addons/sale", "intent": "locate_module_layout"},
            (
                _ObsVariant("inspect_obs", "repository.inspect", "repository_status", "succeeded",
                            "Sale addon layout available for locate",
                            {"path": "addons/sale", "has_manifest": True, "has_models": True}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate into sale addon models", {"path": "addons/sale/models"}),
            ),
        ),
        _Family(
            "ctx_repo_addons_root", "repository_inspection", "odoo",
            "repository.inspect", "Inspect addons root for module discovery",
            {"path": "addons", "intent": "locate_addons_root"},
            (
                _ObsVariant("inspect_obs", "repository.inspect", "repository_status", "succeeded",
                            "Addons root present for module discovery",
                            {"path": "addons", "module_count_hint": "many"}),
                _ObsVariant("partial", "repository.inspect", "repository_status", "partial",
                            "Addons root present but sparse",
                            {"path": "addons", "module_count_hint": "few"}),
            ),
        ),
        _Family(
            "ctx_repo_src_layout", "repository_inspection", None,
            "repository.inspect", "Inspect src package layout for application code",
            {"path": "src", "intent": "locate_package"},
            (
                _ObsVariant("inspect_obs", "repository.inspect", "repository_status", "succeeded",
                            "src package tree available",
                            {"path": "src", "packages": ["api", "config", "widgets"]}),
                _ObsVariant("failed", "repository.inspect", "repository_status", "failed",
                            "src package not found at repository root",
                            {"path": "src", "present": False}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to src/api after inspect", {"path": "src/api"}),
            ),
        ),
        _Family(
            "ctx_repo_tests_tree", "repository_inspection", None,
            "repository.inspect", "Inspect tests directory structure",
            {"path": "tests", "intent": "locate_package"},
            (
                _ObsVariant("inspect_obs", "repository.inspect", "repository_status", "succeeded",
                            "tests directory layout available",
                            {"path": "tests", "present": True}),
            ),
        ),
        # --- navigate / read primary families ---
        _Family(
            "ctx_nav_controllers", "workspace_navigation", None,
            "workspace.navigate", "Navigate to controllers package after retrieval",
            {"path": "controllers", "intent": "open_located_path"},
            (
                _ObsVariant("navigate_obs", "workspace.navigate", "workspace_change", "succeeded",
                            "Navigated to controllers package",
                            {"path": "controllers", "opened": True}),
                _ObsVariant("failed", "workspace.navigate", "workspace_change", "failed",
                            "controllers path not reachable",
                            {"path": "controllers", "opened": False}),
            ),
        ),
        _Family(
            "ctx_nav_partner_model", "workspace_navigation", "odoo",
            "workspace.navigate", "Navigate to partner model source path",
            {"path": "models/partner.py", "intent": "open_located_path"},
            (
                _ObsVariant("navigate_obs", "workspace.navigate", "workspace_change", "succeeded",
                            "Navigated to models/partner.py",
                            {"path": "models/partner.py", "opened": True}),
            ),
            followup_intents=(
                ("workspace.read", "Read partner model after navigation", {"path": "models/partner.py"}),
            ),
        ),
        _Family(
            "ctx_read_controller_main", "workspace_read", None,
            "workspace.read", "Read located controller after retrieval",
            {"path": "controllers/main.py", "intent": "read_located_artifact"},
            (
                _ObsVariant("read_obs", "workspace.read", "artifact_result", "succeeded",
                            "Retrieved controller content after locate",
                            {"path": "controllers/main.py", "bytes_read": 2048}),
                _ObsVariant("failed", "workspace.read", "artifact_result", "failed",
                            "Could not read controllers/main.py",
                            {"path": "controllers/main.py", "bytes_read": 0}),
            ),
        ),
        _Family(
            "ctx_read_manifest_sale", "workspace_read", "odoo",
            "workspace.read", "Read located sale module manifest",
            {"path": "addons/sale/__manifest__.py", "intent": "read_located_artifact"},
            (
                _ObsVariant("read_obs", "workspace.read", "artifact_result", "succeeded",
                            "Retrieved sale manifest content",
                            {"path": "addons/sale/__manifest__.py", "bytes_read": 512}),
            ),
        ),
        # --- related / reference locate ---
        _Family(
            "ctx_related_inherit", "related_artifact", "odoo",
            "workspace.search", "Locate _inherit declaration for sale.order extension",
            {"query": "_inherit = 'sale.order'", "intent": "locate_related_artifact"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located sale.order inherit extension",
                            {"match_count": 2, "ranked_artifacts": [_art("models/sale_order_ext.py", "_inherit", "inherit", 91, "inherit_marker")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Multiple modules inherit sale.order",
                            {"match_count": 6, "ambiguous": True}),
            ),
        ),
        _Family(
            "ctx_related_xpath", "related_artifact", "odoo",
            "workspace.search", "Locate xpath inherit for partner form",
            {"query": "expr=\"//field[@name='vat']\"", "intent": "locate_related_artifact"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located xpath targeting partner vat field",
                            {"match_count": 1, "ranked_artifacts": [_art("views/partner_ext.xml", "vat", "xpath", 90, "xpath_expr")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "No xpath matching partner vat field",
                            {"match_count": 0}),
            ),
        ),
        _Family(
            "ctx_impl_ref_generic", "implementation_reference", None,
            "workspace.search", "Locate callers of parse_config helper",
            {"query": "parse_config(", "intent": "locate_implementation_reference"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located parse_config call sites",
                            {"match_count": 3, "ranked_artifacts": [
                                _art("src/app/bootstrap.py", "parse_config", "reference", 88, "call_site"),
                                _art("src/cli/main.py", "parse_config", "reference", 80, "call_site"),
                            ]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "parse_config references include tests and docs",
                            {"match_count": 14, "ambiguous": True, "match_quality": "fuzzy"}),
            ),
            followup_intents=(
                ("workspace.read", "Read bootstrap call site after locate", {"path": "src/app/bootstrap.py"}),
            ),
        ),
        _Family(
            "ctx_impl_ref_odoo_onchange", "implementation_reference", "odoo",
            "workspace.search", "Locate @api.onchange handlers for partner_id",
            {"query": "@api.onchange('partner_id')", "intent": "locate_implementation_reference"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located onchange handlers for partner_id",
                            {"match_count": 2, "ranked_artifacts": [_art("models/sale_order.py", "partner_id", "onchange", 92, "decorator_match")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "No onchange for partner_id found",
                            {"match_count": 0}),
            ),
        ),
        # --- exact vs fuzzy / ranked ---
        _Family(
            "ctx_ranked_exact_model", "ranked_results", "odoo",
            "workspace.search", "Exact-match locate for model stock.quant",
            {"query": "stock.quant", "intent": "locate_model_definition", "match_mode": "exact"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Exact match ranked stock.quant first",
                            {"match_count": 1, "match_quality": "exact", "ranked_artifacts": [_art("models/stock_quant.py", "stock.quant", "model", 100, "exact_match")]}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to stock_quant model file", {"path": "models/stock_quant.py"}),
            ),
        ),
        _Family(
            "ctx_ranked_fuzzy_confirm", "ranked_results", "odoo",
            "workspace.search", "Fuzzy search for confirm actions on orders",
            {"query": "confirm", "intent": "locate_method_definition", "match_mode": "fuzzy"},
            (
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Fuzzy confirm search returned many ranked hits",
                            {"match_count": 22, "match_quality": "fuzzy", "ambiguous": True,
                             "ranked_artifacts": [
                                 _art("models/sale_order.py", "action_confirm", "method", 85, "fuzzy_rank"),
                                 _art("models/purchase_order.py", "button_confirm", "method", 70, "fuzzy_rank"),
                             ]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "Fuzzy confirm search emptied by over-constrained filters",
                            {"match_count": 0, "match_quality": "fuzzy"}),
            ),
        ),
        _Family(
            "ctx_ranked_generic_init", "ranked_results", None,
            "workspace.search", "Ranked locate for package __init__ exports",
            {"query": "__all__", "scope": "src", "intent": "locate_export", "match_mode": "exact"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Ranked __all__ exports under src",
                            {"match_count": 4, "match_quality": "exact", "ranked_artifacts": [
                                _art("src/api/__init__.py", "__all__", "export", 90, "exact_match"),
                                _art("src/config/__init__.py", "__all__", "export", 88, "exact_match"),
                            ]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Conflicting export lists across packages",
                            {"match_count": 4, "ambiguous": True, "conflicting": True}),
            ),
        ),
        # --- additional Odoo diversity ---
        _Family(
            "ctx_model_hr_employee", "model_lookup", "odoo",
            "workspace.search", "Locate hr.employee model definition",
            {"query": "hr.employee", "intent": "locate_model_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located hr.employee model",
                            {"match_count": 1, "ranked_artifacts": [_art("models/hr_employee.py", "hr.employee", "model", 100, "direct_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "hr.employee not in workspace",
                            {"match_count": 0}),
            ),
            followup_intents=(
                ("workspace.read", "Read hr_employee model after locate", {"path": "models/hr_employee.py"}),
            ),
        ),
        _Family(
            "ctx_field_qty_available", "field_lookup", "odoo",
            "workspace.search", "Locate qty_available computed field",
            {"query": "qty_available", "intent": "locate_field_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located qty_available field",
                            {"match_count": 2, "ranked_artifacts": [_art("models/product_product.py", "qty_available", "field", 95, "field_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "qty_available defined on product and template",
                            {"match_count": 3, "ambiguous": True}),
            ),
        ),
        _Family(
            "ctx_method_action_post", "method_lookup", "odoo",
            "workspace.search", "Locate action_post on account.move",
            {"query": "def action_post", "intent": "locate_method_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located action_post method",
                            {"match_count": 1, "ranked_artifacts": [_art("models/account_move.py", "action_post", "method", 98, "method_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "action_post appears in move and payment flows",
                            {"match_count": 5, "ambiguous": True}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to account_move.py", {"path": "models/account_move.py"}),
            ),
        ),
        _Family(
            "ctx_data_xml_demo", "related_artifact", "odoo",
            "workspace.search", "Locate demo data XML for sale orders",
            {"query": "sale_order_demo", "scope": "data", "intent": "locate_related_artifact"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located sale demo data XML",
                            {"match_count": 1, "ranked_artifacts": [_art("data/sale_demo.xml", "sale_order_demo", "data", 94, "xml_id")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "sale demo data not found under data/",
                            {"match_count": 0}),
            ),
        ),
        # --- additional generic diversity ---
        _Family(
            "ctx_generic_router", "controller_lookup", None,
            "workspace.search", "Locate FastAPI router registration",
            {"query": "APIRouter(", "intent": "locate_controller"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located APIRouter registration",
                            {"match_count": 2, "ranked_artifacts": [_art("src/api/routes.py", "APIRouter", "router", 96, "direct_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "APIRouter not found in workspace",
                            {"match_count": 0}),
            ),
            followup_intents=(
                ("workspace.read", "Read routes module after locate", {"path": "src/api/routes.py"}),
            ),
        ),
        _Family(
            "ctx_generic_dockerfile", "related_artifact", None,
            "workspace.search", "Locate Dockerfile for service image",
            {"query": "FROM ", "scope": "Dockerfile", "intent": "locate_related_artifact"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located base image line in Dockerfile",
                            {"match_count": 1, "ranked_artifacts": [_art("Dockerfile", "FROM", "config", 90, "direct_path")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Multiple Dockerfiles across services",
                            {"match_count": 4, "ambiguous": True}),
            ),
            followup_intents=(
                ("repository.inspect", "Inspect repository root for container files", {"path": "."}),
            ),
        ),
        _Family(
            "ctx_generic_schema", "field_lookup", None,
            "workspace.search", "Locate JSON schema for request payload",
            {"query": "RequestPayload", "intent": "locate_class_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located RequestPayload schema class",
                            {"match_count": 1, "ranked_artifacts": [_art("src/api/schemas.py", "RequestPayload", "class", 97, "class_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "RequestPayload schema missing",
                            {"match_count": 0}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to schemas module", {"path": "src/api/schemas.py"}),
                ("workspace.read", "Read RequestPayload schema after locate", {"path": "src/api/schemas.py"}),
            ),
        ),
        _Family(
            "ctx_generic_migration", "related_artifact", None,
            "workspace.search", "Locate database migration for users table",
            {"query": "create_table('users'", "intent": "locate_related_artifact"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located users table migration",
                            {"match_count": 1, "ranked_artifacts": [_art("migrations/001_users.py", "users", "migration", 95, "direct_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "users table referenced in several migrations",
                            {"match_count": 3, "ambiguous": True}),
            ),
        ),
        _Family(
            "ctx_nav_generic_config", "workspace_navigation", None,
            "workspace.navigate", "Navigate to configuration package",
            {"path": "src/config", "intent": "open_located_path"},
            (
                _ObsVariant("navigate_obs", "workspace.navigate", "workspace_change", "succeeded",
                            "Navigated to src/config",
                            {"path": "src/config", "opened": True}),
                _ObsVariant("failed", "workspace.navigate", "workspace_change", "failed",
                            "src/config path unreachable",
                            {"path": "src/config", "opened": False}),
            ),
            followup_intents=(
                ("workspace.read", "Read settings module after navigate", {"path": "src/config/settings.py"}),
            ),
        ),
        _Family(
            "ctx_read_generic_readme", "workspace_read", None,
            "workspace.read", "Read repository README after locate",
            {"path": "README.md", "intent": "read_located_artifact"},
            (
                _ObsVariant("read_obs", "workspace.read", "artifact_result", "succeeded",
                            "Retrieved README content",
                            {"path": "README.md", "bytes_read": 4096}),
                _ObsVariant("failed", "workspace.read", "artifact_result", "failed",
                            "README.md missing",
                            {"path": "README.md", "bytes_read": 0}),
            ),
        ),
        _Family(
            "ctx_conflicting_paths", "conflicting_evidence", "odoo",
            "workspace.search", "Locate product.template with conflicting path evidence",
            {"query": "product.template", "intent": "locate_model_definition"},
            (
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Conflicting locate evidence for product.template",
                            {"match_count": 2, "ambiguous": True, "conflicting": True,
                             "ranked_artifacts": [
                                 _art("models/product_template.py", "product.template", "model", 90, "direct_definition"),
                                 _art("legacy/product.py", "product.template", "model", 60, "legacy_path"),
                             ]}),
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Resolved product.template to primary models path",
                            {"match_count": 1, "ranked_artifacts": [_art("models/product_template.py", "product.template", "model", 100, "direct_definition")]}),
            ),
            followup_intents=(
                ("workspace.read", "Read primary product_template after resolve", {"path": "models/product_template.py"}),
            ),
        ),
        _Family(
            "ctx_no_result_totally_missing", "no_result", None,
            "workspace.search", "Search for TotallyMissingSymbol with no results",
            {"query": "TotallyMissingSymbol", "intent": "locate_class_definition"},
            (
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "No matches for TotallyMissingSymbol",
                            {"match_count": 0, "query": "TotallyMissingSymbol"}),
            ),
        ),
        _Family(
            "ctx_no_result_odoo_ghost", "no_result", "odoo",
            "workspace.search", "Search for ghost.model.xyz with no results",
            {"query": "ghost.model.xyz", "intent": "locate_model_definition"},
            (
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "No matches for ghost.model.xyz",
                            {"match_count": 0, "query": "ghost.model.xyz"}),
            ),
        ),
        _Family(
            "ctx_repo_security_dir", "repository_inspection", "odoo",
            "repository.inspect", "Inspect security directory for access rules",
            {"path": "security", "intent": "locate_package"},
            (
                _ObsVariant("inspect_obs", "repository.inspect", "repository_status", "succeeded",
                            "security directory present",
                            {"path": "security", "present": True}),
                _ObsVariant("failed", "repository.inspect", "repository_status", "failed",
                            "security directory missing",
                            {"path": "security", "present": False}),
            ),
            followup_intents=(
                ("workspace.search", "Locate ACL CSV after security inspect", {"query": "ir.model.access.csv", "scope": "security"}),
            ),
        ),
        _Family(
            "ctx_nav_views_odoo", "workspace_navigation", "odoo",
            "workspace.navigate", "Navigate to views package for form locate follow-up",
            {"path": "views", "intent": "open_located_path"},
            (
                _ObsVariant("navigate_obs", "workspace.navigate", "workspace_change", "succeeded",
                            "Navigated to views package",
                            {"path": "views", "opened": True}),
            ),
            followup_intents=(
                ("workspace.search", "Search form views after navigating to views/", {"query": ".form", "scope": "views"}),
                ("workspace.read", "Read sale_views.xml after navigate", {"path": "views/sale_views.xml"}),
            ),
        ),
        _Family(
            "ctx_read_odoo_security", "workspace_read", "odoo",
            "workspace.read", "Read located security access CSV",
            {"path": "security/ir.model.access.csv", "intent": "read_located_artifact"},
            (
                _ObsVariant("read_obs", "workspace.read", "artifact_result", "succeeded",
                            "Retrieved access CSV content",
                            {"path": "security/ir.model.access.csv", "bytes_read": 1024}),
                _ObsVariant("failed", "workspace.read", "artifact_result", "failed",
                            "Access CSV unreadable",
                            {"path": "security/ir.model.access.csv", "bytes_read": 0}),
            ),
        ),
        # --- AT-6.2.1 expansion toward preferred ~250 ---
        _Family(
            "ctx_model_project_task", "model_lookup", "odoo",
            "workspace.search", "Locate project.task model definition",
            {"query": "project.task", "intent": "locate_model_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located project.task model",
                            {"match_count": 1, "ranked_artifacts": [_art("models/project_task.py", "project.task", "model", 100, "direct_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "project.task and project.project both matched",
                            {"match_count": 2, "ambiguous": True}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "project.task missing from workspace",
                            {"match_count": 0, "query": "project.task"}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to project_task model", {"path": "models/project_task.py"}),
                ("workspace.read", "Read project.task source after locate", {"path": "models/project_task.py"}),
            ),
        ),
        _Family(
            "ctx_field_date_order", "field_lookup", "odoo",
            "workspace.search", "Locate date_order field on sale.order",
            {"query": "date_order", "scope": "models/sale_order.py", "intent": "locate_field_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located date_order field",
                            {"match_count": 1, "ranked_artifacts": [_art("models/sale_order.py", "date_order", "field", 96, "field_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "date_order not in sale_order scope",
                            {"match_count": 0, "query": "date_order", "scope": "models/sale_order.py"}),
            ),
        ),
        _Family(
            "ctx_method_button_validate", "method_lookup", "odoo",
            "workspace.search", "Locate button_validate on stock.picking",
            {"query": "def button_validate", "intent": "locate_method_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located button_validate method",
                            {"match_count": 1, "ranked_artifacts": [_art("models/stock_picking.py", "button_validate", "method", 97, "method_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "validate helpers compete with button_validate",
                            {"match_count": 4, "ambiguous": True, "match_quality": "fuzzy"}),
            ),
            followup_intents=(
                ("workspace.read", "Read button_validate after locate", {"path": "models/stock_picking.py"}),
            ),
        ),
        _Family(
            "ctx_view_invoice_tree", "view_lookup", "odoo",
            "workspace.search", "Locate tree view for account.move invoices",
            {"query": "account.move.tree", "intent": "locate_view"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located account.move tree view",
                            {"match_count": 1, "ranked_artifacts": [_art("views/account_move_views.xml", "account.move.tree", "view", 95, "view_id")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "account.move.tree view id missing",
                            {"match_count": 0, "query": "account.move.tree"}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to account move views", {"path": "views/account_move_views.xml"}),
            ),
        ),
        _Family(
            "ctx_generic_logger", "method_lookup", None,
            "workspace.search", "Locate configure_logging helper",
            {"query": "def configure_logging", "intent": "locate_method_definition"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located configure_logging helper",
                            {"match_count": 1, "ranked_artifacts": [_art("src/observability/logging.py", "configure_logging", "method", 96, "method_definition")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "logging setup helpers found in multiple packages",
                            {"match_count": 5, "ambiguous": True}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "configure_logging not found",
                            {"match_count": 0, "query": "configure_logging"}),
            ),
            followup_intents=(
                ("workspace.navigate", "Navigate to observability logging module", {"path": "src/observability/logging.py"}),
                ("workspace.read", "Read configure_logging after locate", {"path": "src/observability/logging.py"}),
            ),
        ),
        _Family(
            "ctx_generic_cache", "field_lookup", None,
            "workspace.search", "Locate CACHE_TTL constant definition",
            {"query": "CACHE_TTL", "intent": "locate_constant"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located CACHE_TTL constant",
                            {"match_count": 1, "ranked_artifacts": [_art("src/cache/policy.py", "CACHE_TTL", "constant", 94, "direct_definition")]}),
                _ObsVariant("failed", "workspace.search", "search_result", "failed",
                            "CACHE_TTL constant missing",
                            {"match_count": 0, "query": "CACHE_TTL"}),
            ),
            followup_intents=(
                ("repository.inspect", "Inspect cache package after locate", {"path": "src/cache"}),
            ),
        ),
        _Family(
            "ctx_repo_i18n", "repository_inspection", "odoo",
            "repository.inspect", "Inspect i18n directory for translation files",
            {"path": "i18n", "intent": "locate_package"},
            (
                _ObsVariant("inspect_obs", "repository.inspect", "repository_status", "succeeded",
                            "i18n directory present for translation locate",
                            {"path": "i18n", "present": True}),
                _ObsVariant("failed", "repository.inspect", "repository_status", "failed",
                            "i18n directory missing",
                            {"path": "i18n", "present": False}),
            ),
            followup_intents=(
                ("workspace.search", "Locate PO file after i18n inspect", {"query": ".po", "scope": "i18n"}),
            ),
        ),
        _Family(
            "ctx_generic_env_example", "related_artifact", None,
            "workspace.search", "Locate environment example configuration",
            {"query": ".env.example", "intent": "locate_related_artifact"},
            (
                _ObsVariant("success", "workspace.search", "search_result", "succeeded",
                            "Located .env.example",
                            {"match_count": 1, "ranked_artifacts": [_art(".env.example", "ENV", "config", 93, "direct_path")]}),
                _ObsVariant("partial", "workspace.search", "search_result", "partial",
                            "Multiple env example files across services",
                            {"match_count": 3, "ambiguous": True}),
            ),
            followup_intents=(
                ("workspace.read", "Read .env.example after locate", {"path": ".env.example"}),
            ),
        ),
    ]
    # fmt: on
    return tuple(specs)


def _extra(family: _Family, role: str, task_role: str) -> dict[str, Any]:
    return {
        "context_corpus_version": CONTEXT_CONTROLLED_VERSION,
        "scenario_family": family.family_id,
        "context_task": family.task,
        "context_role": role,
        "context_scenario": task_role,
        "fp2_native": True,
        "legacy": False,
        "fixture_scale": False,
        "controlled_context": True,
    }


def _make_intent(
    *,
    index: int,
    family: _Family,
    capability_id: str,
    objective: str,
    args: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    rec = CapabilityIntentRecord(
        record_type="capability_intent",
        record_id=rid("ctxc-ci", index),
        capability_id=capability_id,
        objective=objective,
        args=args,
        reason="fp2-context-controlled",
        provider_capability="context",
        domain_specialization=family.domain,
        metadata=fixture_metadata(
            generator=CONTEXT_CONTROLLED_GENERATOR,
            index=index,
            provider_capability="context",
            domain_specialization=family.domain,
            extra=_extra(family, role, "locate_intent"),
        ),
    )
    return rec.to_dict()


def _make_obs(
    *,
    index: int,
    family: _Family,
    variant: _ObsVariant,
) -> dict[str, Any]:
    evidence = dict(variant.evidence)
    evidence.setdefault("scenario_family", family.family_id)
    evidence.setdefault("context_task", family.task)
    rec = ObservationRecord(
        record_type="observation",
        record_id=rid("ctxc-obs", index),
        kind=variant.kind,
        status=variant.status,
        capability_id=variant.capability_id,
        summary=f"[{family.family_id}] {variant.summary}",
        evidence=evidence,
        provider_capability="context",
        domain_specialization=family.domain,
        metadata=fixture_metadata(
            generator=CONTEXT_CONTROLLED_GENERATOR,
            index=1000 + index,
            provider_capability="context",
            domain_specialization=family.domain,
            extra=_extra(family, variant.role, "locate_observation"),
        ),
    )
    return rec.to_dict()


def generate_context_controlled_records() -> list[dict[str, Any]]:
    """Generate controlled Context corpus (~250), deterministic, family-linked."""
    records: list[dict[str, Any]] = []
    ci_i = 0
    obs_i = 0
    for family in _families():
        ci_i += 1
        records.append(
            _make_intent(
                index=ci_i,
                family=family,
                capability_id=family.intent_cap,
                objective=family.intent_objective,
                args=dict(family.intent_args),
                role="primary_intent",
            )
        )
        for variant in family.observations:
            obs_i += 1
            records.append(_make_obs(index=obs_i, family=family, variant=variant))
        for fi, (cap, objective, args) in enumerate(family.followup_intents, start=1):
            ci_i += 1
            records.append(
                _make_intent(
                    index=ci_i,
                    family=family,
                    capability_id=cap,
                    objective=objective,
                    args=dict(args),
                    role=f"followup_intent_{fi}",
                )
            )
            # Pair each follow-up intent with a matching success observation —
            # include family_id so normalized fingerprints stay distinct.
            obs_i += 1
            target = args.get("path") or args.get("root_hint") or args.get("query") or family.family_id
            if cap == "workspace.navigate":
                variant = _ObsVariant(
                    "navigate_obs",
                    cap,
                    "workspace_change",
                    "succeeded",
                    f"Family {family.family_id}: navigated to retrieved target {target}",
                    {
                        "path": args.get("path"),
                        "opened": True,
                        "scenario_family": family.family_id,
                        "locate_target": str(target),
                    },
                )
            elif cap == "workspace.read":
                variant = _ObsVariant(
                    "read_obs",
                    cap,
                    "artifact_result",
                    "succeeded",
                    f"Family {family.family_id}: read retrieved artifact {target}",
                    {
                        "path": args.get("path"),
                        "bytes_read": 1024 + fi,
                        "scenario_family": family.family_id,
                        "locate_target": str(target),
                    },
                )
            elif cap == "repository.inspect":
                variant = _ObsVariant(
                    "inspect_obs",
                    cap,
                    "repository_status",
                    "succeeded",
                    f"Family {family.family_id}: inspected retrieved layout {target}",
                    {
                        "path": args.get("path") or args.get("root_hint"),
                        "present": True,
                        "scenario_family": family.family_id,
                        "locate_target": str(target),
                    },
                )
            else:
                variant = _ObsVariant(
                    "success",
                    cap,
                    "search_result",
                    "succeeded",
                    f"Family {family.family_id}: follow-up search for {args.get('query')}",
                    {
                        "match_count": 1,
                        "query": args.get("query"),
                        "scope": args.get("scope"),
                        "scenario_family": family.family_id,
                    },
                )
            records.append(_make_obs(index=obs_i, family=family, variant=variant))

    # Validate + enforce invariants
    seen_fp: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for rec in records:
        validate_record_mapping(rec)
        assert rec["provider_capability"] == "context"
        assert rec["record_type"] in CONTEXT_ALLOWED_RECORD_TYPES
        assert rec["training_contract_version"] == SYSTEM_TRAINING_CONTRACT_VERSION
        assert rec["metadata"].get("scenario_family")
        assert rec["metadata"].get("legacy") is False
        eng = extract_engineering_capability(rec)
        assert eng in CONTEXT_LOCATE_CAPS
        assert not scan_forbidden_how(rec)
        assert not scan_taxonomy(rec)
        fp = fingerprint_record(rec)
        if fp in seen_fp:
            continue  # drop exact semantic duplicates if any
        seen_fp.add(fp)
        deduped.append(rec)

    if not (TARGET_MIN <= len(deduped) <= TARGET_MAX):
        raise RuntimeError(
            f"controlled Context count {len(deduped)} outside [{TARGET_MIN}, {TARGET_MAX}]"
        )
    return deduped


def normalized_fingerprint(record: Mapping[str, Any]) -> str:
    """Normalize superficial tokens for near-duplicate reporting."""
    blob = stable_dumps(
        {
            "record_type": record.get("record_type"),
            "provider": record.get("provider_capability"),
            "domain": record.get("domain_specialization") or "generic",
            "cap": extract_engineering_capability(record),
            "input": record.get("input"),
            "expected_output": record.get("expected_output"),
            "evidence": record.get("evidence"),
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


def analyze_context_controlled(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(r["record_type"]) for r in records)
    caps = Counter(extract_engineering_capability(r) or "?" for r in records)
    families = Counter(
        str((r.get("metadata") or {}).get("scenario_family") or "?") for r in records
    )
    domain = domain_distribution(list(records))
    exact = find_duplicates(list(records))
    normalized = find_normalized_duplicates(records)

    # Splits + leakage
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
        or "context_v1_0" in json.dumps(r)
        or r.get("provider_capability") != "context"
    )

    neg_results = [evaluate_negative_case(c) for c in NEGATIVE_CASES]
    neg_ok = all(r["matched"] for r in neg_results)
    # Ensure no negative cases leak into corpus ids / quality_corpus markers
    neg_contam = sum(
        1
        for r in records
        if str((r.get("metadata") or {}).get("quality_corpus") or "").startswith("negative")
    )

    pack = format_fp2_pack(list(records), pack="development")
    pack_ok = len(pack) == len(records) and all(
        ex.dataset_type.value == "context" for ex in pack
    )

    hard_fail = False
    reasons: list[str] = []
    if not (TARGET_MIN <= len(records) <= TARGET_MAX):
        hard_fail = True
        reasons.append("count_out_of_band")
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
    if not neg_ok:
        hard_fail = True
        reasons.append("negatives_mismatched")
    # Capability coverage: each locate cap must appear
    for cap in CONTEXT_LOCATE_CAPS:
        if caps.get(cap, 0) < 10:
            hard_fail = True
            reasons.append(f"weak_cap:{cap}")
    if by_type.get("capability_intent", 0) < 60 or by_type.get("observation", 0) < 60:
        hard_fail = True
        reasons.append("record_type_imbalance")
    if domain["odoo"] < 40 or domain["generic"] < 40:
        hard_fail = True
        reasons.append("domain_imbalance")
    # Soft: normalized near-duplicates warn but do not auto-fail unless severe
    if normalized["normalized_duplicate_groups"] > 5:
        hard_fail = True
        reasons.append("normalized_duplicates_severe")

    verdict = "CONTEXT_CORPUS_NEEDS_FIXES" if hard_fail else "CONTEXT_CORPUS_READY"

    scorecard = {
        "native_records": len(records),
        "capability_intents": by_type.get("capability_intent", 0),
        "observations": by_type.get("observation", 0),
        "odoo": domain["odoo"],
        "generic": domain["generic"],
        "workspace.search": caps.get("workspace.search", 0),
        "workspace.navigate": caps.get("workspace.navigate", 0),
        "workspace.read": caps.get("workspace.read", 0),
        "repository.inspect": caps.get("repository.inspect", 0),
        "scenario_families": len(families),
        "largest_family_concentration": max(families.values()) if families else 0,
        "duplicate_groups": exact["duplicate_groups"],
        "normalized_duplicate_groups": normalized["normalized_duplicate_groups"],
        "forbidden_how": how_hits,
        "taxonomy_violations": tax_hits,
        "negative_contamination": neg_contam,
        "train": split_counts.get("train", 0),
        "validation": split_counts.get("validation", 0),
        "test": split_counts.get("test", 0),
        "family_leakage": len(leakage),
        "pack_validity": pack_ok,
        "negatives_ok": neg_ok,
    }

    return {
        "version": CONTEXT_CONTROLLED_VERSION,
        "verdict": verdict,
        "fail_reasons": reasons,
        "scorecard": scorecard,
        "by_record_type": dict(by_type),
        "by_capability": dict(caps),
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
    }


def emit_context_controlled_corpus(
    *,
    training_root: Path | None = None,
    datasets_root: Path | None = None,
) -> dict[str, Any]:
    """Write controlled Context corpus; never touches AT-6.2 fixtures or batch_2."""
    before = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert before == BATCH2_CHECKSUM

    # Guard: do not overwrite AT-6.2 fixture tree
    assert _AT62_FIXTURES.is_dir()
    at62_manifest = json.loads((_AT62_FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    assert at62_manifest.get("version") == "fp2-context-1.0.0"

    records = generate_context_controlled_records()
    analysis = analyze_context_controlled(records)
    pack = analysis.pop("pack_examples")
    split_rows = analysis.pop("split_rows")

    train_out = Path(training_root or _DEFAULT_TRAINING)
    data_out = Path(datasets_root or _DEFAULT_DATASETS)
    written: dict[str, str] = {}

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_type[str(r["record_type"])].append(r)

    for root in (train_out, data_out):
        root.mkdir(parents=True, exist_ok=True)
        # Refuse writing into protected trees
        resolved = root.resolve()
        assert "controlled_batch_2" not in str(resolved)
        assert resolved != _AT62_FIXTURES.resolve()

        write_jsonl(root / "context_native.jsonl", records)
        written[str(root / "context_native.jsonl")] = f"{len(records)} records"
        for rtype, rows in sorted(by_type.items()):
            write_jsonl(root / f"{rtype}.jsonl", rows)
            written[str(root / f"{rtype}.jsonl")] = f"{len(rows)} records"

        pack_rows = [
            {
                "example_id": ex.example_id,
                "dataset_type": ex.dataset_type.value,
                "messages": [dict(m) for m in ex.messages],
                "metadata": dict(ex.metadata),
            }
            for ex in pack
        ]
        write_jsonl(root / "pack_context.jsonl", pack_rows)
        written[str(root / "pack_context.jsonl")] = f"{len(pack_rows)} examples"
        write_jsonl(root / "splits.jsonl", split_rows)

        manifest = {
            "version": CONTEXT_CONTROLLED_VERSION,
            "training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
            "provider_capability": "context",
            "product_plane": "development",
            "generator": CONTEXT_CONTROLLED_GENERATOR,
            "total_records": len(records),
            "by_record_type": {k: len(v) for k, v in sorted(by_type.items())},
            "scorecard": analysis["scorecard"],
            "verdict": analysis["verdict"],
            "legacy_projection": False,
            "controlled_batch_2_modified": False,
            "at62_fixtures_modified": False,
            "split_version": "fp2-split-1.0.0",
            "notes": [
                "AT-6.2.1 controlled Context corpus",
                "Independent from controlled_batch_2 and AT-6.2 fixtures",
                "Not a production adapter pack / not for certification",
            ],
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "quality_report.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        gen_meta = {
            "version": CONTEXT_CONTROLLED_VERSION,
            "family_count": len(_families()),
            "target_preferred": TARGET_PREFERRED,
            "actual_count": len(records),
            "locate_capabilities": sorted(CONTEXT_LOCATE_CAPS),
            "scenario_family_mechanism": (
                "metadata.scenario_family groups intent + observation variants; "
                "assign_split uses family: key (fp2-split-1.0.0)"
            ),
        }
        (root / "generation_metadata.json").write_text(
            json.dumps(gen_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written[str(root / "manifest.json")] = "manifest"

    after = json.loads((BATCH2 / "manifest.json").read_text(encoding="utf-8"))["checksum"]
    assert after == before == BATCH2_CHECKSUM

    return {
        "version": CONTEXT_CONTROLLED_VERSION,
        "count": len(records),
        "verdict": analysis["verdict"],
        "scorecard": analysis["scorecard"],
        "fail_reasons": analysis["fail_reasons"],
        "written": written,
        "batch2_checksum": after,
    }
