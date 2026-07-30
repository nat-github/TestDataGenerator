"""Tests for the CLI command modules.

These were the lowest-coverage modules in the package (cdc 20%,
config_tools 29%, quality 11%) and both critical bugs found in review lived
here: `pii-scan` calling a `ConfigParser.parse_config()` that does not
exist, and the Delta conversion unlinking the flat parquet before the write
that replaces it.

Neither bug was exotic. Both were simply never executed by a test. These
call the handlers with real argparse namespaces and real files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sdp.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "configs" / "yaml"


def _args(command: str, *argv: str):
    """Build a real argparse namespace so defaults match production."""
    return build_parser().parse_args([command, *argv])


@pytest.fixture
def simple_config(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        """
config_format: sdp-yaml-v1
tables:
  - name: customers
    active: true
    num_rows: 20
    columns:
      - name: customer_id
        data_type: N38
        is_pk: true
      - name: email
        data_type: VA256
        special_rules: EMAIL
      - name: full_name
        data_type: VA64
        special_rules: NAME
      - name: tier
        data_type: VA16
        business_values: "gold;silver"
""".strip(),
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# pii-scan — the crash bug
# ---------------------------------------------------------------------------


def test_pii_scan_on_yaml_config_does_not_crash(simple_config, capsys):
    """Regression: this called ConfigParser.parse_config(), which has never
    existed. Every YAML pii-scan raised AttributeError, swallowed into a
    generic 'pii-scan failed' message."""
    from sdp.cli_commands.config_tools import run_pii_scan

    rc = run_pii_scan(_args("pii-scan", "--input", simple_config))
    assert rc == 0
    output = capsys.readouterr().out
    assert output.strip(), "pii-scan produced no report"


def test_pii_scan_detects_pii_column_names(simple_config, capsys):
    from sdp.cli_commands.config_tools import run_pii_scan

    run_pii_scan(_args("pii-scan", "--input", simple_config))
    output = capsys.readouterr().out.lower()
    assert "email" in output or "name" in output


def test_pii_scan_accepts_json_configs(tmp_path, capsys):
    from sdp.cli_commands.config_tools import run_pii_scan

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "config_format": "sdp-json-v1",
        "tables": [{"name": "t", "columns": [
            {"name": "email", "data_type": "VA256"},
        ]}],
    }), encoding="utf-8")

    assert run_pii_scan(_args("pii-scan", "--input", str(path))) == 0


def test_pii_scan_on_csv_data(tmp_path, capsys):
    from sdp.cli_commands.config_tools import run_pii_scan

    csv = tmp_path / "data.csv"
    pd.DataFrame({"email": ["a@b.com"], "amount": [1]}).to_csv(csv, index=False)

    assert run_pii_scan(_args("pii-scan", "--input", str(csv))) == 0


def test_pii_scan_on_parquet_data(tmp_path):
    from sdp.cli_commands.config_tools import run_pii_scan

    pq = tmp_path / "data.parquet"
    pd.DataFrame({"email": ["a@b.com"], "amount": [1]}).to_parquet(pq, index=False)

    assert run_pii_scan(_args("pii-scan", "--input", str(pq))) == 0


def test_pii_scan_rejects_unsupported_type(tmp_path):
    from sdp.cli_commands.config_tools import run_pii_scan

    other = tmp_path / "notes.txt"
    other.write_text("hello", encoding="utf-8")

    assert run_pii_scan(_args("pii-scan", "--input", str(other))) == 1


def test_pii_scan_missing_file_returns_error(tmp_path):
    from sdp.cli_commands.config_tools import run_pii_scan

    assert run_pii_scan(_args("pii-scan", "--input", str(tmp_path / "gone.yaml"))) == 1


def test_pii_scan_unloadable_config_returns_error(tmp_path):
    from sdp.cli_commands.config_tools import run_pii_scan

    bad = tmp_path / "bad.yaml"
    bad.write_text("::: not yaml :::\n", encoding="utf-8")
    assert run_pii_scan(_args("pii-scan", "--input", str(bad))) == 1


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


def test_lint_passes_on_a_good_config(simple_config):
    from sdp.cli_commands.config_tools import run_lint

    assert run_lint(_args("lint", "--config", simple_config)) == 0


def test_lint_fails_on_a_missing_file(tmp_path):
    from sdp.cli_commands.config_tools import run_lint

    assert run_lint(_args("lint", "--config", str(tmp_path / "gone.yaml"))) == 1


def test_lint_reports_schema_violations_without_failing(tmp_path, capsys):
    from sdp.cli_commands.config_tools import run_lint

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "config_format: sdp-yaml-v1\n"
        "tables:\n  - name: t\n    columns:\n      - name: c\n        data_type: VA8\n"
        "workflows:\n  - name: w\n    table: t\n    state_column: c\n"
        "    typo_field: oops\n"
        "    transitions:\n      - {from: A, to: B}\n",
        encoding="utf-8",
    )
    assert run_lint(_args("lint", "--config", str(path))) == 0
    assert "typo_field" in capsys.readouterr().out


def test_strict_schema_turns_violations_into_failure(tmp_path):
    from sdp.cli_commands.config_tools import run_lint

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "config_format: sdp-yaml-v1\n"
        "tables:\n  - name: t\n    columns:\n      - name: c\n        data_type: VA8\n"
        "workflows:\n  - name: w\n    table: t\n    state_column: c\n"
        "    typo_field: oops\n"
        "    transitions:\n      - {from: A, to: B}\n",
        encoding="utf-8",
    )
    assert run_lint(_args("lint", "--config", str(path), "--strict-schema")) == 1


# ---------------------------------------------------------------------------
# quality-report
# ---------------------------------------------------------------------------


@pytest.fixture
def generated_dir(tmp_path):
    out = tmp_path / "generated"
    out.mkdir()
    pd.DataFrame({
        "id": range(60),
        "amount": [float(i % 17) for i in range(60)],
        "tier": ["gold", "silver"] * 30,
    }).to_parquet(out / "customers.parquet", index=False)
    return out


def test_quality_report_univariate_only(generated_dir):
    from sdp.cli_commands.quality import run_quality_report

    assert run_quality_report(_args(
        "quality-report", "--generated", str(generated_dir))) == 0


def test_quality_report_against_a_source(generated_dir, tmp_path):
    from sdp.cli_commands.quality import run_quality_report

    source = tmp_path / "source"
    source.mkdir()
    pd.read_parquet(generated_dir / "customers.parquet").to_parquet(
        source / "customers.parquet", index=False)

    assert run_quality_report(_args(
        "quality-report", "--generated", str(generated_dir),
        "--source", str(source), "--no-utility")) == 0


def test_quality_report_writes_json_and_html(generated_dir, tmp_path):
    from sdp.cli_commands.quality import run_quality_report

    json_out = tmp_path / "q.json"
    html_out = tmp_path / "q.html"
    rc = run_quality_report(_args(
        "quality-report", "--generated", str(generated_dir),
        "--output-json", str(json_out), "--output-html", str(html_out)))

    assert rc == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))["tables"]
    assert "<" in html_out.read_text(encoding="utf-8")


def test_quality_report_missing_directory_returns_error(tmp_path):
    from sdp.cli_commands.quality import run_quality_report

    assert run_quality_report(_args(
        "quality-report", "--generated", str(tmp_path / "gone"))) == 1


def test_quality_report_verbose_prints_markdown(generated_dir, capsys):
    from sdp.cli_commands.quality import run_quality_report

    run_quality_report(_args(
        "quality-report", "--generated", str(generated_dir), "--verbose"))
    assert "Synthetic Data Quality Report" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# validate-data / contract commands
# ---------------------------------------------------------------------------


def test_validate_data_missing_data_dir_returns_error(simple_config, tmp_path):
    from sdp.cli_commands.quality import run_validate_data

    rc = run_validate_data(_args(
        "validate-data", "--config", simple_config, "--input", str(tmp_path / "gone")))
    assert rc == 1, "a missing --input directory must be a clean error, not a traceback"


def test_contract_diff_identical_configs_has_no_breaking_changes(simple_config):
    from sdp.cli_commands.quality import run_contract_diff_cmd

    assert run_contract_diff_cmd(_args(
        "contract-diff", "--old", simple_config, "--new", simple_config)) == 0


def test_contract_diff_detects_a_dropped_column(simple_config, tmp_path):
    from sdp.cli_commands.quality import run_contract_diff_cmd

    new = tmp_path / "new.yaml"
    new.write_text(
        """
config_format: sdp-yaml-v1
tables:
  - name: customers
    active: true
    num_rows: 20
    columns:
      - name: customer_id
        data_type: N38
        is_pk: true
      - name: email
        data_type: VA256
        special_rules: EMAIL
      - name: full_name
        data_type: VA64
        special_rules: NAME
""".strip(),
        encoding="utf-8",
    )
    # Check the fixture before trusting the assertion — a "new" config that
    # accidentally equals the old one makes this test vacuously pass.
    assert "tier" not in new.read_text(encoding="utf-8")

    rc = run_contract_diff_cmd(_args(
        "contract-diff", "--old", simple_config, "--new", str(new),
        "--fail-on-breaking"))
    # 2 is the documented "breaking changes found" code; 1 means the command
    # itself failed. Pinning the exact value keeps CI able to tell them apart.
    assert rc == 2, "dropping a column is a breaking change"


def test_contract_diff_missing_file_returns_error(simple_config, tmp_path):
    from sdp.cli_commands.quality import run_contract_diff_cmd

    assert run_contract_diff_cmd(_args(
        "contract-diff", "--old", simple_config,
        "--new", str(tmp_path / "gone.yaml"))) == 1


# ---------------------------------------------------------------------------
# delta / scd2 (cdc.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def cdc_config(tmp_path):
    path = tmp_path / "cdc.yaml"
    path.write_text(
        """
config_format: sdp-yaml-v1
tables:
  - name: accounts
    active: true
    num_rows: 10
    primary_key_columns: [account_id]
    cdc:
      mode: scd2
      track: [status]
    columns:
      - name: account_id
        data_type: N38
        is_pk: true
      - name: status
        data_type: VA16
        business_values: "OPEN;CLOSED"
""".strip(),
        encoding="utf-8",
    )
    return str(path)


def _snapshot(directory: Path, statuses):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "account_id": list(range(len(statuses))),
        "status": statuses,
    }).to_parquet(directory / "accounts.parquet", index=False)
    return directory


def test_delta_detects_updates(cdc_config, tmp_path):
    from sdp.cli_commands.cdc import run_delta

    previous = _snapshot(tmp_path / "v1", ["OPEN", "OPEN", "OPEN"])
    current = _snapshot(tmp_path / "v2", ["OPEN", "CLOSED", "OPEN"])
    out = tmp_path / "delta"

    rc = run_delta(_args("delta", "--config", cdc_config,
                         "--previous", str(previous), "--current", str(current),
                         "--output", str(out)))
    assert rc == 0
    assert out.exists()


def test_delta_missing_previous_returns_error(cdc_config, tmp_path):
    from sdp.cli_commands.cdc import run_delta

    current = _snapshot(tmp_path / "v2", ["OPEN"])
    rc = run_delta(_args("delta", "--config", cdc_config,
                         "--previous", str(tmp_path / "gone"),
                         "--current", str(current),
                         "--output", str(tmp_path / "d")))
    assert rc == 1


def test_delta_missing_config_returns_error(tmp_path):
    from sdp.cli_commands.cdc import run_delta

    rc = run_delta(_args("delta", "--config", str(tmp_path / "gone.yaml"),
                         "--previous", str(tmp_path), "--current", str(tmp_path),
                         "--output", str(tmp_path / "d")))
    assert rc == 1


def test_scd2_produces_effective_dated_history(cdc_config, tmp_path):
    from sdp.cli_commands.cdc import run_scd2

    previous = _snapshot(tmp_path / "v1", ["OPEN", "OPEN"])
    current = _snapshot(tmp_path / "v2", ["OPEN", "CLOSED"])
    out = tmp_path / "scd2"

    rc = run_scd2(_args("scd2", "--config", cdc_config,
                        "--previous", str(previous), "--current", str(current),
                        "--output", str(out)))
    assert rc == 0

    history = pd.read_parquet(out / "accounts.parquet")
    assert "is_current" in history.columns
    assert "effective_from_ts" in history.columns
    # The changed key must now have two versions, only one of them current.
    changed = history[history["account_id"] == 1]
    assert len(changed) == 2
    assert int(changed["is_current"].sum()) == 1


def test_scd2_missing_current_returns_error(cdc_config, tmp_path):
    from sdp.cli_commands.cdc import run_scd2

    previous = _snapshot(tmp_path / "v1", ["OPEN"])
    rc = run_scd2(_args("scd2", "--config", cdc_config,
                        "--previous", str(previous),
                        "--current", str(tmp_path / "gone"),
                        "--output", str(tmp_path / "s")))
    assert rc == 1


# ---------------------------------------------------------------------------
# Delta conversion — the data-loss bug
# ---------------------------------------------------------------------------


def test_failed_delta_write_leaves_the_flat_parquet(tmp_path, monkeypatch):
    """Regression: the flat parquet was unlinked *before* the Delta write.

    A failing write therefore destroyed the source and produced nothing —
    the run ended with neither the parquet nor the Delta table.
    """
    import sdp.services.generation as gen

    out = tmp_path / "out"
    out.mkdir()
    flat = out / "orders.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(flat, index=False)

    pytest.importorskip("deltalake")
    import deltalake

    def exploding_writer(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(deltalake, "write_deltalake", exploding_writer)

    with pytest.raises(RuntimeError, match="disk full"):
        gen._write_delta_outputs(str(out), partition_col="run_date",
                                 partition_value="2024-01-01")

    assert flat.exists(), "the flat parquet was destroyed by a failed Delta write"
    assert pd.read_parquet(flat).shape[0] == 2


def test_successful_delta_write_removes_the_flat_parquet(tmp_path):
    """The conversion is meant to replace the flat file — on success only."""
    pytest.importorskip("deltalake")
    import sdp.services.generation as gen

    out = tmp_path / "out"
    out.mkdir()
    flat = out / "orders.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(flat, index=False)

    gen._write_delta_outputs(str(out), partition_col="run_date",
                             partition_value="2024-01-01")

    assert not flat.exists()
    assert (out / "orders" / "_delta_log").is_dir()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    "generate", "delta", "scd2", "lint", "enrich", "collibra-import",
    "infer-config", "pii-scan", "infer-relationships", "record-feedback",
    "mock-init", "mock-render", "mock-lint", "mock-enrich",
    "validate-data", "quality-report", "contract-test", "contract-diff",
])
def test_every_command_resolves_to_a_handler(command):
    """A subcommand with no handler fails only when someone runs it."""
    import sdp.cli as cli

    assert command in cli.KNOWN_COMMANDS


def test_main_rejects_an_unknown_command(capsys):
    import sdp.cli as cli

    with pytest.raises(SystemExit):
        cli.main(["definitely-not-a-command", "--config", "x"])
