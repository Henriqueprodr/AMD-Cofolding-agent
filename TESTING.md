# Testing Guide

This document lists all tests that can be run against the current codebase (Phases 1, 2, 3, 4, and 5a). Tests are organized into three tiers: unit tests that need no external dependencies, integration tests that require REINVENT4, and end-to-end tests that verify the full agent stack.

## Quick Start

```bash
cd dd_agentic
pip install pytest numpy pydantic PyYAML pandas fastapi httpx requests rdkit gemmi
python -m pytest tests/ -v
```

All unit tests mock REINVENT4 imports and external services, so they run anywhere with Python 3.10+ and the packages above. `rdkit` and `gemmi` are not mocked: candidate harvesting sanitizes every SMILES it returns, and the structure-prep numbering, construct and alignment stages are thin wrappers over gemmi, so tests for those import the real libraries.

---

## Tier 1: Unit Tests (no REINVENT4 needed)

These tests validate internal logic with mocked REINVENT4. Run them on any machine.

### TTP Schema Validation (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 1 | Transform config types | All 8 transform types accepted (`sigmoid`, `reverse_sigmoid`, `double_sigmoid`, `right_step`, `left_step`, `step`, `value_mapping`, `exponential_decay`) | `pytest tests/test_ttp.py::TestTransformConfig -v` |
| 2 | Transform extra params | Extra params like `low`, `high`, `k` pass through | same class |
| 3 | Invalid transform rejected | Unknown transform type raises `ValidationError` | same class |
| 4 | Target protein sequence | Valid amino-acid sequences accepted, short (<10 aa) and invalid characters rejected, auto-uppercased | `pytest tests/test_ttp.py::TestTargetConfig -v` |
| 5 | Pre-scoring components | Valid components (`QED`, `MolecularWeight`, etc.) accepted; unknown components and zero/negative weights rejected | `pytest tests/test_ttp.py::TestPreScoringComponent -v` |
| 6 | Pre-scoring duplicates | Duplicate component names within pre-scoring rejected | `pytest tests/test_ttp.py::TestPreScoringSection -v` |
| 7 | Scoring objectives | Native requires `component`, external requires `service`; `service_url` is filled in for a service this deployment configures and required for any other; valid entries accepted | `pytest tests/test_ttp.py::TestScoringObjective -v` |
| 8 | Scoring section | `depends_on` referencing nonexistent objective rejected; self-dependency rejected; duplicate names rejected | `pytest tests/test_ttp.py::TestScoringSection -v` |
| 9 | Constraint config | `custom_alerts` requires non-empty `smarts` list | `pytest tests/test_ttp.py::TestConstraintConfig -v` |
| 10 | Full TTP validation | Minimal and full TTPs parse correctly; invalid ID rejected; Boltz without target rejected; cross-section name collision rejected | `pytest tests/test_ttp.py::TestTTP -v` |
| 11 | TTP properties | `has_external_objectives`, `native_objective_names`, `external_objective_names` return correct values | same class |
| 12 | TTP JSON roundtrip | `model_dump` -> `model_validate` preserves all fields | same class |

### TTP Store (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 13 | Create and get | TTP persisted as JSON, retrievable by ID | `pytest tests/test_ttp.py::TestTTPStore -v` |
| 14 | Duplicate create rejected | `FileExistsError` on duplicate ID | same class |
| 15 | Get nonexistent raises | `FileNotFoundError` for unknown ID | same class |
| 16 | Update increments version | Version auto-incremented, fields merged | same class |
| 17 | Update preserves ID | Cannot change ID via update | same class |
| 18 | Version history | All versions retrievable by number; latest returned by default | same class |
| 19 | List all | Returns summary metadata for all TTPs | same class |
| 20 | Delete | Removes TTP and all version files | same class |
| 21 | Delete nonexistent raises | `FileNotFoundError` for unknown ID | same class |
| 22 | Exists check | `exists()` returns correct boolean | same class |

### TTP Mapper (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 23 | Extract native components | Correct flat list from full TTP (6 pre-scoring + 1 native scoring); external excluded | `pytest tests/test_ttp.py::TestMapper -v` |
| 24 | Transform preservation | Transforms pass through to extracted components | same class |
| 25 | Full scorer config structure | `ttp_to_scorer_config` produces correct shape (type + component list), including externals | `pytest tests/test_ttp.py::TestTTPToScorerConfig -v` |
| 26 | Native component format | Endpoint has `name`, `weight`, `transform`, `params` | same class |
| 27 | External component format | `ExternalRESTScorer` entries for Boltz/PoseBusters with service params | same class |
| 28 | Constraint format | `CustomAlerts` appended with SMARTS params | same class |
| 29 | Ordering | Pre-scoring first, scoring middle, constraints last | same class |
| 30 | Aggregation from TTP | TTP's `aggregation` field used as scorer `type` | same class |
| 31 | Boltz target injection | Protein sequence, MSA path, PDB path injected into external params | same class |
| 32 | PoseBusters depends_on | Pipeline dependency preserved in config | same class |
| 33 | Service params forwarded | `affinity_mode`, `recycling_steps`, etc. passed through | same class |
| 34 | External transforms | Transform preserved on external objectives | same class |
| 35 | Null transform omitted | Objectives with `transform: null` have no transform key | same class |
| 36 | Native-only scorer config | `ttp_to_native_scorer_config` excludes all externals | `pytest tests/test_ttp.py::TestTTPToNativeScorerConfig -v` |
| 37 | Native-only includes constraints | Constraints included even in native-only config | same class |
| 38 | All-external returns empty | Native-only config has empty component list for all-external TTP | same class |

### Topological Sort (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 39 | Boltz before PoseBusters | Dependency ordering respected in output | `pytest tests/test_ttp.py::TestTopologicalSort -v` |
| 40 | Reversed input reordered | Even if PoseBusters listed before Boltz in TTP, output is correct | same class |
| 41 | No dependencies preserves order | Original order kept when no `depends_on` | same class |
| 42 | Chain A -> B -> C | Multi-hop chains correctly ordered | same class |

### Mapper Summary (`tests/test_ttp.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 43 | Full TTP summary | Correct counts for pre-scoring, native, external, constraints, total | `pytest tests/test_ttp.py::TestGetMapperSummary -v` |
| 44 | Component names | Native and external names listed correctly | same class |
| 45 | Pipeline dependencies listed | `from`/`to` pairs in summary | same class |
| 46 | Minimal TTP summary | Zeros for absent sections | same class |

### Score Molecules Routing (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 47 | Empty SMILES | Returns empty result, no error | `pytest tests/test_scoring.py::TestScoreMoleculesRouting -v` |
| 48 | No mode specified | Returns error with guidance | same class |
| 49 | Both ttp_id and inline TTP | Returns error (ambiguous) | same class |

### TTP Resolution in Scoring (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 50 | Nonexistent ttp_id | Returns "not found" error | `pytest tests/test_scoring.py::TestTTPResolution -v` |
| 51 | Invalid inline TTP | Returns "Invalid" error | same class |
| 52 | All-external TTP | Returns error about external-only TTPs | same class |

### TTP-Based Scoring (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 53 | Minimal inline TTP | Mock scorer called, results formatted correctly | `pytest tests/test_scoring.py::TestScoreWithInlineTTP -v` |
| 54 | TTP metadata in response | `ttp_id`, `ttp_version` present | same class |
| 55 | Mapper summary in response | `mapper_summary` with correct counts | same class |
| 56 | External objectives noted | `note` field lists skipped external objectives | same class |
| 57 | No external = no note | Native-only TTP has no `note` | same class |
| 58 | Pre-scoring included | Pre-scoring components counted in mapper summary | same class |
| 59 | Scorer config structure | Config passed to Scorer has correct shape | same class |
| 60 | Constraints in config | `CustomAlerts` present in scorer config | same class |

### Stored TTP Scoring (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 61 | Score by ttp_id | Stored TTP loaded and scored | `pytest tests/test_scoring.py::TestScoreWithStoredTTP -v` |
| 62 | Updated version reflected | After `update`, scoring uses new version | same class |

### Summary Statistics (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 63 | Summary fields | `n_molecules`, `n_valid`, `mean_total_score`, `best_total_score`, `per_component` all correct | `pytest tests/test_scoring.py::TestSummaryStatistics -v` |

### Scoring Error Handling (`tests/test_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 64 | Scorer creation failure | Catches `RuntimeError`, returns `"Failed to create scorer"` | `pytest tests/test_scoring.py::TestScoringErrors -v` |
| 65 | Compute results failure | Catches `RuntimeError`, returns `"Scoring failed"` | same class |

### Boltz2 Wrapper Service (`tests/test_boltz_wrapper.py`)

Tests for the Boltz2 FastAPI wrapper (`services/boltz2/app.py`). Validates YAML generation, CLI command construction, output parsing, and endpoint behavior. No GPU or real Boltz installation required -- subprocess calls and file I/O are mocked.

**Dependencies**: `fastapi`, `httpx`, `pyyaml`

#### YAML Generation

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 66 | Basic structure | YAML has `version: 1`, protein + ligand entries, affinity properties | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_basic_structure -v` |
| 67 | MSA path included | `msa` field present in protein entry when `msa_path` is provided | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_msa_path_included_when_provided -v` |
| 68 | MSA path absent | `msa` field absent when `msa_path` is `None` | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_msa_path_absent_when_none -v` |
| 69 | SMILES preserved | Stereochemistry-rich SMILES passed through without modification | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_smiles_preserved_exactly -v` |
| 70 | YAML roundtrip | Dump and reload preserves all fields | `pytest tests/test_boltz_wrapper.py::TestYAMLGeneration::test_yaml_roundtrip -v` |

#### Batch YAML Writing

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 71 | One YAML per SMILES | `mol_0.yaml`, `mol_1.yaml`, ... created for each molecule | `pytest tests/test_boltz_wrapper.py::TestBatchYAMLWriting::test_creates_one_yaml_per_smiles -v` |
| 72 | Content matches SMILES | Each YAML file contains the correct SMILES and MSA path | `pytest tests/test_boltz_wrapper.py::TestBatchYAMLWriting::test_yaml_content_matches_smiles -v` |
| 73 | Directory creation | Batch directory created automatically, including nested parents | `pytest tests/test_boltz_wrapper.py::TestBatchYAMLWriting::test_creates_batch_directory -v` |

#### Predict Command Construction

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 74 | Basic command | `boltz predict <dir> --out_dir --override --recycling_steps` present | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_basic_command_structure -v` |
| 75 | MSA server flag | `--use_msa_server` included when `use_msa_server=True` | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_msa_server_flag -v` |
| 76 | No MSA server default | Flag absent when using pre-computed MSA | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_no_msa_server_flag_by_default -v` |
| 77 | MW correction flag | `--affinity_mw_correction` included when enabled | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_affinity_mw_correction_flag -v` |
| 78 | Custom sampling steps | `--sampling_steps` and `--sampling_steps_affinity` with custom values | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_custom_sampling_steps -v` |
| 79 | Output format | `--output_format mmcif` when specified | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_output_format -v` |
| 253 | Optimized kernels off | `--no_kernels` present while `BOLTZ_NO_KERNELS` keeps them off | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_no_kernels_flag_present_when_kernels_disabled -v` |
| 254 | Optimized kernels on | `--no_kernels` absent once the kernels are enabled | `pytest tests/test_boltz_wrapper.py::TestPredictCommand::test_no_kernels_flag_absent_when_kernels_enabled -v` |

#### Kernel Flag Parsing

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 255 | `BOLTZ_NO_KERNELS` | Unset or blank keeps today's behaviour; `0`, `false`, `no`, and `off` enable the optimized kernels | `pytest tests/test_boltz_wrapper.py::TestKernelFlagEnv -v` |

#### Output Parsing

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 80 | Complete outputs | Affinity score, confidence metadata, and structure path all parsed | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_parse_complete_outputs -v` |
| 81 | Probability binary mode | `affinity_probability_binary` selected when `affinity_mode="probability_binary"` | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_parse_probability_binary_mode -v` |
| 82 | Missing affinity JSON | Score is `None`, structure path still returned | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_missing_affinity_returns_none_score -v` |
| 83 | Missing mol directory | All outputs `None` when molecule directory absent | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_missing_mol_dir_returns_none -v` |
| 84 | Missing structure file | Score still parsed, structure path is `None` | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_missing_structure_file -v` |
| 85 | mmCIF output format | Structure path ends in `.mmcif` when format is `mmcif` | `pytest tests/test_boltz_wrapper.py::TestOutputParsing::test_mmcif_output_format -v` |

#### Results Directory Discovery

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 86 | Finds expected dir | Locates `boltz_results_<batch>/predictions` under output dir | `pytest tests/test_boltz_wrapper.py::TestFindResultsDir::test_finds_expected_dir -v` |
| 87 | Missing dir returns None | Returns `None` when no results directory exists | `pytest tests/test_boltz_wrapper.py::TestFindResultsDir::test_returns_none_when_missing -v` |

#### Response Assembly

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 88 | Multi-molecule response | Scores, metadata lists, and structure paths aligned across molecules | `pytest tests/test_boltz_wrapper.py::TestResponseAssembly::test_assembles_multi_molecule_response -v` |
| 89 | Partial failure | First molecule scored, second returns `None` when its directory is absent | `pytest tests/test_boltz_wrapper.py::TestResponseAssembly::test_handles_partial_failure -v` |
| 90 | Metadata alignment | Metadata keys padded with `None` for molecules missing that field | `pytest tests/test_boltz_wrapper.py::TestResponseAssembly::test_metadata_aligned_with_molecules -v` |

#### FastAPI Endpoints

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 91 | Health endpoint | `GET /health` reports `status: ok` | `pytest tests/test_boltz_wrapper.py::TestHealthEndpoint::test_health_returns_ok -v` |
| 256 | Health reports kernels | `GET /health` carries `no_kernels`, so an A/B run can confirm the live arm | `pytest tests/test_boltz_wrapper.py::TestHealthEndpoint::test_health_reports_the_kernel_flag -v` |
| 92 | Missing MSA config → 400 | No `msa_path` and `use_msa_server=False` returns HTTP 400 | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_missing_msa_config_returns_400 -v` |
| 93 | Subprocess failure → 500 | Non-zero exit code from `boltz predict` returns HTTP 500 | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_subprocess_failure_returns_500 -v` |
| 94 | Successful scoring | Mocked subprocess + output → HTTP 200 with scores and artifacts | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_successful_scoring -v` |
| 95 | Timeout → 500 | `subprocess.TimeoutExpired` returns HTTP 500 with "timed out" | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_timeout_returns_500 -v` |
| 96 | Missing results dir → 500 | Subprocess succeeds but no output directory → HTTP 500 | `pytest tests/test_boltz_wrapper.py::TestScoreEndpoint::test_missing_results_dir_returns_500 -v` |

### PoseBusters Wrapper Service (`tests/test_posebusters_wrapper.py`)

Tests for the PoseBusters FastAPI wrapper (`services/posebusters/app.py`). Validates fraction-of-checks scoring, metadata alignment, mode handling, and endpoint validation. PoseBusters library is mocked.

**Dependencies**: `fastapi`, `httpx`, `pandas` (the `TestBoltzPdbToSdf` class additionally requires `rdkit` and is skipped automatically when it is not installed)

#### Single-Molecule Scoring

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 97 | All checks pass | Score is `1.0`, all checks `True`, counts correct | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_all_checks_pass -v` |
| 98 | Partial checks pass | Score is `n_passed / n_total` (e.g., 3/4) | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_partial_checks_pass -v` |
| 99 | Empty DataFrame → None | Unloadable structure returns `None` score | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_empty_dataframe_returns_none -v` |
| 100 | Exception → None | RDKit parse failure returns `None` score | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_exception_returns_none -v` |
| 101 | Passes mol_cond/mol_true | Protein and reference paths forwarded to PoseBusters.bust() | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_passes_mol_cond_and_mol_true -v` |
| 102 | No bool columns → None | DataFrame with only non-bool columns returns `None` | `pytest tests/test_posebusters_wrapper.py::TestScoreSingleMolecule::test_no_bool_columns_returns_none -v` |

#### Multi-Molecule Scoring

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 103 | Multiple molecules | Scores computed independently per molecule | `pytest tests/test_posebusters_wrapper.py::TestScoreMolecules::test_scores_multiple_molecules -v` |
| 104 | Metadata aligned | Per-check boolean arrays positionally aligned with SMILES | `pytest tests/test_posebusters_wrapper.py::TestScoreMolecules::test_metadata_aligned_with_smiles -v` |
| 105 | Partial failure | First molecule succeeds, second fails → `[0.85, None]` | `pytest tests/test_posebusters_wrapper.py::TestScoreMolecules::test_partial_failure -v` |
| 106 | Reference paths forwarded | `reference_paths` passed through to PoseBusters for redock mode | `pytest tests/test_posebusters_wrapper.py::TestScoreMolecules::test_reference_paths_forwarded -v` |

#### Endpoint Validation

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 107 | Health endpoint | `GET /health` returns `{"status": "ok"}` | `pytest tests/test_posebusters_wrapper.py::TestHealthEndpoint -v` |
| 108 | Invalid mode → 400 | Unknown mode string returns HTTP 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_invalid_mode -v` |
| 109 | dock needs protein_pdb | `dock` mode without `protein_pdb_path` returns 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_dock_mode_requires_protein_pdb -v` |
| 110 | redock needs references | `redock` mode without `reference_paths` returns 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_redock_mode_requires_reference_paths -v` |
| 111 | SMILES/paths mismatch → 400 | Different-length `smiles` and `structure_paths` returns 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_smiles_structure_paths_length_mismatch -v` |
| 112 | Reference paths mismatch → 400 | Different-length `reference_paths` returns 400 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_reference_paths_length_mismatch -v` |
| 113 | mol mode no protein | `mol` mode does not require protein → 200 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointValidation::test_mol_mode_does_not_require_protein -v` |

#### Endpoint Success & Failure

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 114 | Successful scoring | Returns scores, metadata, and correct HTTP 200 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointSuccess::test_returns_scores_and_metadata -v` |
| 115 | Internal error → 500 | RuntimeError in scoring returns HTTP 500 | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointSuccess::test_internal_error_returns_500 -v` |
| 116 | Null scores for failures | Failed molecules have `null` in scores array | `pytest tests/test_posebusters_wrapper.py::TestScoreEndpointSuccess::test_null_scores_for_failed_molecules -v` |

#### Mode Constants

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 117 | Protein modes valid | `MODES_REQUIRING_PROTEIN` is a subset of `VALID_MODES` | `pytest tests/test_posebusters_wrapper.py::TestModeConstants -v` |
| 118 | Reference modes valid | `MODES_REQUIRING_REFERENCE` subset of `VALID_MODES` and `MODES_REQUIRING_PROTEIN` | same class |

### External Scoring Integration (`tests/test_external_scoring.py`)

Tests for the ExternalRESTScorer REINVENT4 plugin, `list_external_services` MCP tool, pipeline artifact injection, and NaN propagation. All HTTP calls are mocked.

**Dependencies**: `numpy`, `requests`

#### ExternalRESTScorer

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 119 | Init extracts URL/service | `service_url` and `service` unpacked from list params | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_init_extracts_url_and_service -v` |
| 120 | Init builds request params | `protein_sequence`, `affinity_mode`, etc. extracted into `request_params` dict | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_init_builds_request_params -v` |
| 121 | Init skips None values | `None` params (e.g., `msa_path`) excluded from `request_params` | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_init_skips_none_values -v` |
| 122 | Artifacts initially None | `last_artifacts` is `None` before any call | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_last_artifacts_initially_none -v` |
| 123 | Successful scoring | HTTP 200 response → correct scores array and `last_artifacts` populated | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_successful_scoring -v` |
| 124 | Null → NaN conversion | JSON `null` in scores → `np.nan` in numpy array | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_null_to_nan_conversion -v` |
| 125 | All NaN on connection error | `ConnectionError` → all-NaN scores (RL loop continues) | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_all_nan_on_connection_error -v` |
| 126 | All NaN on timeout | `requests.Timeout` → all-NaN scores | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_all_nan_on_timeout -v` |
| 127 | All NaN on HTTP error | HTTP 500 → all-NaN scores | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_all_nan_on_http_error -v` |
| 307 | A stop is not swallowed | REINVENT relays a stop by raising `StageInterrupted*` from a signal handler, which surfaces from inside this component's socket read. Those are plain `Exception`s, so the broad failure handler has to let them through — catching one turns a stop into a batch of NaN, the RL loop runs on to the next step, and the job is killed with no checkpoint to resume from | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_stop_request_is_not_swallowed -v` |
| 308 | Interrupt names match | The exception names resolved at import time are the ones REINVENT's `handler.py` actually raises, so a rename upstream cannot silently restore the swallow | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_stop_request_exceptions_match_reinvents -v` |
| 128 | HTTP payload structure | POST body has `{smiles: [...], params: {...}}` with correct timeout | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_http_call_payload -v` |
| 129 | No artifacts in response | `last_artifacts` stays `None` when response omits `artifacts` | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_last_artifacts_none_when_response_has_no_artifacts -v` |
| 130 | No metadata in response | `metadata` is `None` when response omits it | `pytest tests/test_external_scoring.py::TestExternalRESTScorer::test_metadata_none_when_absent -v` |

#### list_external_services

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 131 | Healthy service | `GET /health` returns `{"status": "ok"}` → status `"ok"` | `pytest tests/test_external_scoring.py::TestListExternalServices::test_healthy_service -v` |
| 132 | Unhealthy service | HTTP error on health check → status `"unhealthy"` | `pytest tests/test_external_scoring.py::TestListExternalServices::test_unhealthy_service -v` |
| 133 | Unreachable service | `ConnectionError` → status `"unreachable"` | `pytest tests/test_external_scoring.py::TestListExternalServices::test_unreachable_service -v` |
| 134 | Multiple services | Mixed healthy/unreachable correctly reported per service | `pytest tests/test_external_scoring.py::TestListExternalServices::test_multiple_services -v` |
| 135 | No services configured | Empty config → empty result, no HTTP calls | `pytest tests/test_external_scoring.py::TestListExternalServices::test_no_services_configured -v` |

#### Artifact Injection (`_ArtifactInjectingWrapper`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 136 | Injects structure paths | Upstream `last_artifacts.structures` injected into dependent's `request_params` | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_injects_structure_paths -v` |
| 137 | Injects protein PDB | `protein_pdb_path` from TTP target injected into dependent | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_injects_protein_pdb_from_ttp_target -v` |
| 138 | No artifacts → delegates | Upstream has no artifacts; dependent still called (may return NaN) | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_delegates_when_no_artifacts -v` |
| 139 | No structures key | Artifacts dict present but missing `structures` → no injection | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_delegates_when_artifacts_have_no_structures -v` |
| 140 | Attribute forwarding | `service`, `timeout` etc. forwarded to wrapped instance via `__getattr__` | `pytest tests/test_external_scoring.py::TestArtifactInjectingWrapper::test_getattr_forwards_to_wrapped -v` |

#### PipelineAwareScorer

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 141 | No dependencies skips wiring | Scorer created without replacing any component instances | `pytest tests/test_external_scoring.py::TestPipelineAwareScorer::test_no_dependencies_skips_wiring -v` |
| 142 | Wires dependent components | PoseBusters instance replaced with `_ArtifactInjectingWrapper` | `pytest tests/test_external_scoring.py::TestPipelineAwareScorer::test_wires_dependent_components -v` |
| 143 | compute_results delegates | `compute_results` calls through to underlying Scorer | `pytest tests/test_external_scoring.py::TestPipelineAwareScorer::test_compute_results_delegates -v` |
| 144 | Missing upstream warns | `depends_on` referencing nonexistent component logs warning | `pytest tests/test_external_scoring.py::TestPipelineAwareScorer::test_missing_upstream_logs_warning -v` |

#### NaN Propagation

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 145 | Service failure → all NaN | Connection error produces all-NaN array without crashing | `pytest tests/test_external_scoring.py::TestNaNPropagation::test_service_failure_produces_all_nan -v` |
| 146 | Partial null → partial NaN | Mix of scores and `null` → mix of floats and NaN | `pytest tests/test_external_scoring.py::TestNaNPropagation::test_partial_null_produces_partial_nan -v` |
| 147 | _format_results handles NaN | NaN total scores → `n_valid` excludes NaN molecules | `pytest tests/test_external_scoring.py::TestNaNPropagation::test_format_results_handles_nan_total_scores -v` |
| 148 | Empty artifacts → delegates | Wrapper still delegates to downstream even when upstream has no artifacts | `pytest tests/test_external_scoring.py::TestNaNPropagation::test_wrapper_delegates_even_with_empty_upstream_artifacts -v` |

### Optimization & Job Management (`tests/test_optimization.py`)

Tests for Phase 4: background job manager, TOML config generation, score trajectory parsing, convergence diagnostics, transfer learning, and MCP tool routing. All subprocess calls are mocked.

**Dependencies**: `pandas`, `numpy`

#### TOML Value Serialization

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 149 | Bool true/false | `_toml_bool` produces `"true"` / `"false"` | `pytest tests/test_optimization.py::TestTomlValueSerialization -v` |
| 150 | Int, float, string | Correct TOML literals for basic types | same class |
| 151 | String with quotes | Backslash-escaped inner quotes | same class |
| 152 | Lists | `[1, 2, 3]` and `["a", "b"]` serialization | same class |
| 153 | Nested lists | `[["CCO"]]` serialization | same class |

#### TOML Config Generation (Staged Learning)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 154 | Minimal config | `run_type`, `prior_file`, `agent_file`, `max_steps`, aggregation, components | `pytest tests/test_optimization.py::TestTOMLGeneration -v` |
| 155 | Agent file defaults | Agent file matches prior when not specified | same class |
| 156 | Checkpoint resume | `use_checkpoint = true` and agent_file points to checkpoint | same class |
| 157 | Full config with externals | `ExternalRESTScorer` entries with service params | same class |
| 158 | Transforms in TOML | `double_sigmoid` and `reverse_sigmoid` transforms | same class |
| 159 | Constraints in TOML | `CustomAlerts` with SMARTS params | same class |
| 160 | Diversity filter | Filter type, bucket size, minscore | same class |
| 161 | Learning strategy | DAP sigma and learning rate | same class |
| 162 | Device setting | Custom device string | same class |

#### TOML Config Generation (Transfer Learning)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 163 | Minimal TL config | `run_type = "transfer_learning"`, input/output model, SMILES file | `pytest tests/test_optimization.py::TestTLTOMLGeneration -v` |
| 164 | Custom TL parameters | num_epochs, batch_size, save_every_n_epochs, device | same class |
| 165 | With validation file | `validation_smiles_file` present when provided | same class |
| 166 | No validation file | `validation_smiles_file` absent when not provided | same class |
| 167 | No RL sections | TL config has no `[[stage]]`, `[learning_strategy]`, `[diversity_filter]` | same class |

#### Model Resolution

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 168 | File path unchanged | Absolute and relative paths returned as-is | `pytest tests/test_optimization.py::TestModelResolution -v` |
| 169 | Registry key fallback | `.reinvent` returned as-is when REINVENT4 unavailable | same class |

#### Safe Float Helper

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 170 | Normal float | `0.5` → `0.5` | `pytest tests/test_optimization.py::TestSafeFloat -v` |
| 171 | NaN → None | `float("nan")` → `None` | same class |
| 172 | None → None | `None` → `None` | same class |
| 173 | String number | `"0.75"` → `0.75` | same class |
| 174 | Bad string | `"not_a_number"` → `None` | same class |

#### Per-Component Stats

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 175 | Basic stats | Mean, max, min, mean_raw computed correctly | `pytest tests/test_optimization.py::TestPerComponentStats -v` |
| 176 | Skips non-float columns | Text columns excluded | same class |

#### Step Trajectory

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 177 | Basic trajectory | Per-step mean/max scores, molecule counts | `pytest tests/test_optimization.py::TestStepTrajectory -v` |
| 178 | None when no columns | Returns None for missing columns | same class |
| 179 | Sampling large trajectory | 200 steps sampled to 20 points | same class |

#### Per-Component Trajectory

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 180 | Basic per-component | Step-by-step mean per component (QED, MW) | `pytest tests/test_optimization.py::TestPerComponentTrajectory -v` |
| 181 | Skips non-score columns | SMILES, total_score excluded | same class |
| 182 | None when no step col | Returns None | same class |
| 183 | Sampling large trajectory | 100 steps sampled to 10 points | same class |

#### Component Health

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 184 | Healthy components | NaN rate 0%, status `"healthy"` | `pytest tests/test_optimization.py::TestComponentHealth -v` |
| 185 | Failing component (high NaN) | NaN rate 80% → status `"failing"` | same class |
| 186 | Degraded component | NaN rate 40% → status `"degraded"` | same class |
| 187 | Skips raw columns | `QED (raw)` excluded from health report | same class |
| 188 | High zero rate degraded | 80% zero scores → `"degraded"` | same class |

#### Convergence Diagnostics

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 189 | Early stage (few steps) | `convergence_status = "early"` with < 3 steps | `pytest tests/test_optimization.py::TestConvergenceDiagnostics -v` |
| 190 | Improving trajectory | Upward trend → `convergence_status = "improving"`, positive improvement_rate | same class |
| 191 | Plateau detection | Flat tail → `plateau_detected = true`, `convergence_status = "plateau"` | same class |
| 192 | Diverging trajectory | Declining scores → `diverging = true`, `convergence_status = "diverging"` | same class |
| 193 | Completed job | `convergence_status = "completed"` for COMPLETED/STOPPED jobs | same class |
| 194 | Trend computed | `early_mean < recent_mean`, positive `overall_improvement` | same class |
| 195 | Progress percentage | 50 of 100 steps → `progress_pct = 50.0` | same class |
| 196 | Component health included | Healthy QED + failing Boltz detected | same class |
| 197 | Recent validity rate | Validity rate computed over recent window | same class |
| 198 | None when no total col | Returns None | same class |
| 199 | None when empty df | Returns None | same class |

#### Warnings and Recommendations

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 200 | Plateau warning | `type = "score_plateau"` generated | `pytest tests/test_optimization.py::TestWarningsAndRecommendations -v` |
| 201 | Diverging warning | `type = "diverging"`, severity `"high"` | same class |
| 202 | Component failure warning | `type = "component_failure"` with component name | same class |
| 203 | Low validity warning | `type = "low_validity"`, severity based on rate | same class |
| 204 | Plateau recommendation | Suggests stopping and adjusting TTP | same class |
| 205 | Diverging recommendation | Suggests lowering learning rate / increasing sigma | same class |
| 206 | Failing component recommendation | Lists failing components, suggests checking services | same class |
| 207 | Improving near end | Suggests letting the run finish | same class |
| 208 | Early recommendation | Suggests continuing to monitor | same class |

#### Integration: get_status with Diagnostics

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 209 | Status includes diagnostics | Full `get_status` includes `diagnostics` with convergence, health, trend, progress | `pytest tests/test_optimization.py::TestGetStatusWithDiagnostics -v` |
| 210 | Per-component trajectory | `per_component_trajectory` with named components | same class |
| 211 | Early run no crash | Status with 2 steps returns diagnostics without error | same class |
| 212 | Failing component detection | Boltz with 87.5% NaN flagged as `"failing"` in diagnostics | same class |

#### JobInfo

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 213 | CSV path | `results_1.csv` in working dir | `pytest tests/test_optimization.py::TestJobInfo -v` |
| 214 | Checkpoint path | `stage.chkpt` in working dir | same class |
| 215 | to_dict | Correct serialization of all fields | same class |

#### JobManager Lifecycle

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 216 | Create and start job | TOML and metadata files written, correct TOML content | `pytest tests/test_optimization.py::TestJobManagerLifecycle -v` |
| 217 | Max concurrent enforced | Exceeding limit returns error with running job IDs | same class |
| 218 | Unknown job error | `get_status("nonexistent")` returns error | same class |
| 219 | Status with CSV | Trajectory parsed: step count, molecule count, validity rate | same class |
| 220 | List jobs | Returns all tracked jobs | same class |

#### JobManager Stop

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 221 | Stop non-running job | Returns error with current state | `pytest tests/test_optimization.py::TestJobManagerStop -v` |
| 309 | Stop asks with SIGUSR1 | SIGUSR1 is the only signal REINVENT treats as a *controlled* stop: it checkpoints, logs why it is terminating, and exits 0. Every other signal checkpoints but leaves the exception uncaught, so the job looks like it crashed | same class |
| 310 | Grace spans a step | The wait is `jobs.stop_grace_seconds`, not the old hardcoded 30 s. REINVENT can only act on the stop once it abandons the scoring call it is in, and a cold Boltz batch has measured 9.5 minutes | same class |
| 311 | Stop still escalates | A job that never checkpoints gets SIGTERM and then SIGKILL, and the result says no checkpoint was written and which knob to turn | same class |
| 312 | `checkpoint_on_stop` is read | With it off, the job is terminated immediately rather than waiting a step for a checkpoint nobody asked for. The flag was dead config until this | same class |
| 313 | First-step stop is not a failure | REINVENT has no agent to save until a step has updated the weights, so a job stopped inside its first scoring call reports that rather than blaming the grace period — which would send an operator chasing a timeout that never happened | same class |

#### JobManager Top Candidates

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 222 | Top candidates | Ranked by score, deduplicated, with component breakdowns | `pytest tests/test_optimization.py::TestJobManagerTopCandidates -v` |
| 223 | With min_score | Only candidates above threshold returned | same class |
| 224 | No CSV returns error | Error when results CSV doesn't exist | same class |

#### JobManager Checkpoint Resume

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 225 | Resume from checkpoint | `use_checkpoint = true` and agent_file set in TOML | `pytest tests/test_optimization.py::TestJobManagerResume -v` |
| 226 | Nonexistent checkpoint | Returns error | same class |

#### JobManager Metadata Persistence

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 227 | Metadata written on create | `job_metadata.json` exists with correct fields | `pytest tests/test_optimization.py::TestJobMetadataPersistence -v` |
| 228 | Restore jobs on init | Jobs restored from disk metadata on server restart | same class |
| 229 | Running → failed on restore | Jobs running at server crash time marked as failed | same class |

#### Tools: run_optimization

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 230 | Run with stored TTP | TTP resolved, scorer composed, job launched | `pytest tests/test_optimization.py::TestRunOptimizationTool -v` |
| 231 | Bad TTP returns error | Nonexistent TTP ID → error | same class |
| 232 | Empty scorer config | TTP with no components → error | same class |

#### Tools: stop_optimization

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 233 | Stop delegates to manager | Calls `manager.stop_job()` | `pytest tests/test_optimization.py::TestStopOptimizationTool -v` |

#### Tools: get_optimization_status

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 234 | Status delegates to manager | Calls `manager.get_status()` | `pytest tests/test_optimization.py::TestGetOptimizationStatusTool -v` |

#### Tools: get_top_candidates

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 235 | From job_id | Delegates to manager | `pytest tests/test_optimization.py::TestGetTopCandidatesTool -v` |
| 236 | From csv_path | Parses CSV directly with component breakdowns | same class |
| 237 | Both job_id and csv_path | Returns error (ambiguous) | same class |

#### Tools: read_results

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 238 | Read results | n_rows, n_unique_smiles, validity_rate, total_score_stats | `pytest tests/test_optimization.py::TestReadResultsTool -v` |
| 239 | Missing CSV | Returns error | same class |

#### Transfer Learning: JobManager

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 240 | Create TL job | SMILES file and TOML written, `run_type = "transfer_learning"` | `pytest tests/test_optimization.py::TestJobManagerTL -v` |
| 241 | TL from file | SMILES file path embedded in TOML | same class |
| 242 | TL requires SMILES | Error when neither `smiles` nor `smiles_file` provided | same class |
| 243 | TL rejects both inputs | Error when both `smiles` and `smiles_file` provided | same class |
| 244 | TL with validation SMILES | Validation file written and referenced in TOML | same class |
| 245 | TL max concurrent enforced | Respects `max_concurrent_jobs` limit. **Failed until 2026-10-08**: the cap was dead config and starting a job stopped every running one instead | same class |
| 246 | TL job to_dict | `job_type = "transfer_learning"`, `num_epochs`, `output_model_file` present | same class |
| 247 | TL metadata persistence | TL job restored correctly from disk on restart | same class |

#### Job Concurrency Slots

`max_concurrent_jobs` was defined in config and never read: `create_and_start`
stopped every running job first, so one RL run was the hard ceiling. An RL step
is mostly spent waiting on the scoring HTTP call, and placement/queueing is now
the boltz gateway's job, so several runs share a GPU.

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 314 | Jobs run concurrently | A second job starts while the first keeps running, untouched and un-checkpointed | `pytest tests/test_optimization.py::TestJobManagerLifecycle -v` |
| 315 | Cap enforced | The job past the limit is refused with "Maximum concurrent jobs", for RL and TL alike — they share the GPU so they share the slots | same class |
| 316 | Queued holds a slot | A job registered but not yet reporting `running` still counts, closing the window where two callers both pass the check | same class |
| 317 | Atomic claim | Four callers racing on two slots: exactly two start. The count and the registration share one lock acquisition | same class |
| 318 | Slots free up | A completed, failed, or stopped job releases its slot | same class |
| 319 | Refusal is actionable | The error lists the active job ids to stop, and names the setting to raise | same class |
| 320 | No orphan on refusal | A refused job leaves no working directory behind | same class |
| 321 | Restart frees slots | Both `queued` and `running` jobs are reset on restore, so an interrupted job cannot hold a slot forever | same class |

#### Tools: run_transfer_learning

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 248 | Run with SMILES list | Delegates to `manager.create_and_start_tl` | `pytest tests/test_optimization.py::TestRunTransferLearningTool -v` |
| 249 | Run with SMILES file | `smiles_file` forwarded, `smiles` is None | same class |
| 250 | No input returns error | Error when no SMILES provided | same class |

#### get_top_candidates: Component Breakdowns (CSV mode)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 251 | CSV with components | Per-component transformed + raw scores in output | `pytest tests/test_optimization.py::TestGetTopCandidatesComponentBreakdown -v` |
| 252 | CSV without raw columns | Transformed scores only, no raw key | same class |

### Scoring Latency Harvest (`tests/test_scoring_latency.py`)

Tests for `k8s/scripts/analyze-scoring-latency.py`, which reads
`external_scoring.jsonl` back to recover how long a Boltz batch takes and how
often molecules repeat. Records are synthesized in the shape
`ExternalRESTScorer._log_call` writes, so no cluster, log file, or GPU is
needed.

**Dependencies**: none beyond `pytest`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 257 | Record parsing | Batch size recovered from `request.smiles`; a 2xx with no `error` is usable | `pytest tests/test_scoring_latency.py::TestRecordParsing -v` |
| 258 | Failure classification | `HTTP 500` → `http_500`; a capped `request_failed` → `timeout`; missing `elapsed_seconds` is unusable | same class |
| 259 | Cache-key digest | Predict params change the digest; per-call `structure_paths` do not | same class |
| 260 | Timeouts excluded | Nine 600s calls plus nine capped 1800s timeouts leave p95 at 600s | `pytest tests/test_scoring_latency.py::TestFailureExclusion -v` |
| 261 | Latency percentiles | Linear interpolation; per-molecule latency is `elapsed_seconds / batch size` | `pytest tests/test_scoring_latency.py::TestLatencyStats -v` |
| 262 | Per-batch-size split | Separate p50 per batch size; the modal size breaks ties towards the larger batch | same class |
| 263 | Cache hit rate | Repeated SMILES under one protein sequence are hits; the same SMILES under another target is a miss | `pytest tests/test_scoring_latency.py::TestCacheHitRate -v` |
| 264 | Full cache key | Differing `recycling_steps` turns a repeat into two misses | same class |
| 265 | Fleet sizing | A 10-minute batch against a 5-minute step gives `M = 2N`; a hit rate lowers `M`; 100 steps is 16.7 h on a dedicated replica | `pytest tests/test_scoring_latency.py::TestFleetSizing -v` |
| 266 | Assumption warnings | Too-short window → shrink the task unit; a slow CPU scorer → "not negligible"; mixed targets flagged | `pytest tests/test_scoring_latency.py::TestWarnings -v` |
| 267 | CLI | `--json`, `--service`, `--since`, `--batch-size`, directory input, and unparseable lines | `pytest tests/test_scoring_latency.py::TestCLI -v` |
| 268 | No usable records | An empty or all-failed log exits 2 and prints the single-call fallback | same class |

### Boltz Kernel A/B Harness (`tests/test_boltz_kernels_ab.py`)

Tests for `k8s/scripts/ab-boltz-kernels.py`, which times one `/score` call with
and without Boltz's optimized kernels. The molecule panel is checked against
the wrapper's own validator, so neither arm can silently score a smaller batch.
`kubectl` is stubbed and no arm is run, so no cluster, service, or GPU is
needed.

**Dependencies**: `rdkit`, `pydantic` (both already required to import the wrapper)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 269 | Panel validity | 16 unique molecules, each accepted by `_validate_smiles`, none dropped by `_partition_smiles` | `pytest tests/test_boltz_kernels_ab.py::TestPanel -v` |
| 270 | BRD4 BD1 sequence | The baked-in sequence is foldable and matches the query record of `brd4_bd1.a3m` (skipped when that gitignored file is absent) | same class |
| 271 | Request shape | Carries `msa_path` and no `use_msa_server`, keeps production affinity settings, and validates as a `ScoreRequest` | `pytest tests/test_boltz_kernels_ab.py::TestPayload -v` |
| 272 | Replica arithmetic | A 10-minute batch against a 5-minute step gives `M = 2N`; counts round up | `pytest tests/test_boltz_kernels_ab.py::TestUrlsAndMath -v` |
| 273 | Deployment env | `BOLTZ_NO_KERNELS` read from the right container; an originally unset value is restored by removing it | `pytest tests/test_boltz_kernels_ab.py::TestDeploymentEnv -v` |
| 274 | CLI guards | A two-arm run without a restart mechanism, conflicting arm flags, and a batch larger than the panel are all refused | `pytest tests/test_boltz_kernels_ab.py::TestCLIGuards -v` |

### Kubernetes Manifests (`tests/test_k8s_manifests.py`)

Tests for the split manifests under `k8s/manifests/`. The six Compose
containers were one pod with loopback scorer URLs; they are now one Deployment
plus one ClusterIP Service each. These parse the YAML directly, so no cluster,
`kubectl`, or GPU is needed.

**Dependencies**: `PyYAML`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 275 | Pod split | Each of the five scorers has its own Deployment, every Deployment runs exactly one container, and `dd-agentic` carries only `dd-agent` | `pytest tests/test_k8s_manifests.py::TestPodSplit -v` |
| 276 | GPU accounting | Still two `amd.com/gpu` in total, and every GPU Deployment uses `Recreate` so a surge pod cannot ask for a third | same class |
| 277 | Boltz replica cap | `dd-agentic-boltz` stays at one replica while a plain Service would round-robin onto a busy replica (§5.2) | same class |
| 278 | Scorer Services | Each scorer has a ClusterIP Service whose selector matches its own pods and no one else's, on the container's named port | `pytest tests/test_k8s_manifests.py::TestScorerServices -v` |
| 279 | Cluster DNS URLs | No `127.0.0.1`/`localhost` anywhere in the manifests; every `service_url` resolves to a Service in the set, on the right port, unqualified so the namespace stays free | `pytest tests/test_k8s_manifests.py::TestConfigMapServiceUrls -v` |
| 280 | Node pinning | `apply.sh` patches `nodeSelector[kubernetes.io/hostname]` onto Deployments with no name or label restriction, so future workloads are pinned too | `pytest tests/test_k8s_manifests.py::TestSharedVolumePinning -v` |
| 281 | Shared volume | The `/shared` users are exactly the agent, Boltz, PoseBusters, and the interaction profiler, all mounting the one claim at the same path; ADMET and RetroDMTA mount nothing | same class |
| 282 | Never `ReadWriteOncePod` | The claim stays `ReadWriteOnce`, which is per node and admits the whole pinned fleet | same class |
| 283 | `/shared` is writable | No `/shared` mount is `readOnly`: PoseBusters extracts the ligand *beside* the PDB Boltz handed it, and a read-only mount fails that exactly like a missing file | same class |
| 284 | Kustomization | Every manifest file is a listed resource, images are remapped by tag and never pulled from a registry, and the `part-of` label `apply.sh` selects on is stamped | `pytest tests/test_k8s_manifests.py::TestKustomization -v` |
| 285 | Script targets | `ab-boltz-kernels.py` defaults to `dd-agentic-boltz`/`boltz`, and `uat-scoring.sh` still targets `dd-agentic`/`dd-agent` | `pytest tests/test_k8s_manifests.py::TestScriptTargets -v` |
| 354 | Model switch is per tier | `switch-aim-model.py` takes `--tier` and carries no single-service default. The old script restarted `dd-agentic` unconditionally, so changing the conductor's model would have restarted the supervisors and **lost whatever tasks they were holding** | same class |
| 355 | The tier table names real objects | Read from the script's own `TIERS` dict rather than grepped, so the test fails when the table drifts rather than when the prose does: every service is an AIMService, every model key is in the ConfigMap, every client is a Deployment | same class |
| 356 | Two AIM tiers exist | One AIMService each for the conductor and the supervisors, each with a stable Service selecting its own predictor — AIM generates a KServe name, so the agents need a repeatable base URL | `pytest tests/test_k8s_manifests.py::TestAimTiers -v` |
| 357 | Each tier's profile matches its role | The conductor's profile is `latency`, the supervisors' is `throughput`, and both are `tp1`. B10 was that one latency-tuned endpoint is sized for a single interactive session, which is the opposite of what N supervisors do; `tp1` is what keeps the split affordable, since a tensor-parallel conductor model would consume a node and starve Boltz | same class |
| 358 | Each client dials its own tier | The conductor reads `CONDUCTOR_AGENT_*`, the supervisors and interactive agent read `REINVENT4_AGENT_*`, the two base URLs differ, and no manifest still references the pre-split `http://dd-agentic-aim/v1` | same class |
| 286 | Handoff harness targets | Every Deployment/container pair `verify-artifact-handoff.py` execs into exists and is one of the three that mount `/shared` | same class |
| 287 | Gateway placement | `balance leastconn` plus `maxconn 1` per replica, `http-reuse never` so an idle connection cannot hold the slot, and `BOLTZ_MAX_CONCURRENCY=1` on the replicas to match | `pytest tests/test_k8s_manifests.py::TestBoltzGateway -v` |
| 288 | Gateway timeouts | `client`, `server`, and `queue` all exceed `PREDICT_TIMEOUT_SECONDS`, and a connection failure redispatches instead of becoming a NaN batch | same class |
| 289 | Gateway discovery | Replicas come from the headless Service by DNS with `init-addr none`, enough slots for every replica, and no cluster DNS address or namespace baked in | same class |
| 290 | Gateway shape | Exactly one gateway (one view of who is busy), no GPU, no `/shared`, readiness that fails with no healthy replica but liveness that does not, and 10 min grace so a live batch is not killed | same class |
| 291 | Several replicas | More than one Boltz replica is only allowed when the ConfigMap dials the gateway; GPU demand is the agent plus the replicas; no GPU Deployment surges | `pytest tests/test_k8s_manifests.py::TestPodSplit -v` |
| 292 | Store is one writer | `dd-agentic-campaign-store` is one replica with `Recreate` and one uvicorn worker, because the atomic claim relies on SQLite's write lock | `pytest tests/test_k8s_manifests.py::TestCampaignStore -v` |
| 293 | Store lease sizing | `DEFAULT_LEASE_SECONDS` exceeds the gateway's `UPSTREAM_TIMEOUT_SECONDS`, so a lease cannot expire inside a normal RL step and hand the same weight vector to a second supervisor | same class |
| 294 | Store placement | The ledger sits on the mounted `/shared` so it outlives the pod, needs no GPU, and is reachable on `store-http` 8087 with `/health` probes | same class |
| 314 | Supervisor pool shape | `dd-agentic-supervisor` ships at `replicas: 0`, runs the *agent* image with `python -m reinvent4_agent.supervisor`, needs no GPU, and has no Service because nothing dials it | `pytest tests/test_k8s_manifests.py::TestSupervisorPool -v` |
| 315 | Supervisor wiring | It dials the store's own Service and port, is named in the ledger by `metadata.name`, heartbeats well inside the store's lease, shares the agent's TTP mount, and takes the API key from the Secret | same class |
| 328 | One conductor | `replicas: 1` with `Recreate`. Two would each top the pool up to N and the campaign would run at 2N — the one way to overcommit the GPUs the ledger cannot catch, because each dispatch is individually legitimate | `pytest tests/test_k8s_manifests.py::TestConductor -v` |
| 329 | Conductor RBAC | It carries its own ServiceAccount, Role and RoleBinding, because the `deployments/scale` grant in `k8s/rbac.yaml` is bound to the `oidc:demo-lux` developer group and authorises a human's kubectl, not a pod | same class |
| 330 | Scale is least-privilege | The Role is scoped by `resourceNames` to `dd-agentic-supervisor` and grants only `get` and `patch`, so a conductor cannot scale Boltz or the runner; `list`/`watch` are withheld because they cannot be restricted by name | same class |
| 331 | Conductor is read-only on `/shared` | Every data mount is `readOnly`. A writable mount would invite it to edit a base TTP mid-campaign, silently changing what later tasks in the sweep mean | same class |
| 332 | Conductor wiring | It dials the ledger and the gateway's `/stats` on their own Services and ports, knows its namespace from `metadata.namespace`, names the Deployment it scales, needs no GPU, and stays interactive for the REPL | same class |

### Conductor (`tests/test_conductor.py`)

Tests for `reinvent4_agent/conductor.py`. `tick()` is a pure function of one
ledger read, so the whole loop — the windows, the top-up arithmetic, the
proposals, admission control and the hard stop — is covered against a fake
store with no cluster and no GPU.

**Dependencies**: `requests`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 338 | Admission control | A deep Boltz queue holds dispatch back, a shallow one lets work through, and an unreachable gateway is *unknown* rather than idle — but unknown does not block dispatch, or a silent gateway would stall the campaign | `pytest tests/test_conductor.py::TestQueuePressure -v` and `::TestAdmissionControl` |
| 339 | Top-up arithmetic | It dispatches only the shortfall; queued and claimed both count against the target, finished tasks do not; and it scales the pool to the campaign's concurrency | `pytest tests/test_conductor.py::TestDispatchWindow -v` |
| 340 | Task deadlines | Each task gets the task budget, capped by the campaign's hard stop, so no supervisor starts a run that would outlive the campaign | `pytest tests/test_conductor.py::TestDeadlines -v` |
| 341 | Windows | Draining closes the queue exactly once and dispatches nothing; the hard stop finalizes and scales the pool to 0; a finalized campaign is left alone | `pytest tests/test_conductor.py::TestWindows -v` |
| 346 | Orphaned jobs are stopped | Abandoning a task does not stop its RL job — the job lives in the runner and holds a GPU until told otherwise, which left two running after the first live wind-down. The hard stop now stops the jobs of open tasks, skips finished ones, closes the ledger *before* stopping so a supervisor cannot post against a dead campaign, and treats freeing GPUs as best-effort so an unreachable runner does not block the wind-down | same class |
| 342 | Status view | Renders the things an operator asks about — counts, who holds what, how many results are in, time left — and the supervisor pool when it is known, without printing a pool of zero when the store did not answer | `pytest tests/test_conductor.py::TestStatusView -v` |
| 343 | Pool scaler | Outside a cluster every call is a logged no-op so a laptop run still exercises the loop; inside, it sends a merge patch of only `spec.replicas` to the scale subresource, skips the write when already at the target, and reports a refused scale instead of raising | `pytest tests/test_conductor.py::TestPoolScaler -v` |
| 351 | Campaign resolution | With no id the loop attaches to the only live campaign (draining counts as live), says how to open one when there are none, and refuses to guess between two — naming them so one can be picked, because conducting the wrong campaign spends GPU on the wrong goal | `pytest tests/test_conductor.py::TestCampaignResolution -v` |
| 352 | Objective defaults | A new campaign's objectives default to every one the base TTP declares, across both sections and deduplicated. Typing them by hand is the easiest way to get a campaign wrong: an unknown objective fails every task referencing it, and surfaces minutes later as a task failure rather than a typo | `pytest tests/test_conductor.py::TestObjectiveDefaults -v` |
| 353 | Entrypoint guards | `--start` requires a goal and a base TTP, cannot also name a campaign to attach to, and the opening flags are refused without it — quietly ignoring them would look like the campaign had been configured | `pytest tests/test_conductor.py::TestEntrypointGuards -v` |
| 365 | Conductor-directed mode | When an agent is dispatching the tasks itself, the loop still scales the pool and keeps the windows but proposes nothing of its own — two things filling the same queue would spend the pool on work nobody asked for | `pytest tests/test_conductor.py::TestConductorDirectedMode -v` |
| 366 | A hand-closed campaign finalizes when it empties | A campaign closed by hand has no hard stop to fire, so without this the pool stayed held after the last task came back | `pytest tests/test_conductor.py::TestWindows -v` |

### Supervisor (`tests/test_supervisor.py`)

Tests for `reinvent4_agent/supervisor.py`. The campaign run is injected into
`Supervisor`, so everything around it — leases, failures, deadlines, shutdown,
and reading the achieved profile out of a results CSV — is covered with no LLM,
no MCP server and no GPU. The loop drives its own event loop, since the repo
has no `pytest-asyncio`.

**Dependencies**: `requests`, `pydantic`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 316 | Weight overrides | Weights and transforms are replaced by name in both `scoring.objectives` and `pre_scoring.components`, and the base dict is never mutated — N supervisors read the same base out of a shared store | `pytest tests/test_supervisor.py::TestApplyOverrides -v` |
| 317 | An unknown objective fails | A typo is an error, not a no-op. Ignoring it would have the task report a profile for a weight vector it never ran, and the conductor would draw conclusions from a point that was never tried | same class |
| 318 | TTP variants | The variant is named after the task, carries the overridden weights, starts at version 1, leaves the base alone, and two tasks cannot collide in the shared store | `pytest tests/test_supervisor.py::TestPrepareTtpVariant -v` |
| 347 | A retried task is runnable | `create_ttp` refuses an id that exists, so the first live retry died with "TTP ... already exists" before it started, making every retry impossible. The variant's content is a function of the task, so it is reused | same class |
| 348 | Job discovery by observation | The supervisor finds its job by the task's unique TTP id, not from the Driver's handoff. A session ending `incomplete` produces no handoff, and the first live run failed a task with "the Driver finished without starting an optimization job" while that job was running fine | `pytest tests/test_supervisor.py::TestJobDiscovery -v` |
| 349 | The Driver cannot end its own run | `stop_optimization` is withheld from a supervisor's Driver, along with the TTP writers and the fleet-steering tools. The first live run had the Driver poll 13 times in six minutes, decide the job had stalled, and stop it inside its first step — where there is nothing to checkpoint and no candidates | `pytest tests/test_supervisor.py::TestDriverToolBoundary -v` |
| 350 | The Driver is told the pacing | The goal quotes how long a step takes and that a quiet run is normal, so its patience has a basis — without a number it cannot tell a slow step from a broken one | `pytest tests/test_supervisor.py::TestBuildGoal -v` |
| 319 | Objective profile | The profile is the best *single molecule's* per-objective scores, not the best value seen per objective — the latter describes a compound nobody made | `pytest tests/test_supervisor.py::TestObjectiveProfile -v` |
| 320 | A failed scorer is not a number | A `null` component is left out of the profile rather than becoming a score | same class |
| 321 | The Driver's goal | Names the prepared TTP instead of restating the weights, passes the step budget as a ceiling, and tells a validation-tier task not to use the cache | `pytest tests/test_supervisor.py::TestBuildGoal -v` |
| 322 | Task lifecycle | A successful task posts its profile; a failing one is reported and requeued; an empty queue is not work | `pytest tests/test_supervisor.py::TestSupervisorLoop -v` |
| 323 | A lost lease posts nothing | A task taken from us is neither completed nor failed — either would overwrite the new holder's work | same class |
| 324 | Heartbeating | The lease is renewed while the task runs and stops when the task does; a store outage mid-task is logged and retried rather than ending the run, because the work in flight is worth more than the ledger entry | same class |
| 325 | Shutdown and outages | A drain wakes the empty-queue poll immediately instead of waiting it out, and an unreachable store backs off and retries instead of taking the pod down | same class |
| 326 | Deadlines | A future deadline is the time remaining, a past one is zero rather than negative (a negative timeout would raise instead of stopping the task), a naive timestamp is read as UTC, and an unparseable one is ignored | `pytest tests/test_supervisor.py::TestDeadlines -v` |
| 327 | Store client | 204 means nothing to claim, 409 is a lost lease rather than an outage (different problems: one means stop, the other means retry), connection and server errors are outages, and the owner is sent on every call | `pytest tests/test_supervisor.py::TestCampaignStoreClient -v` |
| 359 | Molecules without scores are reported | A run that made molecules but whose per-objective scores could not be read posts them with a note in the summary, instead of failing and spending the GPU-hour again | `pytest tests/test_supervisor.py::TestHarvestedProfileIsScopedToTheCampaign -v` |
| 360 | A run with no molecules fails | The task is handed back for a retry rather than posted as a result that found nothing | same class |
| 361 | A non-design sub-agent cannot start a run | Research, analysis, retrosynthesis and question tasks are refused the job-starting tools: nothing would be watching that run or harvesting it, so it would hold a GPU past the task that started it. It may still write a TTP, which a later design task can use | `pytest tests/test_supervisor.py::TestSubAgentToolBoundary -v` |
| 362 | The brief is the whole task | Intent, context and deliverable reach the sub-agent verbatim, a non-design one is told plainly that it cannot start a run (a tool refused without a reason reads as a broken toolchain), and the campaign goal comes along so the task is not read out of context | `pytest tests/test_supervisor.py::TestSubAgentBrief -v` |
| 363 | A design task reports in prose | The report leads with the TTP, the profile and how many molecules came back, keeps the Driver's own account, and strips the Reviewer handoff JSON — the conductor needs what happened, not a scoreboard it already has | `pytest tests/test_supervisor.py::TestDesignReport -v` |
| 364 | Jobs are attributed to the task that started them | Job ids are collected from the sub-agent's own tool results rather than the shared job table, which has no column saying which task launched what | `pytest tests/test_supervisor.py::TestJobIdCollection -v` |

### Campaign Store (`tests/test_campaign_store.py`)

Tests for `services/campaign-store/app.py`: the fleet control-plane ledger —
campaigns, the task queue, leases, results, and the event feed. The module is
imported per test against a `tmp_path` SQLite file, so there is no cluster, no
GPU, and no shared state between tests.

The case worth reading is #297: it is the one that costs real GPU time when it
regresses.

**Dependencies**: `fastapi`, `pydantic` (SQLite is in the standard library)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 295 | Campaign lifecycle | Create, read back, 404 on unknown, neither a base TTP nor objectives required, and deadlines derived from the dispatch and grace windows | `pytest tests/test_campaign_store.py::TestCampaigns -v` |
| 296 | Close vs finalize | Closing shuts the queue to new tasks but leaves queued work claimable through the drain window; finalizing abandons whatever is still outstanding instead of leaving it queued forever | same class |
| 297 | Concurrent claims | Eight supervisors claiming at once get eight *distinct* tasks. Verified to fail without `BEGIN IMMEDIATE`, where the same run handed out only 2 distinct tasks — four pairs of supervisors would have run identical weight vectors on separate GPUs | `pytest tests/test_campaign_store.py::TestClaims -v` |
| 298 | Claim order and scope | Oldest task first, the spec the supervisor needs comes with it, a task is handed out only once, and `campaign_id` scopes the claim | same class |
| 299 | Lease ownership | Heartbeat extends the lease, a stranger's heartbeat is refused, and a supervisor that lost its lease cannot post a result over the new holder's work | same class |
| 300 | Lease expiry | An expired lease is requeued for the next supervisor, and a task that keeps losing its lease is abandoned rather than cycling through the pool forever | same class |
| 302 | Failures | A failure is requeued by default, can be marked terminal, and stops being requeued at the attempt limit | `pytest tests/test_campaign_store.py::TestResults -v` |
| 303 | Event feed | Events record the task lifecycle in order and can be polled incrementally by last event id, with the cursor held when nothing is new | `pytest tests/test_campaign_store.py::TestEventsAndStatus -v` |
| 304 | Status aggregate | Task counts, in-flight supervisors with their weight vectors and lease remaining, the result count, and a window phase that follows the clock — so a conductor that missed its own deadline still sees that it should be draining | same class |
| 305 | Status sweeps the dead | A supervisor whose lease expired is out of `in_flight` and its task back in `queued`, so in-flight means alive rather than "claimed by something, once" | same class |
| 306 | Durability | The ledger survives a restart and specs round-trip as JSON, because the conductor rebuilds its loop by reading it back | `pytest tests/test_campaign_store.py::TestDurability -v` |
| 344 | Concurrency is in the ledger | It can be changed and dropped to zero, and survives a conductor restart, so the pool the conductor scales to is always reconcilable | `pytest tests/test_campaign_store.py::TestCampaigns -v` |
| 345 | Cancel overrides the holder | Deliberately not owner-checked — this is the conductor overriding a supervisor. The holder finds out on its next heartbeat, which already treats a lost lease as "stop and do not post a result"; a cancelled task is not handed out again, and a *finished* one cannot be cancelled because its result is already in | `pytest tests/test_campaign_store.py::TestClaims -v` |

### Dispatching and reports (`tests/test_campaign_tools.py`, `tests/test_web_autonomy_watch.py`, `tests/test_web_agents.py`)

The two ends of a conducted campaign: the tools the conducting agent gives
work with and reads answers from, and the gateway loop that wakes it when a
sub-agent reports. Both run against fake stores and fake agents — no ledger,
no cluster, no LLM.

**Dependencies**: `fastapi`, `pydantic`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 367 | Reading what a sub-agent sent back | The report comes through whole, the molecules are left out (they are the one thing that would not fit in the conductor's context), one task can be asked for by id, an unknown id says so, a result posted without a report is named rather than silently missing, and a store outage is an error instead of an exception mid-turn | `pytest tests/test_campaign_tools.py::TestTaskReports -v` |
| 368 | A report wakes the conductor | A `task_done` in the ledger becomes a wake reason naming the task and how to read it; one already read is not told twice; a failure is news too, because the sub-agent holding that slot is free either way | `pytest tests/test_web_autonomy_watch.py -v -k "report or sub_agent"` |
| 369 | Either way a campaign stalls is news | Idle sub-agents with nothing queued wake the conductor, and so does queued work with no sub-agent alive to take it — what a pool that was never scaled up looks like from here. Idle with work waiting is not news, a report is told instead of the idleness it leaves behind, and a reader that does not count the pool raises no alarm | `pytest tests/test_web_autonomy_watch.py -v -k "idle or queued"` |
| 370 | A campaign is looked at with no run of our own going | The quiet backstop fires on an owned campaign as well as on running jobs: a conductor whose sub-agents are all mid-task launched nothing itself and would otherwise sit out the whole campaign | `pytest tests/test_web_autonomy_watch.py -v -k quiet` |
| 371 | The conducting agent knows it is one | The allowance names the tools a campaign is run with, and the playbook says sub-agents cannot see the conversation, cannot come back with a question, and should not sit idle while a time budget runs | `pytest tests/test_web_agents.py -v` |

### Who does the work, and for how long (`tests/test_web_agents.py`, `tests/test_agent_tool_scope.py`, `tests/test_supervisor.py`, `tests/test_web_autonomy.py`, `tests/test_campaign_tools.py`, `tests/test_web_server.py`)

The sub-agent setting decides who runs the long work: at two or more it is
the sub-agents', and the conductor plans, dispatches and reads; at one the
conductor does all of it. Auto mode is the user's to end, and the Campaign
tab needs to know which sub-agent ran a task and what each task built on.
Fake agents, fake stores — no ledger, no cluster, no LLM.

**Dependencies**: `fastapi`, `pydantic`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 381 | Two or more sub-agents make the long work theirs | At two or more the conductor is told the conversation is a campaign and the run starters are withheld from it, each with the reason; at one it works alone and the campaign tools are withheld instead — left the choice, a conductor whose campaign tool failed ran the tracks itself and reported them as the campaign's | `pytest tests/test_web_agents.py -v -k "two_or_more or working_alone"` |
| 382 | A withheld tool is refused with what to do instead | It is not offered to the model, a call made anyway gets the reason back instead of a bare refusal, and a tool handed back is callable again | `pytest tests/test_agent_tool_scope.py -v -k "withheld or handing_a_tool_back"` |
| 383 | Every sub-agent refusal says what to do instead | Over every task kind and both ways of getting a TTP, no denial reason is empty: a tool refused without one reads as a broken toolchain | `pytest tests/test_supervisor.py -v -k every_refusal` |
| 384 | Auto mode ends when the user stops it or the budget runs out | The agent saying it is done is not the end: a check turn, then a look again after the longer idle wait. There is no cap on the number of turns, and a budget that runs out ends the mode and says so | `pytest tests/test_web_autonomy.py -v -k "done_does_not_end or looking_again or nothing_but_the_budget or budget_that_runs_out"` |
| 385 | A failed turn is picked up again | A turn that raises is retried after 30 s, the wait doubling up to 10 min, with auto mode still on rather than handed back to the user | `pytest tests/test_web_autonomy.py -v -k "failed_turn or retry_wait"` |
| 386 | The conductor records what a task built on | `enqueue_task` passes `derived_from` through to the ledger, and a first task carries none — the lines between the bubbles on the Campaign tab | `pytest tests/test_campaign_tools.py::TestEnqueueTask -v` |
| 387 | A finished task still names its sub-agent | The ledger clears the lease when a task is done; the gateway reads who ran it from the claim events, page by page, and a task claimed again belongs to the last claim | `pytest tests/test_web_server.py -v -k "still_names or claimed_again"` |

### What the tools can see, and what survives a reload (`tests/test_mcp_client_env.py`, `tests/test_backend_history.py`)

Two things the agent cannot work without and neither one shows up as an
error: the campaign tools run inside the MCP subprocess, which only sees the
environment the client forwards, and a conversation is saved and restored by
calling the backend. Both failed silently in the first live campaign — the
pool stayed at zero with tasks queued, and every reload lost the thread. No
subprocess is started and no endpoint is contacted.

**Dependencies**: `openai`, `anthropic`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 388 | Everything the pool reads is forwarded | Parsed out of `pool.py` rather than listed by hand, because the failure is a variable someone adds later: without `SUPERVISOR_POOL_CONTROL` the conducting loop found no relay, settled for the null scaler, and held a pool of zero while the tasks sat queued — a campaign that looks alive and never starts | `pytest tests/test_mcp_client_env.py -v` |
| 389 | A conversation survives a reload | Both backends hand their history out and take it back verbatim, tool calls still paired with their results, the system prompt not travelling with it, and an unanswered call healed on the way back in. A backend that does not implement the pair fails the test rather than the gateway at runtime | `pytest tests/test_backend_history.py -v` |
| 390 | A sub-agent's conversation can be read back | The gateway finds a task's transcript by its id — the research agent's or the design driver's, never the reviewer's, the newest when a task ran twice — and hands it out with the local zone on its times, long text cut, broken lines skipped and an id that could name another path refused | `pytest tests/test_subagent_transcript.py -v` |
| 391 | A prior the image lacks is refused up front | Asking for a prior the registry names but the image does not ship fails before the run, with the priors that are installed, instead of REINVENT exiting with code 1 a minute in | `pytest tests/test_optimization.py -v -k ModelResolution` |
| 392 | A message reaches a sub-agent at work | What the gateway leaves in a task's inbox is read once and in order, goes into the conversation after a round of tool results — never inside a call's pairing — marked as interjected in the transcript, and shows as waiting until it has been handed over | `pytest tests/test_task_inbox.py -v` |
| 393 | A failing call resent unchanged ends the turn | A tool that answers only `{"error": …}` counts as failed while a status that carries an error beside its data does not; each result keeps an excerpt in the transcript; the same call failing the same way is answered with a warning from the third time and ends the turn with `RepeatedToolCall` at the sixth, while a call whose arguments changed or that succeeds resets the count | `pytest tests/test_tool_loops.py -v` |
| 394 | A sub-agent going round in circles is visible | In its conversation each call shows what came back, a call resent word for word with the same answer is one row with its count, the agent's stop is the turn's last line, and a running task whose last three calls failed the same way is marked stuck on its sub-agent box, its conversation and the iteration graph | `cd services/ui/frontend && npx vitest run src/lib/subagentChat.test.ts` |
| 395 | A TTP REINVENT could not start from is refused when saved | `create_ttp` and `update_ttp` refuse a transform missing a bound REINVENT needs, naming its objective, and a GroupCount objective with more than one SMARTS pattern; a single GroupCount pattern reaches the TOML as a string while alert filters keep their list | `pytest tests/test_ttp_for_reinvent.py -v` |
| 396 | A fleet left running stays readable | A supervisor is named after its Slurm job as well as its node, so two on one node are two in the ledger; a campaign opened without a window gets the default one, so the clock closes it instead of it dispatching forever; and a pattern parameter is checked into the shape REINVENT4 collects it in — one for a counted group, a list for an alert filter, either way of writing it accepted — without needing REINVENT4 importable | `pytest tests/test_fleet_housekeeping.py -v` |
| 397 | The iteration graph's lines do not run over each other | Tasks worked at the same time get columns of their own, a column is reused once its task has reported or merged, a task forked from the row another merged into may take that one's column, and every step sits in its task's column | `cd services/ui/frontend && npx vitest run src/lib/activity.test.ts` |
| 398 | A campaign nobody conducts gives its queue up | Every conductor tick marks its campaign conducted; a supervisor with no campaign of its own claims only from campaigns conducted within `CAMPAIGN_CONDUCTED_WITHIN_SECONDS` (300 s) or never conducted, so a campaign left open by a session that ended no longer holds a sub-agent | `pytest tests/test_campaign_store.py tests/test_conductor.py -v` |
| 399 | A report wakes the conductor without autonomy | While a campaign of the conversation is open, news from it wakes the conductor for one turn even with autonomy off, and the report line asks it to give the freed sub-agent its next task; a closed campaign's news waits for the user | `pytest tests/test_web_autonomy_watch.py -v` |
| 400 | A sub-agent's page is one task | A sub-agent's Conversation, Molecules and Analysis are the task it holds now, or a finished task of its picked from the trajectory, never everything it did; once it has reported the page is empty and points at its last task; the runs of a task include the ones only its result names | `cd services/ui/frontend && npx vitest run src/lib/fleet.test.ts` |
| 401 | "Answer" is never a title | A heading that only says "Answer" or "Final answer" does not title the conversation or the answer card | `pytest tests/test_web_artifacts.py -v` and `cd services/ui/frontend && npx vitest run src/lib/answer.test.ts src/lib/report.test.ts` |
| 402 | Stopping a conversation stops everything it started | `POST /api/sessions/{id}/stop` cancels the turn, finalizes each of its open campaigns (orphaned jobs stopped, pool scaled to zero) and stops its launched runs that are still active, reporting what it stopped and what it could not | `pytest tests/test_conductor.py tests/test_web_server.py -v` |
| 403 | The structure basket | The basket lists the conversation's structures, then the target's PDB entries it has not opened; ticking adds one at the end and unticking takes it out, but the last stays; each shown structure gets the colour of its place | `cd services/ui/frontend && npx vitest run src/lib/basket.test.ts` |
| 404 | A tail packed elsewhere does not spoil a superposition | The CA fit drops the pairs furthest apart in steps down to 2 Å, so a long tail in another place (EGFR's in 1XKK against 1M17) leaves the core fit instead of dragging it to 11 Å | `cd services/ui/frontend && npx vitest run src/lib/superpose.test.ts` |
| 448 | One agent working alone still keeps a campaign | A lone agent kept the run starters and lost every campaign tool, so a conversation with no sub-agents had no ledger, no workstreams, no task history and nothing in the Campaign tab: the split is now between dispatching work and keeping a campaign, and only the first needs somebody to dispatch to. It opens a campaign, sets its plan, records each experiment with `start_own_task` before it runs and `finish_own_task` after, and is told to, since a tool it is never reminded of is a tool it does not reach for. Its own tasks are taken by name rather than off the head of the queue — the store grew a claim for that — and hold no lease, since there is no pool member to expire; the UI labels them "C" and gives them no S-number. The molecule view no longer shows only what a sub-agent posted: every run of the conversation is folded in beside the reports, one entry per molecule, the reported one preferred where both have it | `pytest tests/test_campaign_store.py::TestTakingOneNamedTask tests/test_campaign_tools.py::TestWorkTheConductorDoesItself tests/test_web_agents.py -k "alone or Named or Itself" -v` and `cd services/ui/frontend && npx vitest run src/lib/fleet.test.ts` |
| 447 | The complexes of one call are one row, not six | Preparation writes a directory rather than a file — the complex and the receptor, but also the waters it kept, the ions, a hydrogen-free receptor for docking and one file per retained ligand instance including the sulfates and glycerols — and every path became a card, so preparing four entries buried the answer under rows of boxes. The chat keeps to one row: crystallisation chemistry and preparation by-products go straight into the "…" card, as does anything that does not fit, which lists all of them and opens any in the viewer. Nothing is filtered out of the Structures tab, and the kind label ("Prepared") is dropped in the row, where every card is of one call and it only costs the width the names need | `cd services/ui/frontend && npx vitest run src/lib/structureNoise.test.ts` |
| 446 | A run that has ended is not listed as running | The job list polls a relayed run as well as a local one, rather than trusting the monitor thread that is the only writer of its terminal state: a manager holding the run without one listed it as running for the life of the process, while `get_optimization_status`, which does poll, said it had finished. Bringing the run up to date takes the manager's lock, which the list already holds, so the lock is reentrant. One read that raises no longer ends the monitor, and a run Slurm has stopped accounting for finishes with that as its reason instead of being followed forever | `pytest tests/test_optimization.py -v -k RelayedRun` |
| 445 | A gateway that cannot be read says so, rather than showing three empty panels | Summary, Results and Analysis all render from one job scope, which waits on both the job list and the conversation list. A read of the conversation list that came back without one counted as "not read yet" and held the scope empty for as long as the gateway was unreachable; it now counts as "the gateway does not say", which falls back to every run being the conversation's. The Summary's loading and error states test the scope's own loading value, not a value it never returns, so its error text and Retry are reachable again | `cd services/ui/frontend && npx vitest run src/hooks/useSessions.test.ts src/lib/jobScope.test.ts` |
| 444 | A dossier source that changed its API is caught here rather than reported as an outage | Every target-dossier tool wraps somebody else's HTTP API, and all four had rotted into permanent failure without raising anything: OpenTargets rejected `associatedDiseases(size:)` with a 400 because the released schema paginates through a `Pagination` object; the Human Protein Atlas answered 404 for the `{SYMBOL}-{ACCESSION}/tissue.json` page id the code built, a route HPA has never served, so expression is read from `/{ENSG}.json` and the column-selected query API instead and banded the way HPA bands it; `fetch_target_orthology` read orthologs out of UniProt's OMA cross-reference, which carries only a group fingerprint and so returned an empty list for every target, and now asks OMA itself; SureChEMBL's compound route is gone from both hosts, so a structure is matched to compound ids and those are looked up **one at a time**, several ids in one request meaning "documents claiming all of them" and matching nothing. A structure nothing claims is reported distinctly from a database that could not be asked | `pytest tests/test_target_dossier_sources.py -v` |
| 443 | Drop-in LibInvent and LinkInvent priors reach the runner that consumes them | Both the session's runner and every relayed REINVENT job bind the checkout's `priors` directory over `/app/REINVENT4/priors`, as the gateway already does, so a scaffold or linker campaign submitted successfully does not fail when execution starts because that container can see only the two priors baked into the image; the same drop-in directory works with either the local or Slurm run backend | `pytest tests/test_slurm_runs.py -v -k drop_in_generation_priors` |
| 442 | A scaffold or linker run remains attached to the conversation that launched it | `run_generation_campaign` is a launch tool in both the gateway's persisted run index and the browser's immediate progress extraction, so a queued LibInvent/LinkInvent job appears in that conversation's bottom progress bar and Jobs list and can be opened, rather than being recorded only as a referenced job and disappearing from both views | `pytest tests/test_session_data.py -v -k generation_campaign` and `cd services/ui/frontend && npx vitest run src/lib/runProgress.test.ts` |
| 441 | A long job can be stopped, and a set of structures prepared in one call | `cancel_preparation` and `cancel_synthesis_job` reach the cancel routes both services already exposed: `cancelled` was a state the status tools could report and nothing could cause, so a stuck protonation stage or an MCTS search submitted against the wrong stock held its worker until the service was restarted. `prepare_structures_batch` takes a list of paths and one profile and returns a job id per path, so preparing several entries of one target is a single call; a path that will not resolve carries an `error` in place of its id rather than losing the rest of the batch. All three are registered extension tools and so reachable by a pipeline step | `pytest tests/test_tool_exposure.py -v` |
| 440 | A cluster set really spans the series, and a medoid belongs to one cluster | k-medoids seeds by maximin — each medoid is the molecule furthest from those already chosen — so three clusters over three chemotypes return three representatives below 0.5 Tanimoto of each other; seeded by nearest instead, all three came from one chemotype while the cluster count stayed right, which is why the count is not what is asserted. The similarity matrix diagonal is set to 1.0, without which a medoid is the worst match for its own cluster, is assigned to another, and appears in two | `pytest tests/test_series_analysis.py -v` |
| 439 | A series can be clustered and its SAR read as matched molecular pairs | `cluster_molecules_by_similarity` groups molecules by Tanimoto similarity — Butina (distance cutoff, the MedChem standard) or k-medoids (fixed cluster count) — and returns the centroid per cluster so "ten diverse winners" is a named list rather than an impressionistic pick; `find_matched_molecular_pairs` finds every pair that differs by exactly one structural change and reports the transform and, when activities are supplied, the activity delta, which is the SAR relationship that is otherwise read by eye. Every pair is two of the molecules given and names both sides of its transform; an unrelated set yields no pairs rather than forced ones; the largest activity swing sorts first, including when only part of the series is annotated | `pytest tests/test_series_analysis.py tests/test_tool_exposure.py -v` |
| 438 | A target can be validated beyond its biochemical assay profile | `fetch_target_genetic_evidence` returns GWAS, pheWAS and rare-variant evidence from OpenTargets for a target's Ensembl id; `fetch_target_expression` checks the Human Protein Atlas for tissue expression so a target absent from the disease organ is caught before a campaign starts; `fetch_pathway_annotations` returns the Reactome and KEGG pathways the target participates in; `fetch_target_orthology` finds rat and mouse orthologs via OMA; all four are registered, gated read-only, and reachable by a pipeline step. Registration and exposure are what is asserted here; the request and response shapes each endpoint accepts are row 444 | `pytest tests/test_tool_exposure.py -v` |
| 437 | A generated molecule can be checked for patent claims, not just novelty | `search_patents_by_structure` searches SureChEMBL by SMILES (exact or similarity) and returns composition-of-matter claims; `search_patents_by_target` searches EPO OPS by gene name or target description; both say when their database could not be reached, because an unreachable database is not evidence of freedom to operate, and both carry a note that a hit is a flag for a patent attorney rather than a determination of infringement. Registration, exposure and the unreachable-database wording are what is asserted here; the two-step SureChEMBL request the search now makes is row 444 | `pytest tests/test_tool_exposure.py -v` |
| 436 | A TTP can hold a workflow, and that workflow can drive any tool the server has | The TTP schema gains an optional `pipeline` field: an ordered list of steps, each naming an MCP tool, its params, its `depends_on`, where to store the result, and whether to run after a failed dependency; `topological_order()` returns a valid execution order and cycles, unknown dependencies and duplicate ids are all refused at construction; existing TTPs without the field load unchanged. Critically, a step can name **any** of the server's 108 tools — the 50 extension tools are closures inside `register_tools` and so absent from `server.py`'s `globals()`, which left them uncallable by name until the registry began recording each implementation as it is registered; without that a pipeline could not chain the scaffold derivation it exists to chain | `pytest tests/test_tool_exposure.py tests/test_ttp.py -v` |
| 435 | The TTP UI covers every available ADMET stack and shows drug and agro endpoints separately | The UI reads `model_sets` from the ADMET service's `/health` response and generates one picker group per stack — ADMET for drug, ADMET (agro) for agro — so the agro endpoints (BCF, IGC50, LD50 for bee/mouse/rat, logKoc, vapour pressure) appear as scoring objectives for the first time; default transforms are defined for all 15 drug and all 8 agro-only endpoints; objectives include `model_set` in `service_params` for agro so the RL scorer uses the right model; and both `serviceObjective` and `objectiveGroup` honour the new field | `cd services/ui/frontend && npx tsc --noEmit` |
| 434 | A scaffold can be derived from a series, not just drawn | `derive_scaffold_from_series` finds the core a series shares and returns it with numbered `[*:n]` attachment points at the positions the series actually varies — the form LibInvent requires, refused on a bare `[*]` — using Bemis-Murcko first and falling back to the maximum common substructure when the Murcko core is shared too narrowly; it warns when the core found is too small to be a real scaffold (MCS always shares at least one atom, even on a structurally unrelated set, and decorating a one-atom core is de novo generation in disguise); `decompose_series_r_groups` turns the series around the same core into an R-group table, one row per molecule, so the substituents explored at each position can be read without eyeballing; both tools are in the agent's `REVIEWER_READ_ONLY_TOOLS` and in the UI's `DIRECT_TOOLS` list; and both are confirmed to be registered under the names the system prompt uses | `pytest tests/test_scaffold.py -v` |
| 433 | A scaffold can be drawn, not just typed | The profile builder offers the two generators that decorate a structure — LibInvent and LinkInvent — and will not start one without the scaffold or fragment pair it has nothing to generate from; the sketcher opens in a mode that knows what is being drawn, keeps the attachment-point tool available only there, refuses a scaffold with nowhere to decorate or a fragment with anything but one join, and numbers the points on the way out, because a sketcher draws the position but never the number and REINVENT refuses a bare `[*]`; a fragment drawn second is renumbered against the first so the two ends of the linker stay distinguishable; and the request names `run_generation_campaign`, since `run_optimization` refuses these two by name | `cd services/ui/frontend && npx vitest run src/lib/attachment.test.ts src/lib/ttpRun.test.ts` |
| 432 | A structure can be asked about, not just a name | `fetch_compound` returns the InChIKey, which is what joins a PubChem hit to a ChEMBL one where a name and a SMILES string both fail, and says when a name matched several compounds — naming the CID it took and the others — instead of returning a salt or the wrong enantiomer as though it were the only answer; a CID, an InChIKey or a SMILES can be looked up directly, the SMILES by POST because `#` and `+` in a URL path are PubChem's own documented trap; and `check_compound_is_known` answers whether a generated structure is already a known compound in either database, with a database that could not be reached saying so rather than counting as evidence of novelty | `pytest tests/test_pubchem.py tests/test_cross_check.py -v` |
| 431 | A service says what it can currently do | `/health` on the ADMET predictor names the stack its legacy `endpoints` list belongs to, so a deployment with the agro stack loaded stops reading as drug-only; the profiler reports `fpocket`, a binary its version scan could not see, as present or `absent`, so an image built before that Dockerfile layer stops answering every subpocket request with an empty list while calling itself well; `posebusters` and `admet-predictor` have their `app.py` bound over the baked copy like the profiler and campaign-store, so a one-line fix ships without an image rebuild; and InterPro's 204 — a good accession with no entries — is an empty list on the direct path as it already was through the annotation service, instead of the same protein looking domain-free or broken depending on whether `ANNOTATION_SERVICE_URL` is set | `pytest tests/test_service_self_report.py -v` |
| 430 | The suite fails only where something is wrong | `pytest tests/` completes on a checkout without REINVENT4 or a POSIX host, where it used to abort during collection and then report two dozen failures that said nothing about the code: the six modules that read `services/boltz2/app.py` or the agent's display skip on the missing POSIX-only `fcntl` or `rich` instead of killing the whole run; the tests that stop a job with `SIGUSR1` or read liveness from `/proc` say why they cannot run there; the TTP fixture checks that its base TTP was stored, so the eleven that depend on it skip naming the missing REINVENT4 checkout rather than each failing on `TTP not found` far from the cause; a file's size is compared against the bytes written and not a line-translated copy of them; kustomize's resource list is compared as POSIX paths, which is what it holds; a deadline two hours past is allowed to be *over* two hours past, an assertion that capped it at exactly two hours and so passed only where the clock was too coarse to tick between writing the deadline and reading it; and the watcher is given a session already listening before it starts polling, since its first tick consumes the feed and a tick landing before autonomy was switched on threw the news away with nothing awake to receive it | `pytest tests/ -q` |
| 429 | The ensemble answer reaches the caller, not just its size | `hotspot_ensemble` returns the fragment families with their binding mode, residue and member count, and the compatible pairs with their linker distance and overlap, instead of counting both and dropping them — the raw response can exceed the context window, so a ranked head of each travels back and the true count survives the cap; `ingest_fragment_activities` is registered, so the `potency` dict ligand efficiency needs can be built from a CSV instead of by hand; and the chemical component dictionary and citation abstracts `MetadataClient` already fetches have tools of their own | `pytest tests/test_interaction_profiler_tools.py -v` |
| 428 | A scoring config cannot quietly score something else | An unknown component name is refused where it is rendered, with the closest known name, instead of becoming a TOML table verbatim and failing inside REINVENT4 on the node minutes later; names still go through where the registry cannot be read, since the runner relays a TOML built elsewhere; a name differing from a known one only in case is accepted, because REINVENT4 keys its registry on the lowercased class name while the mapper emits the class name itself, so `ExternalRESTScorer` was refused against `externalrestscorer` and every external objective failed to render; and a config that mapped to no components says so in the log and in the TOML itself, because scored on QED alone it is otherwise indistinguishable from a healthy run | `pytest tests/test_ttp_for_reinvent.py -v` |
| 427 | A sub-agent cannot steer the fleet it belongs to | Every tool in `FLEET_TOOLS` is withheld from a Driver, asserted as a subset rather than a hand-written list — spelled out by hand it held `enqueue_task` and not `enqueue_tasks`, so a Driver could queue work for the whole campaign, rewrite the plan and claim proposals while the denial's own reason said only the conducting agent divides the work — and every withheld tool carries a reason, so a refusal never leaves the sub-agent without a way round | `pytest tests/test_supervisor.py::TestDriverToolBoundary -v` |
| 426 | A scaffold or linker run is asked for by name and started by the right tool | `run_optimization` refuses a TTP whose `generation.mode` is `libinvent` or `linkinvent` and names `run_generation_campaign`, instead of quietly launching the default de novo prior and reporting a healthy campaign that never touched the scaffold; the Driver's brief names that starter and stops treating `create_generation_campaign` as a way round writing the TTP, since it persists one under the id it is given | `pytest tests/test_optimization.py::TestRunOptimizationTool tests/test_supervisor.py::TestBuildGoal -v` |
| 425 | A generation run can be harvested by the task that started it | `submit_toml` keeps the TTP a run belongs to apart from the tag shown in job listings, so `job_for_ttp` finds a scaffold or linker run instead of comparing `generation-ttp-7` against `ttp-7` and reporting a healthy run as nothing started; `run_generation_campaign` forwards the real `ttp_id`; and a caller that passes no `ttp_id` keeps the old behaviour | `pytest tests/test_optimization.py -k submit_toml tests/test_generation_campaign_runtime.py -v` |
| 424 | A profiler route in the checkout is a profiler route on the node | `run.sh` binds the interaction-profiler's `app.py` over the copy baked into the image, so a route added since the last build answers on a running node instead of 404; the v2 `/profile_complex` and `/characterize_site` routes are registered on the app; and no route handler annotates a return type that is missing from module scope, which under postponed annotations leaves FastAPI with an unresolvable forward reference and makes `/openapi.json` 500 — the one request that would show a route is missing; and the Apptainer definition installs what the Dockerfile does — fpocket, `requests` and `freesasa` are each optional at import, so the image LUMI was running had none of them and `characterize_site_profile` answered with an empty pocket profile and the reasons buried in `warnings`, which reads as "no druggable sub-pocket" rather than "nothing was computed" | `pytest tests/test_interaction_profiler_app.py -v` |
| 423 | Boltz exploration-tier throughput | Exploration defaults (`recycling_steps=2`, `sampling_steps_affinity=50`, `diffusion_samples_affinity=1`) flow from config → mapper → scorer config; validation tier overrides to full-accuracy parameters; an exploration Boltz run with no explicit `max_steps` caps at 64 steps; the supervisor threads `fidelity_tier` through to `run_optimization`; the Boltz wrapper detects free VRAM per GCD at startup and shards a scoring batch across GCD slots concurrently, each subprocess pinned to its own GCD via `ROCR_VISIBLE_DEVICES` | `pytest tests/test_ttp.py tests/test_optimization.py::TestBoltzFidelityTier tests/test_supervisor.py::TestStartingTheRunDirectly -v` |
| 422 | Work is admitted as fast as the bottleneck takes it, not to a fixed number | The run window starts at the configured run limit and halves when a relayed run has waited more than five minutes for Slurm to start it — once per five minutes however often it is looked at, never below two — and earns back about one run per window of prompt starts; a local run is never a sign of a slow scheduler, and a refusal while it is closed says the cluster is slow to start runs and why, rather than to raise the limit. The runner's health and `get_pool_capacity` report the window, and a full one answers "not design" since other kinds need no run. The conductor's fixed Boltz threshold is a dispatch window instead: halved over the limit, held between half the limit and the limit, opened one a pass under half, so a deep queue lets a quiet campaign through at half instead of nothing. A batch is queued whole and says how many of its design tasks can start a run now, counting this campaign's design tasks already waiting for one. A design task waits for room before its agents start, and a run refused for a slot is retried until the deadline instead of failing the task and spending an attempt | `pytest tests/test_admission.py tests/test_slurm_runs.py tests/test_conductor.py tests/test_campaign_tools.py tests/test_supervisor.py -v` |
| 421 | What the campaign knows is on its Overview | The gateway reads the atlas and the current findings in one call, says when the store's cap left findings out, and answers 503 while the ledger is away; the Overview lists each branch with its latest claim, those with a disputed finding first and those nothing is known about last, the plan's order kept within each; a branch opens to its findings with their confidence, author, tags, evidence and notes, and a job id among the evidence opens its run only if the gateway has that run; the atlas is read again when a finding, a note or a report arrives, with no timer of its own, also after the history the panel holds has stopped growing | `pytest tests/test_web_server.py -v -k "knows or capped_read"` and `cd services/ui/frontend && npx vitest run src/lib/atlas.test.ts` |
| 420 | The UI does not work on what nobody is looking at | The dock hides a panel rather than unmounting it, so a view marks whether it is on screen and the panels nest: the analysis left open inside a tab that was switched away from polls no more than the tab does. The job list has one reader for the whole page instead of one per panel — the run bar, the Jobs tab, the results, the analysis, the picker and the campaign views read the same answer, one timer runs at the shortest period any of them asked for and none at all while nobody is watching, and a gateway that answers with an error keeps the last good list rather than blanking the run out from under the reader | `cd services/ui/frontend && npx vitest run src/hooks/PanelShown.test.tsx src/hooks/useJobList.test.ts` |
| 419 | The atlas: what the campaign knows, not what it ran | A finding is recorded under a branch with its evidence, confidence, tags and author, and is never deleted — a wrong one is superseded by the one correcting it, which hides it from the current view but keeps it readable, and no campaign may supersede another's; a note hangs off a finding as supports, disputes or extends rather than overwriting it; search wants every word somewhere in the claim, detail or tags, and a tag matches exactly; the overview counts each branch's findings and disputes with its latest claim, shows the branches the plan named but nobody filled, and keeps a branch outside the plan; case and spacing fold to one branch on the way in and the way out, so a workstream the plan spells differently still meets its own findings; the five atlas tools sign what they write with the agent's own id, and a store that is down is an error rather than a lost turn; a sub-agent's brief carries its branch's findings and is told the campaign, the task and the branch to record against, and so does the goal of a design task writing its own profile | `pytest tests/test_campaign_store.py tests/test_campaign_tools.py tests/test_supervisor.py -v -k "Atlas or SubAgentBrief or BuildGoal"` |
| 418 | The conductor fills the team on its first turn | The opening-dispatch note names the team size and says every sub-agent must get a task in the first enqueue_tasks call; the stalled-idle notice fires once per stall, not on every watcher look | `pytest tests/test_web_agents.py tests/test_web_autonomy_watch.py -v -k "allowance or stalled or told_once or stalls_again"` |
| 417 | The sub-agent allowance control accepts any number | The fixed-preset "Max" button is replaced by a "…" button that opens a numeric input; a number above the deployment's capacity is clamped; the clamping is shown; the fixed steps (2 4 6 8) still work | (UI only — no pure-unit test for browser widget) |
| 416 | A campaign molecule opens its cofolded complex | Each harvested molecule names the run that scored it and the ledger's narrow read keeps that id (a molecule posted without one has none); the pooled list takes the molecule's run, else its result's; the Molecules view asks once per run on screen for its structures and shows 3D on the rows that have one | `pytest tests/test_supervisor.py tests/test_campaign_store.py -v -k "names_the_run or keeps_the_run"` and `cd services/ui/frontend && npx vitest run src/lib/fleet.test.ts` |
| 415 | Every run of the conversation in one Jobs view | The job list says whether a run scores with Boltz by the relay's own test, and its Slurm job; the Conductor page's Jobs view lists the conductor's and every sub-agent's runs (or every conversation's), grouped running, waiting and finished, filterable by who started them, naming the task, the wait for GPUs and the run time; picking one opens its Summary, Results and Analysis there, and the run bar's Job button opens it whoever started it | `pytest tests/test_optimization.py -v -k to_dict` and `cd services/ui/frontend && npx vitest run src/lib/jobList.test.ts src/lib/jobs.test.ts` |
| 414 | A message sent while the conductor works is never lost | A message arriving mid-turn — the user's own or one sent while a wake-up or a continuation runs — is echoed marked as interjected, read by the running turn after its current step, once; one the turn never read starts the next turn, several joined, and is answered after a Stop too; the transcript keeps it inside the open turn so the calls still running get their results, and the composer sends while busy | `pytest tests/test_web_autonomy.py tests/test_web_server.py -v -k "mid_turn or never_read or never_run_at_once or kept_and_answered"` and `cd services/ui/frontend && npx vitest run src/lib/transcript.test.ts` |
| 413 | Unknown transform parameters caught at TTP creation | A `TransformConfig` with a field not in the transform's spec (e.g. `value` on `step`, which takes `low`/`high`) is rejected immediately with a named error; a non-numeric string in an unrecognised field is also caught; known parameters (`low`, `high`, `k`, etc.) still work; defaults are still filled in; validation happens before any resource is allocated | `pytest tests/test_ttp.py -v -k "UnknownParameter"` |
| 412 | A run as its own Slurm job | With `REINVENT_RUN_RELAY` set the runner starts nothing itself: it writes the worker's command as `worker.sh` and a submit marker for the session's relay, holds the slot as queued while Slurm has the job pending, running once it runs, completed on a clean exit and failed with the job's own output otherwise (a refused `sbatch` included); a stop is asked of the relay and reads as stopped, a run still waiting can be stopped, a runner restart adopts a live Slurm run, and listing jobs does not wait on its own lock; `SCORER_URL_OVERRIDES` points a scorer at the run's private Boltz and keeps the path; unset, runs are subprocesses as before | `pytest tests/test_slurm_runs.py -v` |
| 411 | A lead runs a workstream and reports it up | A `lead` task opens a campaign under its parent with the parent's objectives and the lead's concurrency (found again, not reopened, when the lead is claimed a second time), proposes its tasks without holding more than its team, stops dispatching after three tasks per sub-agent, drains and finalizes; a parent stopped ends it; its report carries every sub-task's finding, the best score per objective, the best molecules and the runs, and hands up to three open proposals to the parent as follow-ups; the supervisor routes a lead to this rather than to an agent | `pytest tests/test_leading.py tests/test_supervisor.py -v` |
| 410 | The Campaign tab says how busy the team is | The fleet view of a campaign carries the ledger's metrics, or none when the ledger cannot say; the Team card reads out who is working of how many, the queue against what to keep ready (in the warning colour when under it), utilisation since the first task and the median wait between tasks, leaving out what is not measured and closed campaigns; the ledger's metrics count running and queued over the campaign and its leads and time each sub-agent's wait between tasks | `pytest tests/test_campaign_store.py tests/test_web_server.py -v` and `cd services/ui/frontend && npx vitest run src/lib/fleet.test.ts` |
| 409 | A plan the conductor decides from | `set_campaign_plan` keeps the workstreams on the campaign (ids unique); coverage counts each workstream's tasks, its latest finding and its open proposals, names the open ones nobody is working as gaps and counts work outside the plan rather than losing it; the proposer is shown each workstream with what it has had, tags its tasks with one, and a workstream not in the plan is dropped from the task, not the task | `pytest tests/test_campaign_store.py tests/test_campaign_tools.py tests/test_proposing.py -v` |
| 408 | The floor: no sub-agent waits on a conductor that will not dispatch | While an agent conducts, a free slot is its to fill for 90 s (`CONDUCTOR_FLOOR_SECONDS`); past that the loop fills it, from what the sub-agents proposed first and from the proposer for the rest, at priority -1 so the agent's own tasks are claimed first; a team back at work restarts the grace, a lead's campaign's tasks hold slots too, and a floor of 0 leaves every slot to the agent | `pytest tests/test_conductor.py -v` |
| 407 | A ready queue behind the team | Tasks are queued in batches, whole or not at all (a task without an intent or with a field no task has refuses the batch); higher priority is claimed first, then oldest; who ran a task is kept after it is done; the status says how deep to keep the queue (half the team, at least one), and the conductor is told once when the queue runs under that, again only after it was refilled | `pytest tests/test_campaign_store.py tests/test_campaign_tools.py tests/test_web_autonomy_watch.py -v` |
| 406 | A sub-agent proposes its next tasks | Every brief asks for a "Next tasks" section and up to three lines of it are read off the report; an unknown kind is research and a line without a title is titled by its first words; what a report proposes is kept as a proposal nobody can claim until the conductor takes it, taken once, derived from the report that proposed it, and continuing its workstream unless it says otherwise; the conductor's wake says how many were proposed | `pytest tests/test_supervisor.py tests/test_campaign_store.py tests/test_web_autonomy_watch.py -v` |
| 405 | News cuts the conductor's pause short | A report, a failed task or idle sub-agents arriving while an automatic turn is scheduled minutes away (after an answer that saw nothing to do) wake the conductor at the next look instead of after the pause; a pause with nothing new runs its course | `pytest tests/test_web_autonomy_watch.py -v` |

### Staying inside the context window (`tests/test_llm_context.py`, `tests/test_web_autonomy_watch.py`)

A long campaign fills a context: the conductor reads reports from six
sub-agents, a results table runs to tens of thousands of tokens, and one turn
can call tools for hours. Three layers keep that working — the agent bounds a
single tool result, the backend asks the endpoint how much room it has and
gives some back when it runs out, and the gateway restarts the conversation
from a brief. No endpoint is contacted: the model listing, the rejections and
ChEMBL are stubbed.

**Dependencies**: `openai`, `anthropic`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 372 | One tool result cannot swamp the conversation | A result past the cap keeps its head and its tail with the middle named and cut, and says which tool it came from so the model can ask again more narrowly. The browser and the transcript still get the whole thing — only what the model re-reads on every later round is bounded | `pytest tests/test_llm_context.py::TestToolResultSize -v` |
| 373 | The window is read, not guessed | `max_model_len` from `/v1/models` is what the deployment serves, which is the only number that matters: the same weights are served at 32k on one endpoint and 128k on another. Asked once, overridden by `REINVENT4_AGENT_CONTEXT_TOKENS`, and an endpoint that will not say is not an error | `pytest tests/test_llm_context.py::TestTheServedWindow -v` |
| 374 | A reply cap that no longer fits is lowered | vLLM derives the room for a reply by subtracting the prompt from `max-model-len`, so a cap sized for an empty conversation starts being refused once the conversation is long | `pytest tests/test_llm_context.py::TestTheReplyCapFitsTheWindow -v` |
| 375 | A full window is recovered from, not reported | The reply is given up first and the oldest exchange only then, in every wording the endpoints use — including vLLM's negative cap, which must not be read as a request for the other token-parameter name | `pytest tests/test_llm_context.py -v -k "MakingRoom or wording"` |
| 376 | Dropping history keeps it valid | The cut lands on a question, never between a tool call and its result (which both APIs reject outright), the system prompt stays, and a backend with nothing to drop says so rather than pretending | `pytest tests/test_llm_context.py -v -k Forget` |
| 377 | The conversation is condensed before it overflows | A history past the threshold is replaced by a brief on the user's own turns as well as the agent's, with the question still last; the threshold follows the window the endpoint serves, so a 32k deployment condenses where a 128k one does not | `pytest tests/test_web_autonomy_watch.py -v -k "condensed or window"` |
| 378 | One long turn is recovered from too | A turn has no older exchange to drop, so the results the model has already answered are cut to their opening lines with a note to call again; the latest stays whole, and once they are short it gives up instead of looping | `pytest tests/test_llm_context.py -v -k "long_turn or short_it"` |
| 379 | An overflow is reported as one | When nothing more can be given up, the error says the conversation outgrew the window — not the URL-routing checks that a 404 or a wrong deployment would get | `pytest tests/test_llm_context.py -v -k routing` |
| 380 | A ChEMBL target search is a summary | A raw `/target/search` record carries every component's cross-references and synonyms — twenty of them for "JAK3 human" were ~190k tokens, a whole window. Without `only` the records are the same short form `get_chembl_target` returns; with `only` they are what was asked for | `pytest tests/test_chembl.py -v -k "slim or only_keeps"` |

### Boltz Gateway (`tests/test_boltz_gateway.py`)

Tests for `services/boltz-gateway/app.py`: the per-molecule result cache, batch
coalescing, in-flight de-duplication, `/stats`, and the `202`+poll contract. The
upstream Boltz call is stubbed and the cache points at a `tmp_path`, so no GPU,
cluster, or service is needed.

**Dependencies**: `fastapi`, `pydantic`, `rdkit`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 298 | Cache keying | The same molecule spelled differently shares a key (`randomize_smiles` makes that the common case); target and predict params do not; policy flags and per-call plumbing are excluded | `pytest tests/test_boltz_gateway.py::TestCacheKeying -v` |
| 299 | MSA keyed by content | Two paths holding one alignment share the whole key, editing an `.a3m` invalidates it, and the path is still forwarded so Boltz can find it | same class |
| 300 | Cache behaviour | A repeat batch predicts nothing, a randomised spelling still hits, a duplicate inside one batch is predicted once, and a partly cached batch only predicts the rest | `pytest tests/test_boltz_gateway.py::TestCacheBehaviour -v` |
| 301 | Tiers | `validation` bypasses the read but still populates; `cache_policy: none` does neither | same class |
| 302 | Structures survive a hit | The cached structure path comes back, and an entry naming a deleted file is dropped rather than served — downstream scorers re-open these by path | same class |
| 303 | What is cacheable | A deterministic skip (too many heavy atoms) is kept; a failed prediction never is, so one bad run cannot poison a molecule | same class |
| 304 | Coalescing | Concurrent batches share one upstream call; different targets or fidelities are never merged; a full batch dispatches without waiting and an oversized one splits | `pytest tests/test_boltz_gateway.py::TestCoalescing -v` |
| 305 | In-flight de-duplication | Three concurrent requests for one molecule cost one prediction — the cross-supervisor saving caching alone cannot give | same class |
| 306 | Response shape | Scores, per-molecule metadata and structures all come back in the caller's order and the right length; an upstream failure is a 502 and leaves nothing cached | `pytest tests/test_boltz_gateway.py::TestResponseShape -v` |
| 307 | Async contract | `202` + `task_id`, `202` while pending, `200` once, then `404`; a failure surfaces as 502 | `pytest tests/test_boltz_gateway.py::TestAsyncApi -v` |
| 308 | Stats | Hit rate, molecules per upstream call, latency percentiles, queue depth and duplicate/invalid counts | `pytest tests/test_boltz_gateway.py::TestStats -v` |
| 309 | Replica view | Healthy replicas and the proxy's **backend** queue row; an unreachable proxy still allows one call rather than wedging | `pytest tests/test_boltz_gateway.py::TestReplicaView -v` |
| 310 | Drop-in contract | The same request body validates against both the gateway and the Boltz wrapper, and the response has the three keys the scorer reads | `pytest tests/test_boltz_gateway.py::TestContract -v` |

### Scorer Timeout and Polling (`tests/test_external_scoring.py`)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 311 | Configurable timeout | Default unchanged at 1800 s; component config beats `EXTERNAL_SCORER_TIMEOUT_SECONDS`; a nonsense env value falls back instead of disabling the timeout | `pytest tests/test_external_scoring.py::TestTimeoutConfiguration -v` |
| 312 | Submit then poll | `?mode=async`, the result URL derived from the score URL, backoff between polls, and the overall deadline still bounding it | `pytest tests/test_external_scoring.py::TestSubmitThenPoll -v` |
| 313 | Async misconfiguration is visible | A service that answers the submit normally instead of 202 is honoured as-is; a 202 with no `task_id` fails rather than hanging | same class |

### REINVENT Runner (`tests/test_reinvent_runner.py`)

Tests for `services/reinvent-runner/app.py` and the client that makes it a
drop-in for the local `JobManager`. REINVENT4 is not installed in the test
environment, so the manager and tool functions are patched out — which is also
the point: nothing above the seam should need them.

**Dependencies**: `fastapi`, `requests`, `PyYAML`

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 322 | Endpoints pass through | `/jobs`, `/tl-jobs`, status, stop, candidates, lists and the registries each delegate to the code that already exists, with `extra="forbid"` so a stale client cannot silently drop a parameter | `pytest tests/test_reinvent_runner.py::TestRunnerEndpoints -v` |
| 323 | Refusals stay refusals | "Maximum concurrent jobs" comes back as `200` with an error dict, because that is the manager's own contract and preserving it keeps the client a drop-in; only unexpected failures are 5xx | same class |
| 324 | Health names the real risk | Reports `reinvent_importable`, since a false there means the image is wrong rather than the service being down | same class |
| 325 | TTPs stay agent-side | `/score` takes the TTP inline and has no `ttp_id` parameter at all | same class |
| 326 | The client is a drop-in | Offers exactly the seven methods callers use, all of which exist on the local manager too | `pytest tests/test_reinvent_runner.py::TestRemoteJobManager -v` |
| 327 | Failures keep their shape | An unreachable runner, a 5xx, and an unparseable body all become `{"error": ...}` rather than raising — callers would crash on a raise | same class |
| 328 | List methods degrade safely | `list_jobs`/`get_alerts` return `[]` on failure, since an error dict would break every caller that iterates | same class |
| 329 | Job ids are URL-encoded | A path-traversal-shaped id cannot escape its endpoint | same class |
| 330 | The seam decides | `get_job_manager()` returns the local manager by default and the remote one when `runner.enabled`, with no restart needed; discovery and scoring follow the same switch | `pytest tests/test_reinvent_runner.py::TestSeam -v` |
| 331 | A ttp_id is resolved first | Sent inline to the runner; an unresolvable id never reaches it | same class |
| 332 | The runner never proxies to itself | `REINVENT_RUNNER_SELF` forces the local path, is set in both the image and the manifest, and the job endpoints hold their own manager so they cannot loop regardless | same class |

#### Image Split

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 333 | Agent image is slim | No `COPY REINVENT4/`, no Zenodo download, no `REINVENT4_ROOT`, and not built on `rocm/pytorch` — 164 MB against 40.8 GB unpacked | `pytest tests/test_reinvent_runner.py::TestImageSplit -v` |
| 334 | It stays slim | The build asserts `reinvent` and `torch` are **absent**, so a transitive dependency pulling them back fails the build instead of restoring a 40 GB image | same class |
| 335 | Runner owns the heavy half | REINVENT4, the priors and the ROCm base are there, `externalrestscorer` is still verified as registered, and `reinvent4_agent` is not installed | same class |
| 336 | No longer FROM dd-agent | The runner builds from its own base with the repo root as context, since it copies `REINVENT4/` and `reinvent4_mcp/` | same class |
| 337 | Agent needs no GPU | `amd.com/gpu` is absent from the agent Deployment's requests and limits; only Boltz and the runner ask for one | same class, plus `pytest tests/test_k8s_manifests.py::TestPodSplit -v` |
| 338 | The runner is mandatory | Both the ConfigMap and the baked `reinvent4_mcp_config.yaml` enable it, because a slim agent has no local REINVENT4 to fall back on | same class |
| 339 | Compose moved the GPU too | The agent service drops the ROCm devices and the 128 g shm and waits on the runner, which has them; the runner still has no `ttps` mount | same class |

### Boltz Placement Harness (`tests/test_boltz_placement.py`)

Tests for `k8s/scripts/verify-boltz-placement.py`, which proves concurrent
batches land on idle replicas instead of stacking on a busy one. The stats
endpoint and the HTTP calls are stubbed, so no cluster, service, or GPU is
needed.

**Dependencies**: `PyYAML`, `rdkit` (via the Boltz wrapper the payload is
validated against)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 292 | Stats parsing | Columns indexed by name, only this backend's rows, and unresolved slots excluded from "ready" | `pytest tests/test_boltz_placement.py::TestStatsParsing -v` |
| 293 | Backend queue row | An unplaceable request waits in the **backend** queue, so per-server `qcur` stays 0 — reading only server rows reports "nothing queued" while a batch is queued | same class |
| 294 | Verdict | Doubling up, an idle replica left unused, and a 503 queue timeout are each reported as their own failure; a missing replica header is inconclusive | `pytest tests/test_boltz_placement.py::TestVerdict -v` |
| 295 | Request shape | Boltz accepts the payload, the panel molecules all fold and are distinct so a cache cannot answer them, and the defaults are cheaper than production | `pytest tests/test_boltz_placement.py::TestRequest -v` |
| 296 | Harness vs manifest | Backend name, both ports, and the `X-Boltz-Replica` header match the deployed gateway, and the client timeout outlasts the gateway's queue | `pytest tests/test_boltz_placement.py::TestAgainstTheManifest -v` |
| 297 | CLI | Defaults to one more request than replicas so the queue is exercised; refuses zero requests, or more than the panel holds | `pytest tests/test_boltz_placement.py::TestCLI -v` |

### Cross-Pod Artifact Handoff (`tests/test_artifact_handoff.py`)

Tests for the handoff itself and for
`k8s/scripts/verify-artifact-handoff.py`, which proves it against a live
cluster. Boltz hands the consumers an absolute path on `/shared` and never the
structure, so the split only works if the consumer pods can open the
producer's files. kubectl and the HTTP calls are stubbed, so no cluster,
service, or GPU is needed.

**Dependencies**: `PyYAML`, `pydantic`, `rdkit`, `pandas` (the last two come
with the service modules under test)

| # | Test | What it verifies | Command |
|---|------|------------------|---------|
| 287 | Path portability | A relative path, or one outside `/shared`, is refused: neither means the same thing in two pods | `pytest tests/test_artifact_handoff.py::TestPathAudit -v` |
| 288 | Index alignment | A short list, a scored molecule with no structure, and a structure sitting where Boltz skipped one are all reported — the last is the off-by-one that pairs a pose with the wrong SMILES | same class |
| 289 | Payload parity | The payloads the harness relays are byte-identical to what `ExternalRESTScorer` injects, holes included rather than compacted | `pytest tests/test_artifact_handoff.py::TestRelayedPayloads -v` |
| 290 | Consumers accept them | PoseBusters' and the interaction profiler's own models validate the index-aligned list with `None` holes; Boltz is asked for `pdb`, since PoseBusters reads HETATM records | same class |
| 291 | Deliberate hole | The panel passes Boltz's validator, the 60-heavy-atom molecule is skipped for an explicit size reason, and it is still a parseable SMILES so the profiler fails it for the right reason | `pytest tests/test_artifact_handoff.py::TestBatch -v` |
| 292 | Extracted-ligand path | `derived_sdf_path` matches where `_boltz_pdb_to_sdf` really writes, which is inside the producer's own output directory | `pytest tests/test_artifact_handoff.py::TestDerivedSdfPath -v` |
| 293 | Failure classification | 200-with-nulls is a storage failure, a 4xx is a config one, a 5xx is neither, and a scored hole is misalignment | `pytest tests/test_artifact_handoff.py::TestConsumerClassification -v` |
| 294 | Verdict | A blind consumer or an unportable path disproves it; nothing produced or an unreachable service is inconclusive; an invisible `_lig.sdf` means the pods resolved one path on two filesystems | `pytest tests/test_artifact_handoff.py::TestVerdict -v` |
| 295 | Same-pod guard | A green run where producer and consumer share a pod exits 2, so a re-run of the in-pod proof cannot pass as the cross-pod one | same class |
| 296 | Findings | Each failure names its next step — the object-storage fallback, the config fix, or `--namespace` | `pytest tests/test_artifact_handoff.py::TestFindings -v` |
| 297 | kubectl probes | The pod is resolved from the `name` label and only when `Running`; a missing kubectl degrades to "unknown"; a zero-byte SDF counts as missing | `pytest tests/test_artifact_handoff.py::TestKubectlProbes -v` |
| 298 | Health gate | An unreachable service, or a profiler reporting `shared_dir_exists: false`, stops the run before the GPU call | `pytest tests/test_artifact_handoff.py::TestHealthGate -v` |
| 299 | Endpoints | The cluster URLs are the ConfigMap's `service_url`s and the documented port-forwards map the same ports; replay mode refuses a structure without its SMILES | `pytest tests/test_artifact_handoff.py::TestEndpointsAndCli -v` |
| 300 | End to end | `main` drives producer → consumers → verdict: three services dialled once each, the paths relayed verbatim, and exit 0/1/2 for proven/disproven/inconclusive | `pytest tests/test_artifact_handoff.py::TestEndToEnd -v` |

### Run All Unit Tests

```bash
python -m pytest tests/ -v
```

Expected: all 650+ tests pass in a few seconds.

---

## Tier 2: Integration Tests (require REINVENT4)

These tests use the real REINVENT4 scoring engine. Run them on a machine where REINVENT4 is installed (or `REINVENT4_ROOT` is set).

### Prerequisites

```bash
export REINVENT4_ROOT=/path/to/REINVENT4
# Verify:
python -c "import sys; sys.path.insert(0, '$REINVENT4_ROOT'); import reinvent; print('OK')"
```

### Config Loading

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 1 | Default config loads | `python -c "from reinvent4_mcp.config import load_config; c = load_config(); print(c.reinvent.device)"` | Prints `cuda:0` (or your device) |
| 2 | Env var config path | `REINVENT4_MCP_CONFIG=reinvent4_mcp_config.yaml python -c "from reinvent4_mcp.config import load_config; c = load_config(); print(c.scoring.aggregation)"` | Prints `geometric_mean` |
| 3 | Invalid config rejected | Create a YAML with `reinvent: {batch_size: -1}` and load it | `ValidationError` raised |
| 4 | Unknown key rejected | Create a YAML with `reinvent: {unknown_field: true}` and load it | `ValidationError` raised |

### Discovery Tools

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 5 | List scoring components | `python -c "from reinvent4_mcp.tools.discovery import list_scoring_components; cs = list_scoring_components(); print(f'{len(cs)} components'); print([c['name'] for c in cs[:5]])"` | Lists 40+ components including `QED`, `MolecularWeight`, `SAScore` |
| 6 | List available models | `python -c "from reinvent4_mcp.tools.discovery import list_available_models; ms = list_available_models(); print([(m['key'], m['exists']) for m in ms])"` | Lists model keys with `exists` status |
| 7 | List transforms | `python -c "from reinvent4_mcp.tools.discovery import list_transforms; ts = list_transforms(); print(f'{len(ts)} transforms'); print([t['name'] for t in ts])"` | Lists transform types (`sigmoid`, `reverse_sigmoid`, etc.) |

### Direct Scoring (Real REINVENT4)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 8 | Score with QED | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O'], components=[{'type': 'QED'}]); print([(m['smiles'], m['total_score']) for m in r['molecules']])"` | Three molecules with QED scores between 0 and 1 |
| 9 | Score with multiple components | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['CC(=O)OC1=CC=CC=C1C(=O)O'], components=[{'type': 'QED', 'weight': 0.5}, {'type': 'MolecularWeight', 'weight': 0.5, 'transform': {'type': 'double_sigmoid', 'high': 500, 'low': 200, 'coef_div': 500, 'coef_si': 20, 'coef_se': 20}}]); print(r['molecules'][0])"` | Aspirin scored with both QED and MW, total score is aggregated |
| 10 | Score with invalid SMILES | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['not_a_smiles'], components=[{'type': 'QED'}]); print(r)"` | Returns result (may show NaN for invalid SMILES) |
| 11 | Score with SAScore | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34'], components=[{'type': 'SAScore'}]); print(r['molecules'][0])"` | Returns SA score for a complex molecule |
| 12 | Score with invalid component | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; r = score_molecules(['CCO'], components=[{'type': 'NonExistentComponent'}]); print(r)"` | Returns `{"error": "Failed to create scorer: ..."}` |

### TTP-Based Scoring (Real REINVENT4)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 13 | Score with inline TTP (native only) | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; ttp = {'id': 'test', 'name': 'Test', 'scoring': {'objectives': [{'name': 'QED', 'type': 'native', 'component': 'QED'}, {'name': 'SA', 'type': 'native', 'component': 'SAScore'}]}}; r = score_molecules(['CCO', 'c1ccccc1'], ttp=ttp); print(r['molecules'])"` | Two molecules with QED and SA scores |
| 14 | Score with TTP + pre-scoring | Same as above but add a `pre_scoring` section with `MolecularWeight` | Pre-scoring component included in results |
| 15 | TTP with transforms | Use transforms (e.g., `reverse_sigmoid` on SA score) | Transformed scores between 0-1, raw scores in original units |
| 16 | Mixed TTP notes externals | Use a TTP with both native and external objectives | `note` field lists skipped external objectives |

### TTP CRUD (Real File Persistence)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 17 | Create TTP | `python -c "from reinvent4_mcp.tools.ttp import create_ttp, reset_store; reset_store(); r = create_ttp({'id': 'test-manual', 'name': 'Manual Test', 'scoring': {'objectives': [{'name': 'SA', 'type': 'native', 'component': 'SAScore'}]}}); print(r)"` | `{"status": "created", "id": "test-manual", "version": 1, ...}` |
| 18 | Get TTP | After creating, run `get_ttp("test-manual")` | Returns full TTP JSON |
| 19 | Update TTP | `update_ttp("test-manual", {"name": "Updated"})` | Version incremented to 2 |
| 20 | List TTPs | `list_ttps()` | Shows all stored TTPs |
| 21 | Delete TTP | `delete_ttp("test-manual")` | `{"status": "deleted"}` |

### External Service Discovery (requires network)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 22 | List external services (none running) | `python -c "from reinvent4_mcp.tools.external import list_external_services; print(list_external_services())"` | All configured services report `"unreachable"` |
| 23 | List external services (Boltz running) | Start Boltz service first, then run the above | Boltz reports `"ok"`, PoseBusters `"unreachable"` |

### TTP-Based Scoring with External Objectives (requires REINVENT4 + services)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 24 | Score mixed TTP (externals down) | `python -c "from reinvent4_mcp.tools.scoring import score_molecules; ttp = {'id': 'test', 'name': 'Test', 'target': {'protein_sequence': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH'}, 'scoring': {'objectives': [{'name': 'QED', 'type': 'native', 'component': 'QED'}, {'name': 'Binding', 'type': 'external', 'service': 'boltz', 'service_url': 'http://localhost:8091/score'}]}}; r = score_molecules(['CCO'], ttp=ttp); print(r.get('note', 'no note'))"` | Returns native scores + `note` listing skipped externals |
| 25 | Score mixed TTP (externals up) | Same TTP but with Boltz + PoseBusters running | All objectives scored, no `note` field, artifacts present |

### RL Optimization (requires REINVENT4 + GPU)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 26 | Run optimization with native TTP | Create a TTP with QED + SA, call `run_optimization(ttp_id=..., max_steps=10, batch_size=4)` | Returns `job_id`, TOML config written, REINVENT4 subprocess starts |
| 27 | Get status during run | Call `get_optimization_status(job_id)` while step < max_steps | `state: "running"`, `trajectory` with current_step, mean_total_score, per_component stats |
| 28 | Convergence diagnostics | Poll status after 10+ steps | `trajectory.diagnostics` present with `convergence_status`, `component_health`, `trend` |
| 29 | Stop and checkpoint | Call `stop_optimization(job_id)` | `state: "stopped"`, `checkpoint_path` present and file exists |
| 30 | Resume from checkpoint | Call `run_optimization(ttp_id=..., resume_from="<checkpoint_path>")` | New job starts with `use_checkpoint = true`, agent_file points to checkpoint |
| 31 | Get top candidates | After run completes: `get_top_candidates(job_id, top_n=5)` | Ranked list with SMILES, total_score, per-component breakdowns |
| 32 | Read results CSV | Call `read_results(csv_path)` on the output CSV | Summary stats: n_rows, n_unique_smiles, validity_rate, score distribution |
| 33 | List jobs | Call `list_jobs()` | All tracked jobs with states, TTP IDs, timestamps |

### Transfer Learning (requires REINVENT4 + GPU)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 34 | Run TL with inline SMILES | `run_transfer_learning(smiles=["CCO", "c1ccccc1", ...], num_epochs=5)` | Job starts, SMILES written to `train.smi`, TOML has `run_type = "transfer_learning"` |
| 35 | Run TL with file | Write SMILES to a file, call `run_transfer_learning(smiles_file="path.smi")` | Job starts, TOML references the provided file path |
| 36 | TL with validation data | Provide `validation_smiles=[...]` | Validation file written, TOML includes `validation_smiles_file` |
| 37 | Monitor TL progress | Poll `get_optimization_status(job_id)` | State transitions from `running` to `completed` |
| 38 | TL output model | After completion: check `output_model_file` in job dict | Model file exists in working directory, epoch checkpoints present |
| 39 | Use TL model in RL | Run `run_optimization(model="<tl_output_path>", ttp_id=...)` | RL starts with TL model as the agent |

---

## Tier 3: End-to-End Tests (full agent stack)

These tests verify the complete `dd-agent` -> `dd-mcp` -> REINVENT4 pipeline.

### Prerequisites

- REINVENT4 installed or `REINVENT4_ROOT` set
- `dd-mcp` and `dd-agent` installed (`pip install ./reinvent4_mcp ./reinvent4_agent`)
- LLM API access (API key or gateway key)

### MCP Server Smoke Test

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 1 | MCP Inspector | `npx @modelcontextprotocol/inspector dd-mcp` | Opens web UI at `localhost:6274`. Lists all 9 tools. Can call `list_scoring_components` and get results. |
| 2 | Server starts | `dd-mcp 2>server.log &` (sends JSON-RPC over stdin/stdout) | Server logs `Starting Drug Discovery MCP server v0.1.0` to stderr |

### Agent CLI Smoke Test

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 3 | Agent starts and connects | `dd-agent --backend anthropic --model claude-sonnet-4.6 --base-url "$REINVENT4_AGENT_BASE_URL" --gateway-key "$LLM_GATEWAY_KEY" --no-verify-ssl` | Prints welcome banner, lists 9 tools, shows `You>` prompt |
| 4 | Help command | Type `/help` | Shows available commands (`/help`, `/reset`, `/exit`) |
| 5 | List components | Ask: `What scoring components are available?` | Agent calls `list_scoring_components` tool, lists 40+ components |
| 6 | List models | Ask: `What generative models are available?` | Agent calls `list_available_models`, lists model priors with availability |
| 7 | Score molecules | Ask: `Score aspirin (CC(=O)OC1=CC=CC=C1C(=O)O) for QED and molecular weight` | Agent calls `score_molecules`, renders a score table |
| 8 | Multi-component scoring | Ask: `Score CCO and c1ccccc1 for QED, SA score, and LogP` | Agent composes a multi-component request, returns all scores |
| 9 | Create TTP | Ask: `Create a TTP called "test-ttp" for lead optimization with QED and SA score objectives` | Agent calls `create_ttp`, confirms creation with ID and version |
| 10 | Score with TTP | Ask: `Score aspirin using the test-ttp profile` | Agent calls `score_molecules` with `ttp_id`, returns TTP-based scores |
| 11 | Update TTP | Ask: `Update test-ttp to add molecular weight with weight 0.3` | Agent calls `update_ttp`, confirms version 2 |
| 12 | Score with updated TTP | Score again after update | Now includes MW in the scoring |
| 13 | List TTPs | Ask: `What TTPs do I have?` | Agent calls `list_ttps`, shows stored profiles |
| 14 | Delete TTP | Ask: `Delete the test-ttp profile` | Agent calls `delete_ttp`, confirms deletion |
| 15 | Conversation reset | Type `/reset`, then ask a follow-up referencing earlier context | Agent has no memory of previous conversation |
| 16 | Exit | Type `/exit` or `/quit` | Agent exits cleanly |

### Agent with Logging

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 17 | Session logging | `dd-agent --log-dir ./test-logs ...` | After a conversation: `test-logs/session_*.log` (human-readable) and `session_*.jsonl` (structured transcript) exist |
| 18 | Transcript structure | Inspect the `.jsonl` file | Each line is valid JSON with `ts`, `turn`, `event` fields |

### Docker Deployment Test (dd-agent)

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 19 | Docker build | `docker build -f services/dd-agent/Dockerfile -t dd-agent .` | Image builds successfully |
| 20 | Docker smoke test | `docker run --rm --entrypoint bash dd-agent -c 'echo $REINVENT4_ROOT && python -c "import reinvent; print(reinvent.__file__)"'` | Prints `/app/REINVENT4` and the reinvent package path |
| 21 | Docker dd-mcp check | `docker run --rm --entrypoint bash dd-agent -c 'which dd-mcp && which dd-agent'` | Both commands found |
| 22 | Docker interactive | `docker run -it --rm -e LLM_GATEWAY_KEY="$LLM_GATEWAY_KEY" dd-agent` | Agent starts inside the container |

### Docker Deployment Test (Boltz2 + PoseBusters services)

Requires Docker (with GPU support for Boltz) and the `docker-compose.yml` in the project root.

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 23 | Boltz2 image builds | `docker compose build boltz` | Image builds from `services/boltz2/Dockerfile` |
| 24 | PoseBusters image builds | `docker compose build posebusters` | Image builds from `services/posebusters/Dockerfile` |
| 25 | Boltz2 health check | `docker compose up -d boltz && sleep 30 && curl http://localhost:8091/health` | `{"status": "ok", "no_kernels": true}` |
| 26 | PoseBusters health check | `docker compose up -d posebusters && sleep 10 && curl http://localhost:8082/health` | `{"status": "ok"}` |
| 27 | Full stack up | `docker compose up -d` | All three services (`reinvent4-mcp`, `boltz`, `posebusters`) healthy |
| 28 | Boltz2 score endpoint | `curl -X POST http://localhost:8091/score -H 'Content-Type: application/json' -d '{"smiles": ["CCO"], "params": {"protein_sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH", "use_msa_server": true}}'` | JSON with `scores`, `metadata`, `artifacts` |
| 29 | PoseBusters score endpoint | `curl -X POST http://localhost:8082/score -H 'Content-Type: application/json' -d '{"smiles": ["CCO"], "params": {"mode": "mol", "structure_paths": ["/shared/predictions/mol_0/mol_0_model_0.pdb"]}}'` | JSON with `scores` and per-check `metadata` |
| 30 | Shared volume | `docker compose exec boltz ls /shared && docker compose exec posebusters ls /shared` | Both containers can see the shared volume |
| 31 | Boltz2 MSA validation | `curl -X POST http://localhost:8091/score -H 'Content-Type: application/json' -d '{"smiles": ["CCO"], "params": {"protein_sequence": "MVLSPADKTN"}}'` | HTTP 400 with error about missing MSA config |
| 32 | Compose down | `docker compose down -v` | All services and volumes removed cleanly |

### Agent with External Scoring (Boltz + PoseBusters)

These tests verify the agent can interact with Boltz2 and PoseBusters via the MCP tools. Requires Docker services running (`docker compose up -d`).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 33 | Check external services | Ask: `What external scoring services are available?` | Agent calls `list_external_services`, reports Boltz and PoseBusters status |
| 34 | Create TTP with Boltz | Ask: `Create a TTP for binding affinity optimization of tyrosine kinase. Use Boltz for binding prediction and PoseBusters for pose quality validation (depends on Boltz) with protein sequence MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH and QED as a native objective.` | Agent creates TTP with both native and external objectives |
| 35 | Score with Boltz TTP | Ask: `Score aspirin using the TTP` | If Boltz is running: returns affinity scores, confidence metadata, and structure artifacts. If Boltz is down: returns native scores + note about skipped externals |
| 36 | Create pipeline TTP | Ask: `Create a TTP that uses Boltz for binding affinity and PoseBusters for pose quality validation (depends on Boltz), plus QED as native.` | TTP created with `depends_on` linking PoseBusters to Boltz |
| 37 | Score pipeline TTP | Ask: `Score CCO using the pipeline TTP` | Boltz runs first, structures passed to PoseBusters, all scores returned |

### Agent RL Optimization

These tests verify the agent can run RL optimization campaigns end-to-end. Requires REINVENT4 + GPU.

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 38 | Launch RL campaign | Ask: `Create a TTP with QED and SA score, then run an optimization with max_steps=20` | Agent creates TTP, calls `run_optimization`, returns job_id |
| 39 | Monitor running job | Ask: `What's the status of job <id>?` | Agent calls `get_optimization_status`, reports step count, scores, convergence status |
| 40 | Interpret diagnostics | Wait for 10+ steps, then ask: `How is the optimization going?` | Agent interprets diagnostics, discusses convergence and component health |
| 41 | Stop and adjust | Ask: `Stop the optimization and reduce the SA weight to 0.05` | Agent calls `stop_optimization`, then `update_ttp`, reports checkpoint saved |
| 42 | Resume from checkpoint | Ask: `Resume the optimization with the updated TTP` | Agent calls `run_optimization` with `resume_from` pointing to checkpoint |
| 43 | Extract results | After completion, ask: `Show me the top 10 candidates` | Agent calls `get_top_candidates`, displays molecules with component scores |
| 44 | List all jobs | Ask: `What optimization jobs have I run?` | Agent calls `list_jobs`, lists all tracked jobs with states |

### Agent RL Optimization with External Scoring (Boltz2 + PoseBusters)

These tests verify the agent can run RL optimization campaigns that include Boltz2 and PoseBusters in the scoring loop. Requires REINVENT4 + GPU + Docker services running (`docker compose up -d`).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 45 | Launch RL with full pipeline | Ask: `Create a TTP for lead optimization against tyrosine kinase with protein sequence MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH. Include QED as a native objective with weight 0.3, Boltz2 for binding affinity prediction as an external objective with weight 0.5 and use_msa_server enabled, and PoseBusters in dock mode for pose quality validation with weight 0.2 (depends on Boltz). Then run an RL optimization with max_steps=30 and batch_size=8.` | Agent creates TTP with native + external objectives and pipeline dependency, calls `run_optimization`, returns job_id. TOML config includes `ExternalRESTScorer` entries for both services. |
| 46 | Monitor external component health | After 5+ steps, ask: `How is the optimization going? Are the external scoring services working?` | Agent calls `get_optimization_status`, reports per-component health. Boltz and PoseBusters NaN rates visible in `component_health`. If services are healthy: NaN rate near 0%. If degraded: agent flags the specific failing component. |
| 47 | Interpret mixed diagnostics | After 15+ steps, ask: `Give me a detailed analysis of the optimization. How are the native vs external objectives performing?` | Agent interprets diagnostics showing convergence status, per-component trajectories for QED (native) vs Boltz (external) vs PoseBusters (external). Discusses whether molecules are improving on binding affinity alongside drug-likeness. |
| 48 | Stop, adjust weights, resume | Ask: `Stop the optimization. The Boltz binding scores look promising but PoseBusters scores are low. Increase the PoseBusters weight to 0.4 and reduce QED to 0.1, then resume.` | Agent calls `stop_optimization`, `update_ttp` with new weights, then `run_optimization` with `resume_from` pointing to checkpoint. New TOML reflects updated weights. |
| 49 | Top candidates with full breakdowns | After completion, ask: `Show me the top 10 candidates with their binding affinity, pose quality, and QED breakdowns.` | Agent calls `get_top_candidates`, displays ranked molecules with per-component scores for QED, Boltz (affinity), and PoseBusters (pose quality) including both transformed and raw values. |
| 50 | External service failure mid-run | While RL is running, stop Boltz (`docker compose stop boltz`). Then ask: `How is the optimization going?` | Agent reports Boltz component as `"failing"` with high NaN rate in diagnostics. PoseBusters also degraded (depends on Boltz for structure artifacts). Agent warns about external service issues and recommends checking service health or stopping the run. |
| 51 | Recover and resume after service restart | Restart Boltz (`docker compose start boltz`), stop the degraded run, then ask: `Resume the optimization now that Boltz is back up.` | Agent verifies services are healthy via `list_external_services`, then resumes from checkpoint. Subsequent steps show Boltz and PoseBusters NaN rates recovering to normal. |

### Agent Transfer Learning

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 52 | Run transfer learning | Ask: `Fine-tune a model on these SMILES: CCO, c1ccccc1, CC(=O)O, CCCCCC, c1ccc(O)cc1` | Agent calls `run_transfer_learning` with inline SMILES |
| 53 | Use TL model in RL | After TL completes, ask: `Now run an optimization using the fine-tuned model` | Agent uses the TL output model path in `run_optimization` |

### Slurm Deployment Test

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 54 | Setup script | `bash services/slurm/setup.sh` | Virtualenv created, all 4 packages installed, `dd-mcp` and `dd-agent` on PATH |
| 55 | Interactive session | `srun --partition=batch --cpus-per-task=4 --mem=16G --time=1:00:00 --pty bash services/slurm/run_interactive.sh` | Agent starts on compute node |
| 56 | Batch job | `echo "List all scoring components" \| sbatch services/slurm/agent.sbatch` | Job output file contains component list |

### Scorer Measurement (requires a campaign's worth of logs, or Docker + GPU)

These produce the two numbers Boltz replica counts are derived from. Neither
needs a dedicated benchmark run: the first reads history, the second is one
`/score` call per arm.

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 57 | Harvest latency and cache hit rate | `./k8s/scripts/analyze-scoring-latency.py data/logs/external_scoring.jsonl` | Per-service latency per batch and per molecule, the breakdown by batch size, the cache hit rate, and the replica counts that follow. Exit 2 when the log has no usable Boltz call |
| 58 | Measure `t_boltz_per_batch` | `docker compose up -d boltz && ./k8s/scripts/ab-boltz-kernels.py --baseline-only` | 16/16 scored against BRD4 BD1 using `brd4_bd1.a3m`, with the per-batch and per-molecule time and the replica count it implies. Measured 386 s on MI300 |
| 59 | A/B the optimized kernels | `./k8s/scripts/ab-boltz-kernels.py --restart-command 'docker compose up -d boltz'` | On ROCm the kernels arm fails with `ModuleNotFoundError: cuequivariance_torch` and the script reports keeping `BOLTZ_NO_KERNELS=1`. On a CUDA node both arms score 16/16 and a speedup is reported |

### Split Scorer Deployments (requires a cluster)

After `apply.sh`, the scorers are separate pods reached over cluster DNS rather
than loopback inside one pod.

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 60 | All six Deployments are up | `kubectl -n "${PROJECT}" get deployments,pods --selector app.kubernetes.io/part-of=dd-agentic --output wide` | Six Deployments `1/1`, every pod on the `--node` host, and two `amd.com/gpu` requested in total. Matching no pods means the `labels` transformer lost `includeTemplates` |
| 61 | Every Service has an endpoint | `kubectl -n "${PROJECT}" get endpointslices --selector app.kubernetes.io/component=scorer` | Five scorer Services, each with exactly one ready address, on ports 8091/8082/8083/8084/8085 |
| 62 | Scorers resolve from the agent pod | The one-liner under [Check scorer health](k8s/README.md#check-scorer-health) | `{"status": "ok"}` from all five cluster-DNS URLs. Boltz adds `no_kernels` only if `boltz2-wrapper:local` was rebuilt since that field landed |
| 63 | Scoring still works across pods | `kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c 'import json; from reinvent4_mcp.tools.external import list_external_services; print(json.dumps(list_external_services(), indent=2))'` | All five report `status: ok` against the ConfigMap's `dd-agentic-*` URLs, which exercises the same config `ExternalRESTScorer` reads |

### Cross-Pod Artifact Handoff (requires a cluster + GPU)

The riskiest assumption in the split: Boltz returns absolute `/shared` paths
and PoseBusters and the interaction profiler re-open them, so the consumers
must see the producer's filesystem. Proven in-pod already; this is the same
handoff across pod boundaries. See
[Prove the structure handoff across pods](k8s/README.md#prove-the-structure-handoff-across-pods).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 64 | Handoff across pods | `kubectl -n "${PROJECT}" exec -i deployment/dd-agentic -c dd-agent -- python - --in-cluster < k8s/scripts/verify-artifact-handoff.py` | `PROVEN`, exit 0: two structures written under `/shared`, both consumers scoring real values off them, the skipped molecule still `--` in every column. Without `--namespace` the verdict says pod identity was not checked |
| 65 | Same handoff, pods named | Three port-forwards (8091/8082/8083), then `./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}"` | The same, plus the pod and node that answered for each service, and PoseBusters' extracted `_lig.sdf` visible from the Boltz pod. Exit 2 if any two share a pod — that is the in-pod proof, not this one |
| 66 | Re-prove after a storage change | `./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}" --structure <path from run 64> --smiles <its SMILES>` | Same verdict with no GPU time: the consumers re-open a pose already on the volume |
| 67 | Same handoff on Compose | `docker compose up -d boltz posebusters interaction-profiler && ./k8s/scripts/verify-artifact-handoff.py` | `PROVEN` against the published 8091/8082/8083. Three separate containers over one host directory, so it rehearses the mechanism before a cluster is involved |

### Boltz Gateway Placement (requires a cluster + 2 GPUs)

Each replica serves one prediction at a time, so the gateway has to place work
on an idle one and queue the surplus. See
[Place batches on idle Boltz replicas](k8s/README.md#place-batches-on-idle-boltz-replicas).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 68 | Gateway discovered the replicas | Port-forward 8404, then `curl -s "http://localhost:8404/stats;csv" \| cut -d, -f1,2,18` | One `UP` row per Boltz replica; the remaining slots sit in `MAINT (resolution)` until DNS fills them, which is how scaling needs no config change |
| 69 | Least-busy placement | `./k8s/scripts/verify-boltz-placement.py --namespace "${PROJECT}"` | `PROVEN`: each concurrent batch on its own replica, peak in-flight 1 per replica, the surplus queued. Measured at M=2: 2.11x speedup, third batch waited 197.6 s |
| 70 | Queueing is not failure | `kubectl -n "${PROJECT}" logs deployment/dd-agentic-boltz-gateway \| tail` | `%Tw` non-zero and `queue=0/1` on the surplus request, with status `200`. A `503` means `timeout queue` is below the prediction time |
| 71 | Scoring still works through the gateway | `./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}"` | `PROVEN` with Boltz reported as "2 replicas" and the extracted ligands visible from all of them |
| 72 | The RL path dials the gateway | `kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c "from reinvent4_mcp.tools.external import list_external_services as l; [print('%-22s %-10s %s' % (n, i['status'], i['url'])) for n, i in l().items()]"` | `boltz ok http://dd-agentic-boltz-gateway:8091/score` and the four CPU scorers direct — the ConfigMap path `ExternalRESTScorer` actually takes |
| 73 | Scaling needs no proxy change | `kubectl -n "${PROJECT}" scale deployment/dd-agentic-boltz --replicas=3`, then watch the stats CSV | A third slot leaves `MAINT (resolution)` and turns `UP` **about 30 s after the pod is Ready** (10 s DNS hold + `rise 2 inter 10s`). Checking immediately shows the old count and looks like a failure |

### REINVENT Runner (requires a cluster + GPU)

The RL lifecycle can only really be exercised where REINVENT4 is installed. See
[Move RL off the agent](k8s/README.md#move-rl-off-the-agent-reinvent-runner).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 78 | Runner is healthy | `kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c "import urllib.request; print(urllib.request.urlopen('http://dd-agentic-reinvent-runner:8086/health').read().decode())"` | `reinvent_importable: true` and `max_concurrent_jobs` matching the config. A false there means the image lacks REINVENT4, which is a different problem from the service being down |
| 79 | Registries over HTTP | `GET /components`, `/models`, `/transforms` | Measured: 61 scoring components, 9 priors, 8 transforms — the real registries, not stubs |
| 80 | A real job over HTTP | `POST /jobs` with a QED-only scorer config and `max_steps: 2`, then poll `GET /jobs/{id}` | `running` then `completed`, and the job appears in `GET /jobs`. Proves REINVENT accepts the generated TOML and the subprocess lifecycle survives the move |
| 81 | Images are split | `docker images dd-agent` after `build-images.sh` | **164 MB** on `python:3.12-slim`, and `docker run dd-agent:local python -c "import importlib.util as u; print(u.find_spec('reinvent'), u.find_spec('torch'))"` prints `None None`. The build fails outright if either is present |
| 82 | The seam is invisible | Set `runner.enabled`, then drive the same job through `get_job_manager()` | Returns `RemoteJobManager`; same result as the direct call, and the third concurrent job is refused naming the two to stop — the slot limit holds across the HTTP boundary |

### Boltz Gateway: Cache and Coalescing (requires a cluster + GPU)

The gateway reduces how much prediction is asked for at all. See
[Cache results and coalesce batches](k8s/README.md#cache-results-and-coalesce-batches).

| # | Test | How to run | Expected result |
|---|------|-----------|-----------------|
| 74 | The chain is wired | `kubectl -n "${PROJECT}" exec deployment/dd-agentic -c dd-agent -- python -c "import json,urllib.request; print(urllib.request.urlopen('http://dd-agentic-boltz-gateway:8091/health').read().decode())"` | `cache_enabled: true`, `upstream` the proxy, `healthy_replicas` matching the Boltz replica count, `upstream_reachable: true` |
| 75 | Cold then warm | `./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}"` twice | Measured 3 m 35 s then **12 s**, with identical affinities and the *same* structure paths, and both consumers re-opening the cached poses. The molecule Boltz skipped hits too — a deterministic skip is cached |
| 76 | Hit rate is reported | `curl -s http://localhost:8091/stats` through a port-forward | `cache.hit_rate` 0.5 after one cold and one warm run, `coalescing.molecules_per_call`, `queue.*` and latency percentiles. This is the number §5.4's admission control has always had to assume |
| 77 | Submit and poll | `curl -s -X POST "http://localhost:8091/score?mode=async" -d @batch.json`, then `GET /result/{task_id}` | `202` with a `task_id`, `202` while queued or predicting, `200` with the scores once, then `404` — results are served once so the registry cannot grow without bound |

---

## Test Matrix Summary

| Tier | Tests | Needs REINVENT4 | Needs LLM API | Needs Docker | Needs GPU | Needs Slurm |
|------|-------|-----------------|---------------|--------------|-----------|-------------|
| 1 - Unit | 650+ | No | No | No | No | No |
| 2 - Integration | 39 | Yes | No | Partially | For RL/TL | No |
| 3 - E2E | 82 | Yes | Yes | Partially | For RL/TL | Partially |

### Recommended Test Sequence

1. **Local development**: Run Tier 1 (`python -m pytest tests/ -v`) after every change
2. **After setup on cluster**: Run Tier 2 integration tests to verify REINVENT4 wiring
3. **Docker services**: Build and start Boltz2 + PoseBusters services (`docker compose up -d`), then run Tier 2 external service tests
4. **RL / Transfer Learning**: Run Tier 2 optimization + TL tests on a GPU node with REINVENT4 installed
5. **Before demo/release**: Run Tier 3 E2E tests for the full agent experience including optimization campaigns and external scoring pipeline
