from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    CheckResult,
    FrameRange,
    FrameRate,
    Intent,
    Provenance,
    ReferenceAudioStructure,
    ReferenceMotionFrameV2,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionPassReceiptV2,
    ReferenceReconstructionSpecV2,
    Scenario,
    TechniqueApplicationRecord,
    TechniqueCompositionEvidenceV2,
    TechniquePacket,
    TimeWarpAnchorV2,
)
from aoa_editing.evals.fixtures import create_transfer_fixture
from aoa_editing.evals.transfer_v2 import run_technique_transfer_corpus_v2
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.knowledge.techniques import (
    portable_motion_effect_v2,
    upgrade_contain_reveal_technique_v2,
)

SOURCE_HASH = "1" * 64
REFERENCE_HASH = "2" * 64
SPEC_HASH = "3" * 64
REVISION = "a" * 40
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _provenance() -> Provenance:
    return Provenance(tool="synthetic", tool_version="1", deterministic=True)


def _base_packet() -> TechniquePacket:
    return TechniquePacket(
        id="technique_continuous_contain_reveal",
        revision=2,
        title="Continuous close-detail to whole-image reveal",
        intent="Settle a close detail into a contained whole still.",
        applicability=["One global still transform is sufficient."],
        contraindications=["Independent layer motion is required."],
        required_evidence=["image.statistics", "image.depth_layers"],
        allowed_operations=["contain", "virtual-camera"],
        rules=["Preserve source-only lineage."],
        heuristics=["Use normalized coordinates."],
        creative_variants=["Reverse the trajectory."],
        constraints=["No reference media in render lineage."],
        failure_modes=["Insufficient source pixels."],
        qc=["Remeasure motion."],
        positive_examples=["Textured illustration."],
        negative_examples=["Layered composition."],
        application_history=[
            TechniqueApplicationRecord(
                case_id="prior_reference",
                source_class="textured still",
                outcome="pass",
                evidence_path="runtime-eval:prior/reference.json",
            ),
            TechniqueApplicationRecord(
                case_id="prior_transfer",
                source_class="procedural landscape",
                outcome="pass",
                evidence_path="runtime-eval:prior/transfer.json",
            ),
        ],
        status="candidate",
        promotion_questions=["Does it transfer across source classes?"],
        provenance=_provenance(),
    )


def _matrix(
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    center_x: float,
    center_y: float,
    scale: float,
    rotation_degrees: float,
) -> list[list[float]]:
    fit = min(output_width / source_width, output_height / source_height)
    angle = math.radians(rotation_degrees)
    linear = fit * scale * np.asarray(
        (
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        ),
        dtype=np.float64,
    )
    source_center = np.asarray(
        (source_width / 2, source_height / 2),
        dtype=np.float64,
    )
    target = np.asarray((center_x * output_width, center_y * output_height))
    translation = target - linear @ source_center
    return [
        [float(linear[0, 0]), float(linear[0, 1]), float(translation[0])],
        [float(linear[1, 0]), float(linear[1, 1]), float(translation[1])],
        [0.0, 0.0, 1.0],
    ]


def _spec(duration: int = 41) -> ReferenceReconstructionSpecV2:
    width, height = 180, 320
    frames: list[ReferenceMotionFrameV2] = []
    for frame in range(duration):
        progress = frame / (duration - 1)
        smooth = progress * progress * (3 - 2 * progress)
        center_x = 0.31 + 0.19 * smooth + 0.006 * math.sin(math.pi * progress)
        center_y = 0.61 - 0.11 * smooth + 0.003 * math.sin(2 * math.pi * progress)
        scale = 1.8 - 0.8 * smooth
        rotation = -6.0 + 6.0 * smooth
        frames.append(
            ReferenceMotionFrameV2(
                frame=frame,
                time_seconds=frame / 25,
                selected_model="similarity",
                matrix_3x3=_matrix(
                    source_width=640,
                    source_height=640,
                    output_width=width,
                    output_height=height,
                    center_x=center_x,
                    center_y=center_y,
                    scale=scale,
                    rotation_degrees=rotation,
                ),
                visible_source_boundary=[
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                center_x=center_x,
                center_y=center_y,
                scale_relative_to_contain=scale,
                rotation_degrees=rotation,
                confidence=0.98,
                confidence_interval={
                    "center_normalized_radius": 0.0002,
                    "scale_relative_radius": 0.001,
                    "rotation_degrees_radius": 0.04,
                },
                match_count=100,
                inlier_count=98,
                inlier_ratio=0.98,
                reprojection_rmse=0.2,
                residual_flow_mean=0.1,
                residual_flow_p95=0.2,
                residual_valid_fraction=1.0,
                velocity={
                    "scale": -0.5,
                    "rotation": 3.75,
                    "center_x": 0.12,
                    "center_y": -0.07,
                },
                acceleration={
                    "scale": 0.0,
                    "rotation": 0.0,
                    "center_x": 0.0,
                    "center_y": 0.0,
                },
                jerk={
                    "scale": 0.0,
                    "rotation": 0.0,
                    "center_x": 0.0,
                    "center_y": 0.0,
                },
                model_scores={
                    "similarity": 0.2,
                    "affine": 0.199,
                    "homography": 0.198,
                },
                outlier=False,
                uncertainty=["pivot_translation_gauge"],
            )
        )
    return ReferenceReconstructionSpecV2(
        evidence_id="referencemotion_" + "b" * 32,
        source_sha256=SOURCE_HASH,
        reference_sha256=REFERENCE_HASH,
        readiness_revision=REVISION,
        motion_gate_revision=REVISION,
        duration_frames=duration,
        duration_seconds=duration / 25,
        frame_rate=FrameRate(numerator=25),
        width=width,
        height=height,
        audio=ReferenceAudioStructure(
            has_audio_stream=False,
            silent_for_full_duration=True,
        ),
        motion_model="similarity",
        motion_frames=frames,
        curve_representation={
            "kind": "dense_observed_matrix_curve",
            "sampling": "every_frame",
        },
        phase_model={
            "onset_frame": 2,
            "peak_velocity_frame": 20,
            "settle_frame": 40,
            "overshoot_detected": False,
            "channel_coupling": {
                "leading_explanation": (
                    "staggered_center_path_with_synchronized_scale_rotation"
                ),
                "center_phase_relation": "center_lags",
                "scale_rotation_progress_rmse": 0.001,
                "maximum_phase_offset": 0.2,
            },
            "twist_candidate": {
                "start_frame": 10,
                "peak_frame": 20,
                "end_frame": 30,
            },
        },
        transform_order={"supported_model": "similarity"},
        uncertainties=["Pivot is not identifiable."],
        alternative_explanations=["Translation and pivot are gauge-equivalent."],
        comparison_criteria={},
        artifacts={},
        provenance=_provenance(),
    )


def _selected_pass(
    spec: ReferenceReconstructionSpecV2,
) -> ReferenceReconstructionPassReceiptV2:
    last = spec.duration_frames - 1
    correction = ReferencePhaseCorrectionV2(
        spec_id=spec.id,
        spec_sha256=SPEC_HASH,
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        comparison_report_id="comparison_" + "c" * 32,
        comparison_report_path="comparison-v2.json",
        comparison_report_sha256="4" * 64,
        candidate_motion_report_id="motionrecovery_" + "d" * 32,
        candidate_motion_path="candidate-motion.json",
        candidate_motion_sha256="5" * 64,
        prior_candidate_sha256="6" * 64,
        duration_frames=spec.duration_frames,
        anchors=[
            TimeWarpAnchorV2(output_frame=0, source_frame=0.0),
            TimeWarpAnchorV2(output_frame=10, source_frame=10.0),
            TimeWarpAnchorV2(output_frame=30, source_frame=30.0),
            TimeWarpAnchorV2(output_frame=last, source_frame=float(last)),
        ],
        identity_ranges=[FrameRange(start=0, duration=spec.duration_frames)],
        protected_coupling_range=FrameRange(start=10, duration=21),
        previous_phase={"onset": 3, "peak_velocity": 20, "settle": 38},
        predicted_phase={"onset": 2, "peak_velocity": 20, "settle": 39},
        predicted_metrics={"all": "pass"},
        predicted_checks=[
            CheckResult(
                id="predicted",
                status="pass",
                summary="synthetic identity warp passes",
            )
        ],
        failed_check_ids=["temporal-phase-onset"],
        selection={"candidate_count": 1},
        provenance=_provenance(),
    )
    return ReferenceReconstructionPassReceiptV2(
        pass_kind="phase_compensated_matrix",
        sequence="5r",
        spec_id=spec.id,
        spec_path="reference-spec-v2.json",
        spec_sha256=SPEC_HASH,
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        project_id="project_" + "7" * 32,
        project_root="runtime-project",
        project_manifest="project.json",
        asset_id="asset_" + "8" * 32,
        frozen_spec_evidence="evidence.json",
        treatment_id="treatment_" + "9" * 32,
        treatment_path="treatment.json",
        proposed_decision_graph="proposed.json",
        version_id="version_" + "a" * 32,
        approved_edit_graph="approved.json",
        timeline_projection="timeline.json",
        profiles={},
        editable_exports={},
        hypotheses={},
        phase_correction=correction,
        phase_correction_path="phase-correction.json",
        phase_correction_evidence="phase-correction-evidence.json",
        provenance=_provenance(),
        overall="pass",
    )


def _upgraded_packet() -> TechniquePacket:
    spec = _spec()
    return upgrade_contain_reveal_technique_v2(
        _base_packet(),
        spec,
        _selected_pass(spec),
    )


def test_upgrade_retains_portable_c1_semantics_without_media_identity() -> None:
    packet = _upgraded_packet()
    serialized = packet.model_dump_json()
    effect = portable_motion_effect_v2(packet)

    assert packet.schema_version == "2.0.0"
    assert packet.revision == 3
    assert packet.status == "candidate"
    assert packet.camera_motion is None
    assert packet.portable_camera_motion_v2 is not None
    assert len(packet.portable_camera_motion_v2.centers) == 41
    assert effect.motion.kind == "decomposed"
    assert effect.motion.position.continuity == "c1"
    assert effect.motion.scale.continuity == "c1"
    assert effect.motion.rotation.continuity == "c1"
    assert SOURCE_HASH not in serialized
    assert REFERENCE_HASH not in serialized
    assert "/home/" not in serialized
    assert "/srv/" not in serialized


def test_application_boundary_refuses_missing_untrusted_and_wrong_aspect(
    tmp_path: Path,
) -> None:
    packet = _upgraded_packet()
    source = create_transfer_fixture(tmp_path / "independent-source.png")
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    editing = EditingService(store)
    project = editing.create_project(
        "Applicability boundary",
        Intent(
            text="Test portable motion applicability.",
            scenario=Scenario.STILL_MOTION,
        ),
    )
    asset, _ = editing.ingest(project.id, source)
    AnalysisService(store).analyze(project.id, asset.id)

    missing = editing.propose_technique_v2(
        project.id,
        asset_id=asset.id,
        technique=packet,
        width=180,
        height=320,
    )
    assert missing.decision.refusal_code == "missing_required_evidence"
    assert missing.treatment is None
    assert store.load_project(project.id).current_version_id is None

    editing.record_technique_composition_v2(
        project.id,
        asset_id=asset.id,
        assessment=TechniqueCompositionEvidenceV2(
            source_class="independent source",
            single_global_transform_sufficient=True,
            independent_layer_motion_required=False,
            authority="analyzer_proposal",
            rationale="An analyzer proposes, but cannot authorize, one global transform.",
        ),
    )
    untrusted = editing.propose_technique_v2(
        project.id,
        asset_id=asset.id,
        technique=packet,
        width=180,
        height=320,
    )
    assert untrusted.decision.refusal_code == "inapplicable_composition"
    assert untrusted.treatment is None

    editing.record_technique_composition_v2(
        project.id,
        asset_id=asset.id,
        assessment=TechniqueCompositionEvidenceV2(
            source_class="independent source",
            single_global_transform_sufficient=True,
            independent_layer_motion_required=False,
            authority="human_assertion",
            rationale="A human confirms that one global transform is sufficient.",
        ),
    )
    wrong_aspect = editing.propose_technique_v2(
        project.id,
        asset_id=asset.id,
        technique=packet,
        width=320,
        height=320,
    )
    assert wrong_aspect.decision.refusal_code == "output_aspect_mismatch"
    assert wrong_aspect.treatment is None
    assert store.load_project(project.id).current_version_id is None

    eligible = editing.propose_technique_v2(
        project.id,
        asset_id=asset.id,
        technique=packet,
        width=180,
        height=320,
    )
    assert eligible.decision.outcome == "eligible"
    assert eligible.treatment is not None
    assert store.load_project(project.id).current_version_id is not None


def test_transfer_corpus_applies_five_classes_and_refuses_two(tmp_path: Path) -> None:
    photograph = create_transfer_fixture(tmp_path / "independent-photograph-fixture.png")
    settings = Settings.for_home(tmp_path / "host")
    report = run_technique_transfer_corpus_v2(
        _upgraded_packet(),
        tmp_path / "transfer-corpus",
        photograph_path=photograph,
        photograph_provenance={
            "owner": "test-suite",
            "authority": "fixture",
            "license_authority": "project-generated-evaluation-fixture",
            "media_type": "photograph-like integration fixture",
        },
        runtime_settings=settings,
        canvas_width=180,
        canvas_height=320,
        forbidden_hashes=[SOURCE_HASH, REFERENCE_HASH],
        implementation_revision=REVISION,
        git_clean=True,
    )

    assert report.overall == "pass"
    assert report.mandatory_skips == 0
    assert len(report.cases) == 7
    assert sum(item.application_outcome == "eligible" for item in report.cases) == 5
    assert sum(item.application_outcome == "refused" for item in report.cases) == 2
    by_id = {item.case_id: item for item in report.cases}
    assert (
        by_id["low_resolution"].applicability.refusal_code
        == "insufficient_source_resolution"
    )
    assert (
        by_id["inapplicable_composition"].applicability.refusal_code
        == "inapplicable_composition"
    )
    assert all(item.deterministic_replay["equivalent"] for item in report.cases)
    assert report.output_technique_revision == 4
    packet_path = Path(report.artifacts["updated_candidate_packet"])
    output_packet = TechniquePacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    assert output_packet.status == "candidate"
    assert len(output_packet.application_history) == 9


def test_checked_in_transfer_candidate_and_upgrade_input_are_source_neutral() -> None:
    current_path = (
        REPOSITORY_ROOT
        / "editing-knowledge"
        / "candidates"
        / "continuous-contain-reveal.json"
    )
    history_path = (
        REPOSITORY_ROOT
        / "editing-knowledge"
        / "history"
        / "continuous-contain-reveal-v1-revision2.json"
    )
    current_text = current_path.read_text(encoding="utf-8")
    history_text = history_path.read_text(encoding="utf-8")
    current = TechniquePacket.model_validate_json(current_text)
    history = TechniquePacket.model_validate_json(history_text)

    assert current.schema_version == "2.0.0"
    assert current.revision == 4
    assert current.status == "candidate"
    assert current.portable_camera_motion_v2 is not None
    assert len(current.application_history) == 9
    assert history.schema_version == "1.0.0"
    assert history.revision == 2
    assert history.status == "candidate"
    assert len(history.application_history) == 2
    for serialized in (current_text, history_text):
        assert SOURCE_HASH not in serialized
        assert REFERENCE_HASH not in serialized
        assert "/home/" not in serialized
        assert "/srv/" not in serialized
