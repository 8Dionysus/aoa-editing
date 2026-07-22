from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aoa_editing.agent import AgentProtocol, AgentRequest
from aoa_editing.api.app import create_app
from aoa_editing.application.reference_workspace import (
    MotionCorrectionReviewError,
    MotionCorrectionService,
    ReferenceWorkspaceService,
)
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    AdjustPivotCurveCorrectionV2,
    Clip,
    EditPatch,
    EvidenceRecord,
    FrameRange,
    FrameRate,
    Intent,
    Matrix3x3V2,
    MatrixMotionCurveV2,
    MatrixMotionKeyframeV2,
    MatrixTransformMotionV2,
    MotionCorrectionContextV2,
    MotionCorrectionReviewDecisionV2,
    MotionTimeV2,
    MovePhaseBoundaryCorrectionV2,
    PatchOperation,
    Provenance,
    ReferenceWorkspaceArtifactV2,
    ReferenceWorkspaceRegistrationV2,
    Scenario,
    Track,
    TrackKind,
    TransformEffectV2,
    new_id,
)
from aoa_editing.domain.motion_corrections import motion_effect_sha256
from aoa_editing.infrastructure.store import ProjectStore

REVISION = "a" * 40


def _readiness(_settings: Settings) -> dict[str, object]:
    return {"overall": "pass", "git_revision": REVISION}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(frame: int, duration: int) -> Matrix3x3V2:
    progress = frame / (duration - 1)
    eased = progress * progress * (3.0 - 2.0 * progress)
    scale = 1.45 - 0.45 * eased
    rotation = math.radians(-8.0 * (1.0 - eased))
    a = scale * math.cos(rotation)
    b = -scale * math.sin(rotation)
    center_x = 32.0 + 4.0 * eased + math.sin(math.pi * progress)
    center_y = 32.0 - 3.0 * eased
    tx = center_x - a * 32.0 - b * 32.0
    ty = center_y - (-b * 32.0 + a * 32.0)
    return ((a, b, tx), (-b, a, ty), (0.0, 0.0, 1.0))


def _effect(duration: int) -> TransformEffectV2:
    return TransformEffectV2(
        motion=MatrixTransformMotionV2(
            matrix=MatrixMotionCurveV2(
                keyframes=[
                    MatrixMotionKeyframeV2(
                        time=MotionTimeV2(frame=frame),
                        matrix_3x3=_matrix(frame, duration),
                    )
                    for frame in range(duration)
                ]
            )
        )
    )


def _fixture(
    tmp_path: Path,
) -> tuple[ProjectStore, ReferenceWorkspaceRegistrationV2, Path]:
    settings = Settings.for_home(tmp_path / "editing-home")
    store = ProjectStore(settings)
    editing = EditingService(store)
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "#c47b24").save(source)
    project = editing.create_project(
        "Reference correction fixture",
        Intent(text="Understand and refine motion", scenario=Scenario.REFERENCE_RECONSTRUCT),
    )
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"sealed reference fixture")
    reference_sha256 = _sha(reference)
    editing.seal_reference(project.id, reference_sha256)
    asset, _probe = editing.ingest(project.id, source)
    duration = 41
    base = editing.initialize_timeline(
        project.id,
        duration_frames=duration,
        width=64,
        height=64,
        frame_rate=FrameRate(numerator=25),
    )
    effect = _effect(duration)
    version = editing.apply_patch(
        project.id,
        EditPatch(
            project_id=project.id,
            base_version_id=base.id,
            operations=[
                PatchOperation(
                    op="replace",
                    path="/tracks",
                    value=[
                        Track(
                            kind=TrackKind.VIDEO,
                            name="Reference motion",
                            clips=[
                                Clip(
                                    asset_id=asset.id,
                                    timeline_range=FrameRange(
                                        start=0,
                                        duration=duration,
                                    ),
                                    effects=[effect],
                                )
                            ],
                        ).model_dump(mode="json")
                    ],
                )
            ],
            rationale="Create a canonical matrix-motion version.",
        ),
        "Approve reference motion fixture",
    )

    evaluation = settings.evals_root / "reference-workspace-fixture"
    evaluation.mkdir(parents=True)
    document_paths = {}
    for name in (
        "study",
        "reference-evidence",
        "spec",
        "comparison",
        "candidate-motion",
    ):
        path = evaluation / f"{name}.json"
        path.write_text(f'{{"fixture":"{name}"}}\n', encoding="utf-8")
        document_paths[name] = path
    diagnostic = evaluation / "diagnostic.png"
    Image.new("RGB", (8, 8), "#202830").save(diagnostic)
    candidate = (
        store.project_path(project.id)
        / "renders"
        / version.id
        / "final"
        / "video.mp4"
    )
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate fixture")
    candidate_sha256 = _sha(candidate)
    baseline = EvidenceRecord(
        project_id=project.id,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        kind="reference.motion_interpretation.v2",
        payload={"workspace_id": "pending"},
        provenance=Provenance(tool="fixture", tool_version="1"),
    )
    workspace_id = new_id("referenceworkspace")
    baseline = baseline.model_copy(
        update={"payload": {"workspace_id": workspace_id}}
    )
    store.save_evidence(baseline)
    required_roles = (
        "side_by_side",
        "aligned_difference",
        "transform_curves",
        "velocity",
        "acceleration",
        "jerk",
        "pivot",
        "phase_contact_sheet",
        "frame_overlay",
        "optical_flow_residual",
        "motion_error",
        "derivative_error",
    )
    artifacts = {
        role: ReferenceWorkspaceArtifactV2(
            role=role,
            scope="editing_home",
            relative_path=str(diagnostic.relative_to(settings.editing_home)),
            sha256=_sha(diagnostic),
            media_type="image/png",
            authority=(
                "comparison_diagnostic"
                if role
                in {
                    "side_by_side",
                    "aligned_difference",
                    "motion_error",
                    "derivative_error",
                }
                else "reference_evidence"
            ),
            reference_derived=True,
        )
        for role in required_roles
    }
    artifacts["reference_media"] = ReferenceWorkspaceArtifactV2(
        role="reference_media",
        scope="reference_binding",
        sha256=reference_sha256,
        media_type="video/mp4",
        authority="reference_evidence",
        reference_derived=True,
    )
    artifacts["candidate_media"] = ReferenceWorkspaceArtifactV2(
        role="candidate_media",
        scope="project",
        relative_path=str(candidate.relative_to(store.project_path(project.id))),
        sha256=candidate_sha256,
        media_type="video/mp4",
        authority="candidate_render",
        reference_derived=False,
    )
    workspace = ReferenceWorkspaceRegistrationV2(
        id=workspace_id,
        project_id=project.id,
        base_version_id=version.id,
        asset_id=asset.id,
        target_effect_path="/tracks/0/clips/0/effects/0",
        target_effect_sha256=motion_effect_sha256(effect),
        baseline_evidence_id=baseline.id,
        study_id=new_id("reconstructionstudy"),
        study_path=str(document_paths["study"].relative_to(settings.editing_home)),
        study_sha256=_sha(document_paths["study"]),
        reference_evidence_id=new_id("referencemotion"),
        reference_evidence_path=str(
            document_paths["reference-evidence"].relative_to(settings.editing_home)
        ),
        reference_evidence_sha256=_sha(document_paths["reference-evidence"]),
        spec_id=new_id("referencespec"),
        spec_path=str(document_paths["spec"].relative_to(settings.editing_home)),
        spec_sha256=_sha(document_paths["spec"]),
        comparison_id=new_id("comparison"),
        comparison_path=str(
            document_paths["comparison"].relative_to(settings.editing_home)
        ),
        comparison_sha256=_sha(document_paths["comparison"]),
        candidate_motion_id=new_id("motionrecovery"),
        candidate_motion_path=str(
            document_paths["candidate-motion"].relative_to(settings.editing_home)
        ),
        candidate_motion_sha256=_sha(document_paths["candidate-motion"]),
        source_sha256=asset.sha256,
        reference_sha256=reference_sha256,
        candidate_sha256=candidate_sha256,
        readiness_revision=REVISION,
        frame_count=duration,
        frame_rate=FrameRate(numerator=25),
        width=64,
        height=64,
        phase_markers={
            "onset": 2,
            "twist_start": 8,
            "twist_peak": 20,
            "twist_end": 30,
            "settle": 40,
        },
        pivot_identifiable=False,
        artifacts=artifacts,
        provenance=Provenance(tool="fixture", tool_version="1"),
    )
    store.save_reference_workspace(workspace)
    store.bind_reference_media(reference_sha256, reference)
    return store, workspace, reference


def test_language_preview_is_immutable_and_individual_review_uses_normal_patch_flow(
    tmp_path: Path,
) -> None:
    store, workspace, _reference = _fixture(tmp_path)
    service = MotionCorrectionService(store, readiness_guard=_readiness)
    version_count = len(store.list_versions(workspace.project_id))

    proposal = service.preview_language(
        workspace.project_id,
        workspace.id,
        "переход должен мягче подкручиваться и чуть позже оседать",
    )

    assert [item.operation.kind for item in proposal.items] == [
        "adjust_tangent",
        "adjust_rotation_coupling",
        "retime_settle",
    ]
    assert all(item.executable and item.diff is not None for item in proposal.items)
    assert len(store.list_versions(workspace.project_id)) == version_count
    decisions = [
        MotionCorrectionReviewDecisionV2(
            correction_id=item.id,
            decision="approve" if index in {0, 2} else "reject",
            rationale="Human editor selected this bounded interpretation.",
        )
        for index, item in enumerate(proposal.items)
    ]
    review = service.review(workspace.project_id, proposal.id, decisions)

    assert review.outcome == "applied"
    assert review.applied_patch_id is not None
    assert review.created_version_id is not None
    assert len(store.list_versions(workspace.project_id)) == version_count + 1
    created = store.load_version(workspace.project_id, review.created_version_id)
    assert created.parent_version_id == proposal.base_version_id
    assert created.applied_patch_id == review.applied_patch_id
    graph = store.load_decision_graph(
        workspace.project_id,
        created.decision_graph_id or "",
    )
    assert graph.state == "approved"
    human = next(
        item
        for item in store.list_evidence(workspace.project_id)
        if item.id == review.human_evidence_id
    )
    assert human.authority.value == "human-correction"
    assert human.payload["accepted_correction_ids"] == [
        proposal.items[0].id,
        proposal.items[2].id,
    ]
    assert human.payload["rejected_correction_ids"] == [proposal.items[1].id]


def test_unidentifiable_pivot_is_visible_but_cannot_cross_review_boundary(
    tmp_path: Path,
) -> None:
    store, workspace, _reference = _fixture(tmp_path)
    service = MotionCorrectionService(store, readiness_guard=_readiness)
    proposal = service.preview_operations(
        workspace.project_id,
        workspace.id,
        [
            AdjustPivotCurveCorrectionV2(
                delta_x=0.05,
                delta_y=0,
                affected_range=FrameRange(start=8, duration=23),
            )
        ],
    )

    assert proposal.items[0].executable is False
    assert "pivot is not identifiable" in proposal.items[0].blockers[0]
    approval = [
        MotionCorrectionReviewDecisionV2(
            correction_id=proposal.items[0].id,
            decision="approve",
            rationale="Try to approve the unsupported interpretation.",
        )
    ]
    with pytest.raises(MotionCorrectionReviewError, match="cannot be approved"):
        service.review(workspace.project_id, proposal.id, approval)
    rejection = [
        MotionCorrectionReviewDecisionV2(
            correction_id=proposal.items[0].id,
            decision="reject",
            rationale="Keep the observable matrix and reject fabricated pivot data.",
        )
    ]
    before = len(store.list_versions(workspace.project_id))
    review = service.review(workspace.project_id, proposal.id, rejection)
    assert review.outcome == "rejected"
    assert review.created_version_id is None
    assert len(store.list_versions(workspace.project_id)) == before


def test_workspace_artifact_routes_are_hash_verified_and_readiness_gated(
    tmp_path: Path,
) -> None:
    store, workspace, reference = _fixture(tmp_path)
    service = ReferenceWorkspaceService(store, readiness_guard=_readiness)

    resolved_reference, media_type = service.resolve_artifact(
        workspace.project_id,
        workspace.id,
        "reference_media",
    )
    assert resolved_reference == reference.resolve()
    assert media_type == "video/mp4"
    candidate, _ = service.resolve_artifact(
        workspace.project_id,
        workspace.id,
        "candidate_media",
    )
    candidate.write_bytes(b"changed candidate")
    with pytest.raises(Exception, match="missing or changed"):
        service.resolve_artifact(
            workspace.project_id,
            workspace.id,
            "candidate_media",
        )

    sealed = ReferenceWorkspaceService(
        store,
        readiness_guard=lambda _settings: {
            "overall": "pass",
            "git_revision": "invalid",
        },
    )
    with pytest.raises(Exception, match="no valid Git revision"):
        sealed.resolve_artifact(
            workspace.project_id,
            workspace.id,
            "reference_media",
        )
    failed = ReferenceWorkspaceService(
        store,
        readiness_guard=lambda _settings: {
            "overall": "fail",
            "git_revision": REVISION,
        },
    )
    with pytest.raises(Exception, match="requires passing readiness"):
        failed.resolve_artifact(
            workspace.project_id,
            workspace.id,
            "reference_media",
        )


def test_phase_models_and_human_review_fail_closed_on_crossed_boundaries(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="strictly ordered"):
        MotionCorrectionContextV2(
            duration_frames=41,
            onset_frame=2,
            twist_start_frame=8,
            twist_peak_frame=8,
            twist_end_frame=30,
            settle_frame=40,
            pivot_identifiable=False,
        )

    store, workspace, _reference = _fixture(tmp_path)
    invalid_workspace = workspace.model_dump(mode="json")
    invalid_workspace["phase_markers"]["twist_peak"] = 8
    with pytest.raises(ValueError, match="strictly ordered"):
        ReferenceWorkspaceRegistrationV2.model_validate(invalid_workspace)

    service = MotionCorrectionService(store, readiness_guard=_readiness)
    proposal = service.preview_operations(
        workspace.project_id,
        workspace.id,
        [
            MovePhaseBoundaryCorrectionV2(
                boundary="twist_peak",
                original_frame=20,
                delta_frames=12,
                affected_range=FrameRange(start=8, duration=33),
            )
        ],
    )
    with pytest.raises(
        MotionCorrectionReviewError,
        match="break canonical phase ordering",
    ):
        service.review(
            workspace.project_id,
            proposal.id,
            [
                MotionCorrectionReviewDecisionV2(
                    correction_id=proposal.items[0].id,
                    decision="approve",
                    rationale="Deliberately test a crossed phase boundary.",
                )
            ],
        )


def test_http_and_agent_transports_share_the_motion_correction_service(
    tmp_path: Path,
) -> None:
    store, workspace, reference = _fixture(tmp_path)
    client = TestClient(
        create_app(store.settings, readiness_guard=_readiness)
    )
    detail = client.get(f"/api/projects/{workspace.project_id}")
    assert detail.status_code == 200
    assert detail.json()["reference_workspaces"][0]["id"] == workspace.id
    artifact = client.get(
        f"/api/projects/{workspace.project_id}/reference-workspaces/"
        f"{workspace.id}/artifacts/reference_media"
    )
    assert artifact.status_code == 200
    assert artifact.content == reference.read_bytes()
    response = client.post(
        f"/api/projects/{workspace.project_id}/reference-workspaces/"
        f"{workspace.id}/corrections/preview-language",
        json={
            "command": "мягче подкручиваться и позже оседать",
            "version_id": workspace.base_version_id,
        },
    )
    assert response.status_code == 201
    assert [item["operation"]["kind"] for item in response.json()["items"]] == [
        "adjust_tangent",
        "adjust_rotation_coupling",
        "retime_settle",
    ]

    protocol = AgentProtocol(store.settings, readiness_guard=_readiness)
    listed = protocol.execute(
        AgentRequest(
            id=1,
            method="reference.workspace.list",
            params={"project_id": workspace.project_id},
        )
    )
    assert listed["ok"] is True
    assert listed["result"][0]["id"] == workspace.id
    previewed = protocol.execute(
        AgentRequest(
            id=2,
            method="reference.correction.preview_language",
            params={
                "project_id": workspace.project_id,
                "workspace_id": workspace.id,
                "version_id": workspace.base_version_id,
                "command": "мягче подкручиваться и позже оседать",
            },
        )
    )
    assert previewed["ok"] is True
    assert [
        item["operation"]["kind"] for item in previewed["result"]["items"]
    ] == [
        "adjust_tangent",
        "adjust_rotation_coupling",
        "retime_settle",
    ]


def test_reference_workspace_decision_is_accepted_and_discoverable() -> None:
    root = Path(__file__).parents[1]
    decision = (
        root
        / "docs"
        / "decisions"
        / "ADR-0014-reference-workspace-and-human-motion-corrections.md"
    )
    text = decision.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    index = (root / "docs" / "decisions" / "README.md").read_text(encoding="utf-8")

    assert "- Status: accepted" in text
    assert "revision-bound readiness" in text
    assert "A human must approve or reject" in text
    assert "`EditPatch` through `EditingService`" in text
    assert "forbidden as render inputs" in normalized
    assert f"({decision.name})" in index
