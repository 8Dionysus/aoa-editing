const state = {
  projects: [],
  bundle: null,
  tab: "edit",
  busy: false,
  languagePreview: null,
  referenceBundle: null,
  referenceError: null,
  referenceFrame: 0,
  referenceChannel: "rotation_degrees",
  referenceDerivative: "value",
  motionProposal: null,
  anatomyPlanId: null,
  anatomyFocusedPlans: [],
  anatomyEstimate: null,
};
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const short = (value, size = 10) => value ? `${value.slice(0, size)}…` : "—";
const seconds = (frames, rate) => (frames / (rate.numerator / rate.denominator)).toFixed(2);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(payload.detail || payload || `HTTP ${response.status}`);
  return payload;
}

function setBusy(value, label = "Выполняю операцию…") {
  state.busy = value;
  $("#busy").classList.toggle("hidden", !value);
  $("#busy span").textContent = label;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.className = "toast", 3600);
}

async function guarded(label, action) {
  if (state.busy) return;
  setBusy(true, label);
  try { return await action(); }
  catch (error) { console.error(error); toast(error.message, true); }
  finally { setBusy(false); }
}

async function loadHealth() {
  try {
    const report = await api("/api/doctor");
    $("#health-dot").className = `health-dot ${report.ok ? "ok" : "fail"}`;
    $("#health-label").textContent = report.ok ? "Среда готова" : "Нужна настройка";
  } catch (_) {
    $("#health-dot").className = "health-dot fail";
    $("#health-label").textContent = "Сервис недоступен";
  }
}

async function loadProjects(selectId = null) {
  state.projects = await api("/api/projects");
  renderProjects();
  if (selectId) await selectProject(selectId);
  else if (state.bundle) {
    const exists = state.projects.some((item) => item.id === state.bundle.project.id);
    if (exists) await selectProject(state.bundle.project.id);
  }
}

function renderProjects() {
  $("#project-list").innerHTML = state.projects.map((project) => `
    <button class="project-button ${state.bundle?.project.id === project.id ? "active" : ""}" data-project="${project.id}">
      <span class="dot"></span><div><strong>${esc(project.name)}</strong><small>${esc(project.intent.scenario)}</small></div>
    </button>`).join("") || `<p class="muted">Пока пусто</p>`;
}

async function selectProject(id) {
  if (state.bundle?.project.id !== id) {
    state.languagePreview = null;
    state.referenceFrame = 0;
    state.motionProposal = null;
    state.anatomyPlanId = null;
    state.anatomyFocusedPlans = [];
    state.anatomyEstimate = null;
  }
  state.bundle = await api(`/api/projects/${id}`);
  state.referenceBundle = null;
  state.referenceError = null;
  const workspace = state.bundle.reference_workspaces?.at(-1);
  if (workspace) {
    try {
      state.referenceBundle = await api(`/api/projects/${id}/reference-workspaces/${workspace.id}`);
      const reviewed = new Set(state.referenceBundle.reviews.map((item) => item.proposal_id));
      state.motionProposal = [...state.referenceBundle.proposals].reverse().find((item) => !reviewed.has(item.id)) || null;
      state.referenceFrame = Math.min(state.referenceFrame, workspace.frame_count - 1);
    } catch (error) {
      state.referenceError = error.message;
    }
  }
  renderProjects();
  renderWorkspace();
}

function currentVersion() {
  const id = state.bundle?.project.current_version_id;
  return state.bundle?.versions.find((item) => item.id === id) || state.bundle?.versions.at(-1) || null;
}

function renderWorkflow() {
  const b = state.bundle;
  const steps = [
    ["01", "Исходники", b.assets.length > 0],
    ["02", "Evidence", b.evidence.some((item) => item.kind !== "media.probe")],
    ["03", "Treatment", b.treatments.length > 0],
    ["04", "Версия", b.versions.length > 1],
    ["05", "Render + QC", b.qc.length > 0],
  ];
  const current = Math.min(steps.findIndex((item) => !item[2]), steps.length - 1);
  $("#workflow-strip").innerHTML = steps.map((item, index) => `
    <div class="workflow-step ${item[2] ? "done" : index === current ? "current" : ""}">
      <b>${item[2] ? "✓" : item[0]}</b><span>${item[1]}</span>
    </div>`).join("");
}

function renderWorkspace() {
  const b = state.bundle;
  if (!b) return;
  $("#welcome").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  $("#project-title").textContent = b.project.name;
  $("#workspace-status").textContent = `${b.project.intent.scenario} · ${b.versions.length} версий`;
  renderWorkflow();
  renderEdit();
  renderAnatomy();
  renderEvidence();
  renderReference();
  renderVersions();
  renderDeliver();
}

function assetMarkup(asset) {
  const analyzed = state.bundle.evidence.filter((item) => item.asset_id === asset.id && item.kind !== "media.probe").length;
  const derived = state.bundle.derivatives.find((item) => item.asset_id === asset.id);
  const thumbnail = derived?.artifacts?.thumbnail;
  const proxy = derived?.artifacts?.proxy;
  const player = proxy && asset.media_kind === "video" ? `<details><summary>Просмотреть proxy</summary><video controls preload="metadata" src="/api/projects/${state.bundle.project.id}/file?path=${encodeURIComponent(proxy)}"></video></details>` : proxy && asset.media_kind === "audio" ? `<details><summary>Прослушать proxy</summary><audio controls preload="metadata" src="/api/projects/${state.bundle.project.id}/file?path=${encodeURIComponent(proxy)}"></audio></details>` : "";
  const artifactLink = (name, label) => derived?.artifacts?.[name] ? `<a class="artifact-link" href="/api/projects/${state.bundle.project.id}/file?path=${encodeURIComponent(derived.artifacts[name])}">${label}</a>` : "";
  return `<div class="asset">
    ${thumbnail ? `<img class="asset-thumb" src="/api/projects/${state.bundle.project.id}/file?path=${encodeURIComponent(thumbnail)}" alt="derived thumbnail">` : `<div class="asset-icon">${asset.media_kind === "image" ? "◫" : asset.media_kind === "audio" ? "⌁" : "▶"}</div>`}
    <div class="asset-main"><strong>${esc(asset.original_name)}</strong><small>${asset.metadata.width || "—"}×${asset.metadata.height || "—"} · ${(asset.size_bytes / 1048576).toFixed(1)} MB · ${short(asset.sha256)}</small><div class="artifact-row">${artifactLink("proxy", "proxy")}${artifactLink("contact_sheet", "contact sheet")}${artifactLink("waveform", "waveform")}</div>${player}</div>
    <span class="badge ${analyzed ? "good" : ""}">${analyzed ? `${analyzed} evidence` : "только probe"}</span>
    <button class="secondary" data-action="analyze" data-asset="${asset.id}">Анализ</button>
  </div>`;
}

function treatmentMarkup(treatment) {
  const graph = state.bundle.decision_graphs.find((item) => item.id === treatment.decision_graph_id);
  const decisions = graph?.decisions || [];
  return `<article class="treatment">
    <div class="card-head"><div><h4>${esc(treatment.title)}</h4><small class="muted">${esc(treatment.scenario)} · ${short(treatment.id)}</small></div><span class="badge warn">proposal</span></div>
    <p class="summary">${esc(treatment.summary)}</p>
    <div class="treatment-facts"><span class="badge">${treatment.duration_frames || "—"} frames</span><span class="badge">${treatment.used_asset_ids.length} sources</span><span class="badge">${esc(treatment.expected_style || "style open")}</span></div>
    <div class="grid two compact-fields"><div><strong>Сильные стороны</strong><ul class="rationale">${treatment.strengths.map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div><div><strong>Компромиссы / риски</strong><ul class="rationale">${[...treatment.tradeoffs, ...treatment.uncertainties, ...treatment.risks].map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div></div>
    <ul class="rationale">${treatment.rationale.map((item) => `<li>${esc(item.claim)} <small>(${Math.round(item.confidence * 100)}%)</small></li>`).join("")}</ul>
    <div class="decision-list">${decisions.map((item) => `<div class="qc-check"><span><strong>${esc(item.operation_type)}</strong> · ${esc(item.intent)}<small>${esc(item.rationale)}</small></span><span class="badge warn">${esc(item.approval)}</span></div>`).join("")}</div>
    <div class="button-row"><button class="primary" data-action="accept" data-treatment="${treatment.id}">Принять → новая версия</button><span class="muted">base ${short(treatment.base_version_id)}</span></div>
    <details><summary>Изменить patch перед применением</summary>
      <textarea class="code" id="patch-${treatment.id}">${esc(JSON.stringify(treatment.patch.operations, null, 2))}</textarea>
      <button class="secondary" data-action="apply-edited" data-treatment="${treatment.id}">Применить изменённый patch</button>
    </details>
  </article>`;
}

function timelineMarkup(version) {
  if (!version) return `<p class="muted">Timeline появится после первого treatment.</p>`;
  const total = version.timeline.duration_frames;
  return `<div class="timeline">${version.timeline.tracks.map((track, trackIndex) => `
    <div class="track-row"><div class="track-name">${esc(track.name)}<br><small>${track.kind}</small></div>
      <div class="track-lane">${track.clips.map((clip, clipIndex) => `<div class="clip ${track.kind} ${clip.enabled ? "" : "disabled"}" style="left:${clip.timeline_range.start / total * 100}%;width:${clip.timeline_range.duration / total * 100}%" title="${esc(clip.id)} · ${esc(clip.role)}"><span>${clip.pinned ? "📌 " : ""}${esc(clip.role)}</span><span class="clip-tools"><button data-action="timeline-pin" data-track="${trackIndex}" data-clip="${clipIndex}" data-value="${!clip.pinned}" title="Pin/unpin">${clip.pinned ? "◇" : "◆"}</button><button data-action="timeline-toggle" data-track="${trackIndex}" data-clip="${clipIndex}" data-value="${!clip.enabled}" title="Include/exclude">${clip.enabled ? "×" : "+"}</button><button data-action="render-segment" data-start="${clip.timeline_range.start}" data-duration="${clip.timeline_range.duration}" title="Render affected range">▶</button></span></div>`).join("")}</div>
    </div>`).join("")}</div>`;
}

function renderEdit() {
  const b = state.bundle;
  const version = currentVersion();
  $("#tab-edit").innerHTML = `
    <div class="grid two">
      <article class="card"><div class="card-head"><h3>Editorial Brief</h3><span class="badge">rev ${b.brief_revisions.length}</span></div><div class="intent-quote">“${esc(b.project.intent.text)}”</div><p>${esc([b.project.intent.audience, b.project.intent.frame_format, b.project.intent.tempo, b.project.intent.mood].filter(Boolean).join(" · ") || "Дополнительные ограничения не заданы")}</p><div class="button-row"><span class="badge">${esc(b.project.intent.privacy_mode)}</span><span class="badge">${esc(b.project.intent.automation_level)}</span></div><details><summary>Создать новую ревизию Brief</summary><form id="brief-revision-form" class="mini-form"><label>Замысел<textarea name="intent" required>${esc(b.project.intent.text)}</textarea></label><div class="grid two compact-fields"><label>Темп<input name="tempo" value="${esc(b.project.intent.tempo || "")}"></label><label>Настроение<input name="mood" value="${esc(b.project.intent.mood || "")}"></label></div><label>Причина ревизии<input name="rationale" required></label><button class="secondary">Сохранить ревизию</button></form></details><details><summary>Явно подтвердить Style Memory</summary><form id="style-confirm-form" class="mini-form"><label>Предпочтение<input name="preference" required placeholder="например, motion"></label><label>Значение<input name="value" required placeholder="например, restrained"></label><button class="secondary">Подтвердить и сохранить</button></form></details></article>
      <article class="card"><div class="card-head"><h3>Исходники</h3><span class="badge ${b.assets.length ? "good" : "warn"}">${b.assets.length}</span></div>
        <div class="asset-list">${b.assets.map(assetMarkup).join("") || `<p class="muted">Добавьте изображение, видео или аудио.</p>`}</div>
        <form id="upload-form" class="drop-zone"><label>Добавить неизменяемый источник<input type="file" name="file" required accept="image/*,video/*,audio/*" /></label><button class="secondary" type="submit">Загрузить локально</button></form>
      </article>
    </div>
    <article class="card" style="margin-top:16px"><div class="card-head"><h3>Treatments</h3><div class="button-row"><input id="proposal-duration" type="number" min="0.1" step="0.1" placeholder="сек." style="width:90px"><button class="primary" data-action="propose" ${b.assets.length ? "" : "disabled"}>Предложить монтаж</button></div></div>
      <div class="treatment-list">${b.treatments.map(treatmentMarkup).join("") || `<p class="muted">После анализа система предложит аргументированный treatment и альтернативы.</p>`}</div>
    </article>
    <article class="card" style="margin-top:16px"><div class="card-head"><h3>ИИ-соредактор · безопасный preview</h3><span class="badge">NL → typed patch</span></div><p>Команда никогда не применяется напрямую: сначала показывается типизированный diff и проверяются timeline-инварианты.</p><textarea id="language-command" rows="2" placeholder="Например: Сделай фон белым и скорость 1.2x"></textarea><div class="button-row"><button class="secondary" data-action="language-preview" ${version ? "" : "disabled"}>Показать diff</button>${state.languagePreview ? `<button class="primary" data-action="language-apply">Подтвердить patch</button>` : ""}</div>${state.languagePreview ? `<p>${esc(state.languagePreview.summary)}</p><textarea class="code" readonly>${esc(JSON.stringify(state.languagePreview.patch.operations, null, 2))}</textarea>${state.languagePreview.warnings.map((item) => `<p class="muted">⚠ ${esc(item)}</p>`).join("")}` : ""}</article>
    <article class="card" style="margin-top:16px"><div class="card-head"><h3>Канонический timeline</h3><span class="badge">${version ? `${seconds(version.timeline.duration_frames, version.timeline.frame_rate)} сек.` : "empty"}</span></div>${timelineMarkup(version)}</article>`;
  $("#upload-form")?.addEventListener("submit", uploadAsset);
  $("#brief-revision-form")?.addEventListener("submit", reviseBrief);
  $("#style-confirm-form")?.addEventListener("submit", confirmStyle);
}

function anatomyForView() {
  const anatomies = [...(state.bundle?.video_anatomies || [])].sort((left, right) =>
    String(left.generated_at).localeCompare(String(right.generated_at))
  );
  if (!anatomies.length) return null;
  const selected = anatomies.find((item) => item.plan.id === state.anatomyPlanId);
  const anatomy = selected || anatomies.at(-1);
  state.anatomyPlanId = anatomy.plan.id;
  return anatomy;
}

function frameEnd(range) { return range.start + range.duration; }

function anatomyAuthorityLegend() {
  return `<div class="authority-legend">
    <span class="authority measured">● измерено</span>
    <span class="authority normalized">● нормализовано</span>
    <span class="authority proposed">● предложение</span>
    <span class="authority human">● решение человека</span>
    <span class="authority canonical">● canonical version</span>
  </div>`;
}

function anatomyTimelineMarkup(anatomy) {
  const range = anatomy.plan.analysis_range;
  const position = (frame) => (frame - range.start) / range.duration * 100;
  const shots = anatomy.structure.shots.map((shot, index) => `
    <button class="anatomy-shot" data-action="anatomy-select-shot" data-shot="${shot.id}"
      style="left:${position(shot.frame_range.start)}%;width:${shot.frame_range.duration / range.duration * 100}%"
      title="${shot.id}: ${shot.frame_range.start}–${frameEnd(shot.frame_range) - 1}">
      <b>${index + 1}</b><span>${shot.frame_range.duration}f</span>
    </button>`).join("");
  const transitions = anatomy.structure.transitions.map((item) => `
    <button class="anatomy-transition ${item.transition_type === "unknown" ? "unknown" : ""}"
      data-action="anatomy-pin-frame" data-frame="${item.frame}"
      style="left:${position(item.frame)}%" title="${esc(item.transition_type)} · frame ${item.frame} · ${Math.round(item.confidence * 100)}%"></button>`).join("");
  const events = (anatomy.audio_timeline?.events || []).map((item) => {
    const startFrame = item.start_seconds * anatomy.plan.time_base.numerator / anatomy.plan.time_base.denominator;
    const durationFrames = Math.max(1, (item.end_seconds - item.start_seconds) * anatomy.plan.time_base.numerator / anatomy.plan.time_base.denominator);
    return `<span class="audio-event ${esc(item.kind)}" style="left:${position(startFrame)}%;width:${durationFrames / range.duration * 100}%" title="${esc(item.kind)} ${item.start_seconds.toFixed(2)}–${item.end_seconds.toFixed(2)} s"></span>`;
  }).join("");
  return `<div class="anatomy-ruler"><div class="anatomy-shot-lane">${shots}${transitions}</div><div class="anatomy-audio-lane">${events || `<span class="lane-empty">audio: событий не обнаружено</span>`}</div></div>`;
}

function anatomyShotMarkup(anatomy, shot) {
  const visual = anatomy.visual_observations.find((item) => item.shot_id === shot.id);
  const motion = anatomy.motion_evidence.find((item) => item.shot_id === shot.id);
  const transition = anatomy.structure.transitions.find((item) => Math.abs(item.frame - shot.frame_range.start) <= 1);
  const samples = anatomy.frame_manifest.samples.filter((item) => item.shot_id === shot.id && item.kept);
  const projectId = anatomy.project_id;
  return `<article class="anatomy-shot-card" data-shot-card="${shot.id}">
    <div class="card-head"><div><strong>${esc(shot.id)}</strong><small>frames ${shot.frame_range.start}–${frameEnd(shot.frame_range) - 1}</small></div><span class="badge ${shot.confidence > .75 ? "good" : "warn"}">${Math.round(shot.confidence * 100)}%</span></div>
    <div class="sample-strip">${samples.slice(0, 6).map((sample) => `<figure><img loading="lazy" src="/api/projects/${projectId}/file?path=${encodeURIComponent(sample.artifact_path)}" alt="frame ${sample.source_frame_index}"><figcaption>${sample.source_frame_index} · ${esc(sample.role)}</figcaption></figure>`).join("")}</div>
    ${transition ? `<p><span class="authority measured">transition</span> ${esc(transition.transition_type)} · ${Math.round(transition.confidence * 100)}%${transition.competing_types.length ? ` · competing: ${transition.competing_types.map(esc).join(", ")}` : ""}</p>` : ""}
    ${visual ? `<p><span class="authority normalized">visual</span> ${esc(visual.description)}</p>` : `<p class="muted">Semantic observation не запрашивался или недоступен.</p>`}
    ${motion ? `<p><span class="authority measured">motion</span> ${esc(motion.classification)} · ${motion.camera_hypotheses.map(esc).join(", ") || "без уверенной camera-гипотезы"} · ${Math.round(motion.confidence * 100)}%</p>` : `<p class="muted">Покадровое motion-evidence ещё не построено.</p>`}
    ${[...shot.unresolved_questions, ...(motion?.uncertainty || [])].map((item) => `<p class="uncertainty">⚠ ${esc(item)}</p>`).join("")}
    <div class="button-row"><button class="secondary" data-action="anatomy-deepen-shot" data-shot="${shot.id}">Углубить shot</button><button class="ghost" data-action="anatomy-pin-frame" data-frame="${shot.frame_range.start}">Pin вход</button><button class="ghost" data-action="anatomy-pin-frame" data-frame="${frameEnd(shot.frame_range) - 1}">Pin выход</button></div>
  </article>`;
}

function anatomyProposalMarkup(proposal, kind, reviews, acceptances) {
  const review = [...reviews].reverse().find((item) => item.proposal_id === proposal.id);
  const acceptance = acceptances.find((item) => item.proposal_id === proposal.id);
  const confidence = proposal.confidence ?? proposal.sections?.reduce((sum, item) => sum + item.confidence, 0) / Math.max(1, proposal.sections?.length || 1);
  const summary = kind === "editorial"
    ? `${proposal.sections.length} sections · ${proposal.likely_techniques.join(", ") || "techniques open"}`
    : `${proposal.shot_skeleton.length} shots · ${proposal.duration_frames} frames · source-neutral`;
  return `<article class="proposal-card">
    <div class="card-head"><div><strong>${kind === "editorial" ? "Editorial structure" : "Reconstruction skeleton"}</strong><small>${short(proposal.id, 18)}</small></div><span class="authority ${acceptance ? "canonical" : review ? "human" : "proposed"}">${acceptance ? "accepted" : review?.decision || "proposal"}</span></div>
    <p>${esc(summary)}</p><div class="button-row"><span class="badge">confidence ${Math.round((confidence || 0) * 100)}%</span><span class="badge">timeline mutated: ${proposal.timeline_mutated ? "yes" : "no"}</span></div>
    ${review ? `<p><span class="authority human">${esc(review.reviewer)}</span> ${esc(review.rationale)}</p>` : `<div class="proposal-review"><input data-reviewer="${proposal.id}" placeholder="Кто проверил" value="editor"><input data-rationale="${proposal.id}" placeholder="Причина решения"><button class="primary" data-action="anatomy-review-proposal" data-kind="${kind}" data-proposal="${proposal.id}" data-decision="approved">Approve</button><button class="danger" data-action="anatomy-review-proposal" data-kind="${kind}" data-proposal="${proposal.id}" data-decision="rejected">Reject</button></div>`}
    ${kind === "reconstruction" && review?.decision === "approved" && proposal.patch_preview && !acceptance ? `<button class="primary" data-action="anatomy-accept-reconstruction" data-proposal="${proposal.id}" data-review="${review.id}">Применить через reversible patch</button>` : ""}
    ${acceptance ? `<p><span class="authority canonical">version ${short(acceptance.version_id, 18)}</span> · inverse operations ${acceptance.inverse_operation_count}</p>` : ""}
    <details><summary>Typed payload</summary><pre>${esc(JSON.stringify(proposal, null, 2))}</pre></details>
  </article>`;
}

function renderAnatomy() {
  const panel = $("#tab-anatomy");
  if (!panel || !state.bundle) return;
  const b = state.bundle;
  const videoAssets = b.assets.filter((item) => item.media_kind === "video");
  const anatomy = anatomyForView();
  const selector = (b.video_anatomies || []).map((item) => `<option value="${item.plan.id}" ${anatomy?.plan.id === item.plan.id ? "selected" : ""}>${esc(item.profile)} · ${item.structure.shots.length} shots · ${short(item.plan.id, 16)}</option>`).join("");
  const videoOptions = videoAssets.map((item) => `<option value="${item.id}">${esc(item.original_name)}</option>`).join("");
  const targetOptions = b.assets.filter((item) => item.media_kind === "image" || item.media_kind === "video").map((item) => `<option value="${item.id}">${esc(item.original_name)}</option>`).join("");
  const versionOptions = b.versions.map((item) => `<option value="${item.id}">${esc(item.message)} · ${short(item.id)}</option>`).join("");
  if (!videoAssets.length) {
    panel.innerHTML = `<article class="card"><div class="card-head"><h3>Video Anatomy</h3><span class="badge warn">нужен video asset</span></div>${anatomyAuthorityLegend()}<p>Добавьте видео во вкладке «Монтаж». Анализ работает внутри проекта и не требует AoA/Abyss runtime.</p></article>`;
    return;
  }
  const structureEvidence = anatomy ? b.evidence.find((item) => anatomy.evidence_refs.includes(item.id) && item.kind === "video.structure") : null;
  panel.innerHTML = `
    <article class="card anatomy-console"><div class="card-head"><div><p class="eyebrow">VIDEO → STRUCTURE EVIDENCE</p><h3>Профилированный анализ без изменения timeline</h3></div><span class="badge ${anatomy?.status === "complete" ? "good" : anatomy ? "warn" : ""}">${anatomy?.status || "not run"}</span></div>
      ${anatomyAuthorityLegend()}
      <div class="anatomy-controls"><label>Видео<select id="anatomy-asset">${videoOptions}</select></label><label>Профиль<select id="anatomy-profile"><option value="quick">quick</option><option value="structural" selected>structural</option><option value="semantic">semantic</option><option value="motion">motion</option><option value="reconstruct">reconstruct</option></select></label><label>Pin frames<input id="anatomy-pins" placeholder="0, 120, 240"></label><label class="check-label"><input id="anatomy-provider-opt-in" type="checkbox"> явный opt-in для provider</label><div class="button-row"><button class="ghost" data-action="anatomy-estimate">Оценить</button><button class="primary" data-action="anatomy-run">Запустить</button></div></div>
      ${state.anatomyEstimate ? `<div class="resource-estimate ${state.anatomyEstimate.admitted ? "admitted" : "refused"}"><strong>${state.anatomyEstimate.admitted ? "admitted" : "refused"}</strong><span>≈ ${state.anatomyEstimate.estimated_runtime_seconds.toFixed(1)} s · artifacts ${(state.anatomyEstimate.estimated_artifact_bytes / 1048576).toFixed(1)} MB · required free ${(state.anatomyEstimate.required_free_bytes / 1073741824).toFixed(2)} GB</span>${state.anatomyEstimate.warnings.map((item) => `<small>⚠ ${esc(item)}</small>`).join("")}</div>` : ""}
      ${anatomy ? `<div class="anatomy-controls secondary-row"><label>Сохранённый проход<select id="anatomy-selector">${selector}</select></label><button class="secondary" data-action="anatomy-focused-plans">Планы углубления</button><a class="artifact-link" href="/api/projects/${b.project.id}/video-anatomy/${anatomy.plan.id}/contact-sheet" target="_blank">Открыть contact sheet</a></div>` : ""}
    </article>
    ${anatomy ? `
      <div class="anatomy-metrics">${Object.entries(anatomy.coverage_matrix).map(([name, value]) => `<div><span>${esc(name)}</span><b>${Math.round(value * 100)}%</b><progress max="1" value="${value}"></progress></div>`).join("")}<div><span>samples</span><b>${anatomy.frame_manifest.selected_count}</b><small>${anatomy.frame_manifest.duplicate_count} deduplicated</small></div></div>
      <article class="card anatomy-timeline-card"><div class="card-head"><h3>Shots, transitions и audio events</h3><span class="badge">${anatomy.structure.analysis_range.start}–${frameEnd(anatomy.structure.analysis_range) - 1}f</span></div>${anatomyTimelineMarkup(anatomy)}${anatomy.unresolved_ranges.map((item) => `<p class="uncertainty">⚠ unresolved frames ${item.start}–${frameEnd(item) - 1}</p>`).join("")}${anatomy.contradictions.map((item) => `<p class="uncertainty">⇄ ${esc(item)}</p>`).join("")}</article>
      <div class="anatomy-shot-grid">${anatomy.structure.shots.map((shot) => anatomyShotMarkup(anatomy, shot)).join("")}</div>
      <div class="grid two" style="margin-top:16px">
        <article class="card"><div class="card-head"><h3>Contact sheet</h3><span class="authority measured">immutable artifact</span></div><img class="anatomy-contact-sheet" loading="lazy" src="/api/projects/${b.project.id}/video-anatomy/${anatomy.plan.id}/contact-sheet" alt="Video Anatomy contact sheet"><p>${short(anatomy.frame_manifest.manifest_sha256, 20)} · ${(anatomy.frame_manifest.artifact_bytes / 1048576).toFixed(2)} MB</p></article>
        <article class="card"><div class="card-head"><h3>Audio / transcript</h3><span class="badge ${anatomy.audio_timeline?.partial ? "warn" : "good"}">${anatomy.audio_timeline ? (anatomy.audio_timeline.partial ? "partial" : "complete") : "not requested"}</span></div>${anatomy.audio_timeline?.transcript_segments.length ? `<div class="transcript">${anatomy.audio_timeline.transcript_segments.map((item) => `<p><time>${item.start_seconds.toFixed(2)}–${item.end_seconds.toFixed(2)}</time>${item.speaker ? `<strong>${esc(item.speaker)}</strong> ` : ""}${esc(item.text)}</p>`).join("")}</div>` : `<p class="muted">Транскрипта нет. Это не маскируется как нулевая уверенность.</p>`}${anatomy.audio_timeline?.unknown_intervals.map((item) => `<p class="uncertainty">unknown audio ${item[0].toFixed(2)}–${item[1].toFixed(2)} s</p>`).join("") || ""}</article>
      </div>
      ${state.anatomyFocusedPlans.length ? `<article class="card" style="margin-top:16px"><div class="card-head"><h3>Адаптивные планы углубления</h3><span class="badge">${state.anatomyFocusedPlans.length}</span></div>${state.anatomyFocusedPlans.map((item) => `<div class="qc-check"><span><strong>${esc(item.profile)}</strong><small>frames ${item.analysis_range.start}–${frameEnd(item.analysis_range) - 1} · ${item.focused_rescan_reasons.map(esc).join("; ")}</small></span><span class="badge">${item.global_frame_budget} samples</span></div>`).join("")}</article>` : ""}
      <div class="grid two" style="margin-top:16px">
        <article class="card"><div class="card-head"><h3>Human correction</h3><span class="authority human">superseding evidence</span></div>${structureEvidence ? `<p>Исходное analyzer-evidence остаётся неизменным. Исправление создаёт отдельную запись.</p><textarea class="code" id="anatomy-correction-payload">${esc(JSON.stringify(structureEvidence.payload, null, 2))}</textarea><input id="anatomy-correction-rationale" placeholder="Что проверено человеком и почему"><button class="secondary" data-action="anatomy-correct-evidence" data-evidence="${structureEvidence.id}" data-asset="${anatomy.asset_id}" data-kind="${structureEvidence.kind}">Записать correction</button>` : `<p class="muted">Связанное structure evidence не найдено.</p>`}</article>
        <article class="card"><div class="card-head"><h3>Предложения реконструкции</h3><span class="authority proposed">не canonical</span></div><p>Proposal не меняет timeline. Для исполнимого preview укажите target и base version.</p><div class="mini-form"><label>Target asset<select id="anatomy-target-asset"><option value="">source-neutral only</option>${targetOptions}</select></label><label>Base version<select id="anatomy-base-version"><option value="">без patch preview</option>${versionOptions}</select></label><div class="button-row"><button class="secondary" data-action="anatomy-create-editorial">Editorial proposal</button><button class="secondary" data-action="anatomy-create-reconstruction">Reconstruction proposal</button></div></div></article>
      </div>
      <div class="proposal-grid" style="margin-top:16px">${b.video_editorial_proposals.filter((item) => item.anatomy_id === anatomy.id).map((item) => anatomyProposalMarkup(item, "editorial", b.video_proposal_reviews, b.video_proposal_acceptances)).join("")}${b.video_reconstruction_proposals.filter((item) => item.reference_anatomy_id === anatomy.id).map((item) => anatomyProposalMarkup(item, "reconstruction", b.video_proposal_reviews, b.video_proposal_acceptances)).join("")}</div>
      <article class="card" style="margin-top:16px"><details><summary>Coverage, confidence, provenance и полный агрегат</summary><pre>${esc(JSON.stringify({coverage: anatomy.structure.coverage, confidence: anatomy.confidence_summary, provenance_graph: anatomy.provenance_graph, evidence_refs: anatomy.evidence_refs, incompleteness_reasons: anatomy.incompleteness_reasons}, null, 2))}</pre></details></article>
    ` : `<article class="card" style="margin-top:16px"><p class="muted">Выберите видео и профиль. Первый структурный проход построит shots, transitions, защищённые boundary triplets, coverage и provenance.</p></article>`}`;
  $("#anatomy-selector")?.addEventListener("change", (event) => { state.anatomyPlanId = event.target.value; state.anatomyFocusedPlans = []; renderAnatomy(); });
}

function renderEvidence() {
  const evidence = state.bundle.evidence;
  const groups = Object.entries(evidence.reduce((result, item) => { const group = item.kind.split(".")[0]; result[group] = (result[group] || 0) + 1; return result; }, {}));
  const transcriptSegments = evidence.filter((item) => item.kind === "speech.transcript").flatMap((item) => item.payload.segments || item.payload.chunks || item.payload.result?.segments || []);
  $("#tab-evidence").innerHTML = `<div class="grid two">
    <article class="card"><div class="card-head"><h3>Evidence ledger</h3><span class="badge ${evidence.length ? "good" : ""}">${evidence.length}</span></div>
      <p>Измерения не принимают решений. Provenance показывает, каким инструментом и с какими параметрами они получены.</p>
    </article>
    <article class="card"><div class="card-head"><h3>Карта материала</h3></div><div class="button-row">${groups.map(([name, count]) => `<span class="badge">${esc(name)} · ${count}</span>`).join("") || `<span class="muted">Нет групп</span>`}</div><input id="evidence-search" type="search" placeholder="Поиск по evidence и транскрипту"></article>
  </div>${transcriptSegments.length ? `<article class="card" style="margin-top:16px"><div class="card-head"><h3>Транскрипт с таймкодами</h3><span class="badge">${transcriptSegments.length}</span></div><div class="transcript">${transcriptSegments.map((item) => `<p class="searchable" data-search="${esc(item.text)}"><time>${Number(item.start ?? item.start_seconds ?? 0).toFixed(2)}–${Number(item.end ?? item.end_seconds ?? 0).toFixed(2)}</time> ${item.speaker ? `<strong>${esc(item.speaker)}</strong>` : ""} ${esc(item.text)}</p>`).join("")}</div></article>` : ""}<div class="evidence-list" style="margin-top:16px">${evidence.map((item) => `<article class="evidence searchable" data-search="${esc(`${item.kind} ${JSON.stringify(item.payload)}`)}"><div class="evidence-head"><div><strong>${esc(item.kind)}</strong><small class="muted"> · asset ${short(item.asset_id)}</small></div><div><span class="badge">${esc(item.authority)}</span> <span class="badge ${item.confidence > .75 ? "good" : "warn"}">${Math.round(item.confidence * 100)}%</span></div></div><p>${esc(item.provenance.tool)} · ${esc(item.provenance.tool_version)}</p><details><summary>Показать payload и provenance</summary><pre>${esc(JSON.stringify({payload:item.payload, provenance:item.provenance, ranges:item.ranges, artifacts:item.artifacts}, null, 2))}</pre></details></article>`).join("") || `<article class="card"><p class="muted">Evidence пока нет.</p></article>`}</div>`;
  $("#evidence-search")?.addEventListener("input", (event) => { const query = event.target.value.toLocaleLowerCase(); document.querySelectorAll("#tab-evidence .searchable").forEach((node) => node.classList.toggle("hidden", !node.dataset.search.toLocaleLowerCase().includes(query))); });
}

function referenceArtifactUrl(role) {
  const projectId = state.bundle.project.id;
  const workspaceId = state.referenceBundle.workspace.id;
  return `/api/projects/${projectId}/reference-workspaces/${workspaceId}/artifacts/${encodeURIComponent(role)}`;
}

function curveValue(sample, side, channel, derivative) {
  const curve = sample[side];
  if (derivative === "value") return Number(curve[channel] ?? 0);
  const key = channel === "rotation_degrees" ? "rotation" : channel;
  return Number(curve[derivative]?.[key] ?? 0);
}

function motionCurveSvg(bundle) {
  const samples = bundle.samples;
  const channel = state.referenceChannel;
  const derivative = state.referenceDerivative;
  const values = samples.flatMap((sample) => [
    curveValue(sample, "reference", channel, derivative),
    curveValue(sample, "candidate", channel, derivative),
  ]);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const span = Math.max(maximum - minimum, 1e-9);
  const width = 1000;
  const height = 260;
  const x = (frame) => 42 + frame / (samples.length - 1) * 930;
  const y = (value) => 220 - (value - minimum) / span * 180;
  const points = (side) => samples.map((sample) => `${x(sample.frame).toFixed(2)},${y(curveValue(sample, side, channel, derivative)).toFixed(2)}`).join(" ");
  const phaseLines = Object.entries(bundle.workspace.phase_markers).map(([label, frame]) => `
    <line class="phase-line" x1="${x(frame)}" x2="${x(frame)}" y1="24" y2="224"></line>
    <text class="phase-label" x="${x(frame) + 4}" y="18">${esc(label)} · ${frame}</text>`).join("");
  return `<svg class="motion-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Reference and candidate ${esc(channel)} ${esc(derivative)} curves">
    <line class="chart-axis" x1="42" x2="972" y1="224" y2="224"></line>
    ${phaseLines}
    <polyline class="curve reference" points="${points("reference")}"></polyline>
    <polyline class="curve candidate" points="${points("candidate")}"></polyline>
    <line class="frame-cursor" data-frame-cursor x1="${x(state.referenceFrame)}" x2="${x(state.referenceFrame)}" y1="24" y2="224"></line>
    <text class="chart-bound" x="4" y="42">${maximum.toPrecision(4)}</text>
    <text class="chart-bound" x="4" y="222">${minimum.toPrecision(4)}</text>
  </svg>`;
}

function confidenceSvg(bundle) {
  const samples = bundle.samples;
  const width = 1000;
  const x = (frame) => 36 + frame / (samples.length - 1) * 940;
  const y = (value) => 122 - value * 92;
  const points = (side) => samples.map((sample) => `${x(sample.frame).toFixed(2)},${y(sample[side].confidence).toFixed(2)}`).join(" ");
  const uncertain = samples.filter((sample) => sample.reference.uncertainty.length || sample.candidate.uncertainty.length).map((sample) => `<circle class="uncertainty-dot" cx="${x(sample.frame)}" cy="132" r="2.2"><title>frame ${sample.frame}: ${esc([...sample.reference.uncertainty, ...sample.candidate.uncertainty].join("; "))}</title></circle>`).join("");
  return `<svg class="confidence-chart" viewBox="0 0 ${width} 150" role="img" aria-label="Confidence and uncertainty timeline">
    <line class="chart-axis" x1="36" x2="976" y1="122" y2="122"></line>
    <polyline class="curve reference" points="${points("reference")}"></polyline>
    <polyline class="curve candidate" points="${points("candidate")}"></polyline>
    ${uncertain}
    <line class="frame-cursor" data-frame-cursor x1="${x(state.referenceFrame)}" x2="${x(state.referenceFrame)}" y1="18" y2="136"></line>
  </svg>`;
}

function closestDiagnosticRole(prefix, frame) {
  const roles = Object.keys(state.referenceBundle.workspace.artifacts)
    .filter((role) => role.startsWith(`${prefix}_`) && /^\d+$/.test(role.split("_").at(-1)))
    .map((role) => ({role, frame: Number(role.split("_").at(-1))}));
  if (!roles.length) return prefix;
  return roles.sort((left, right) => Math.abs(left.frame - frame) - Math.abs(right.frame - frame) || left.frame - right.frame)[0].role;
}

function sampleDetailMarkup(sample) {
  if (!sample) return "";
  const row = (label, key, digits = 5) => `<tr><th>${label}</th><td>${Number(sample.reference[key]).toFixed(digits)}</td><td>${Number(sample.candidate[key]).toFixed(digits)}</td><td>${(Number(sample.candidate[key]) - Number(sample.reference[key])).toExponential(2)}</td></tr>`;
  return `<div class="card-head"><h3>Кадр ${sample.frame}</h3><span class="badge">${sample.time_seconds.toFixed(2)} сек.</span></div>
    <table class="sample-table"><thead><tr><th>Канал</th><th>Reference</th><th>Candidate</th><th>Δ</th></tr></thead><tbody>
      ${row("center x", "center_x")}${row("center y", "center_y")}${row("scale", "scale")}${row("rotation", "rotation_degrees", 4)}
    </tbody></table>
    <div class="button-row"><span class="badge ${sample.reference.confidence > .8 ? "good" : "warn"}">ref confidence ${(sample.reference.confidence * 100).toFixed(1)}%</span><span class="badge ${sample.candidate.confidence > .8 ? "good" : "warn"}">candidate ${(sample.candidate.confidence * 100).toFixed(1)}%</span><span class="badge">residual ${sample.reference.residual_flow_mean?.toFixed(4) ?? "—"}</span></div>
    ${sample.reference.uncertainty.map((item) => `<p class="uncertainty">⚠ ${esc(item)}</p>`).join("")}`;
}

function proposalRange(proposal) {
  const ranges = proposal?.items.map((item) => item.diff?.affected_range).filter(Boolean) || [];
  if (!ranges.length) return null;
  const start = Math.min(...ranges.map((item) => item.start));
  const end = Math.max(...ranges.map((item) => item.start + item.duration));
  return {start, duration: end - start};
}

function motionProposalMarkup(proposal) {
  if (!proposal) return `<p class="muted">Сначала сформулируйте просьбу или настройте tangent/phase вручную. Preview не меняет проект.</p>`;
  const range = proposalRange(proposal);
  return `<div class="proposal-head"><div><strong>Typed proposal ${short(proposal.id)}</strong><small>base ${short(proposal.base_version_id)} · effect ${short(proposal.base_effect_sha256)}</small></div><span class="badge warn">ожидает решения</span></div>
    <div class="correction-list">${proposal.items.map((item, index) => `<article class="correction-item ${item.executable ? "" : "blocked"}">
      <div class="card-head"><div><strong>${index + 1}. ${esc(item.operation.kind)}</strong><small>${esc(item.rationale)}</small></div><span class="badge ${item.executable ? "good" : "fail"}">${item.executable ? "исполняемо" : "blocked"}</span></div>
      <pre>${esc(JSON.stringify(item.operation, null, 2))}</pre>
      ${item.diff ? `<div class="typed-diff"><span>frames ${item.diff.affected_range.start}–${item.diff.affected_range.start + item.diff.affected_range.duration - 1}</span><span>${item.diff.changed_frame_count} changed</span><span>max Δ ${item.diff.maximum_matrix_coefficient_delta.toExponential(3)}</span><span>endpoints ${item.diff.endpoint_preserved ? "preserved" : "changed"}</span></div>` : item.blockers.map((blocker) => `<p class="uncertainty">⛔ ${esc(blocker)}</p>`).join("")}
      <fieldset class="review-choice"><legend>Решение редактора</legend><label><input type="radio" name="decision-${item.id}" value="approve" ${item.executable ? "" : "disabled"}> approve</label><label><input type="radio" name="decision-${item.id}" value="reject"> reject</label><input class="decision-rationale" data-correction="${item.id}" placeholder="Обязательная причина решения"></fieldset>
    </article>`).join("")}</div>
    <div class="button-row"><button class="primary" data-action="reference-review" data-proposal="${proposal.id}">Записать все решения</button>${range ? `<button class="secondary" data-action="reference-render-segment" data-start="${range.start}" data-duration="${range.duration}">Точечный render ${range.start}–${range.start + range.duration - 1}</button>` : ""}</div>`;
}

function referenceVersionVideo(version, side) {
  if (!version) return `<div class="drop-zone muted">Нет выбранной версии</div>`;
  const profile = version.preview_available ? "preview" : version.final_available ? "final" : null;
  if (!profile) return `<div class="drop-zone muted">Для этой версии нет рендера.</div>`;
  const path = `renders/${version.id}/${profile}/video.mp4`;
  return `<video id="reference-version-player-${side}" controls preload="metadata" src="/api/projects/${state.bundle.project.id}/file?path=${encodeURIComponent(path)}"></video>`;
}

function bindSynchronizedMedia(ids) {
  const media = ids.map((id) => document.getElementById(id)).filter(Boolean);
  if (media.length < 2) return;
  let syncing = false;
  const mirror = (source, operation) => {
    if (syncing) return;
    syncing = true;
    media.filter((item) => item !== source).forEach((item) => operation(item));
    queueMicrotask(() => { syncing = false; });
  };
  media.forEach((source) => {
    source.addEventListener("play", () => mirror(source, (item) => item.play().catch(() => {})));
    source.addEventListener("pause", () => mirror(source, (item) => item.pause()));
    source.addEventListener("ratechange", () => mirror(source, (item) => { item.playbackRate = source.playbackRate; }));
    source.addEventListener("seeking", () => mirror(source, (item) => { item.currentTime = source.currentTime; }));
    source.addEventListener("timeupdate", () => {
      if (source.id !== ids[0] || syncing || !state.referenceBundle) return;
      const fps = state.referenceBundle.workspace.frame_rate.numerator / state.referenceBundle.workspace.frame_rate.denominator;
      const frame = Math.max(0, Math.min(state.referenceBundle.workspace.frame_count - 1, Math.round(source.currentTime * fps)));
      updateReferenceFrame(frame, false);
      mirror(source, (item) => { if (Math.abs(item.currentTime - source.currentTime) > .06) item.currentTime = source.currentTime; });
    });
  });
}

function updateReferenceFrame(frame, seek = true) {
  const bundle = state.referenceBundle;
  if (!bundle) return;
  state.referenceFrame = Math.max(0, Math.min(bundle.workspace.frame_count - 1, Number(frame)));
  const slider = $("#reference-frame");
  if (slider) slider.value = String(state.referenceFrame);
  const label = $("#reference-frame-label");
  if (label) label.textContent = `frame ${state.referenceFrame} · ${(state.referenceFrame / (bundle.workspace.frame_rate.numerator / bundle.workspace.frame_rate.denominator)).toFixed(2)} s`;
  const detail = $("#reference-sample-detail");
  if (detail) detail.innerHTML = sampleDetailMarkup(bundle.samples[state.referenceFrame]);
  const overlay = $("#reference-frame-overlay");
  if (overlay) overlay.src = referenceArtifactUrl(closestDiagnosticRole("frame_overlay", state.referenceFrame));
  const residual = $("#reference-flow-residual");
  if (residual) residual.src = referenceArtifactUrl(closestDiagnosticRole("optical_flow_residual", state.referenceFrame));
  const chartX = 42 + state.referenceFrame / (bundle.workspace.frame_count - 1) * 930;
  document.querySelectorAll(".motion-chart [data-frame-cursor]").forEach((line) => { line.setAttribute("x1", chartX); line.setAttribute("x2", chartX); });
  const confidenceX = 36 + state.referenceFrame / (bundle.workspace.frame_count - 1) * 940;
  document.querySelectorAll(".confidence-chart [data-frame-cursor]").forEach((line) => { line.setAttribute("x1", confidenceX); line.setAttribute("x2", confidenceX); });
  if (seek) {
    const secondsAtFrame = state.referenceFrame / (bundle.workspace.frame_rate.numerator / bundle.workspace.frame_rate.denominator);
    ["reference-player", "candidate-player", "difference-player"].forEach((id) => {
      const player = document.getElementById(id);
      if (player && Math.abs(player.currentTime - secondsAtFrame) > .03) player.currentTime = secondsAtFrame;
    });
  }
}

function renderReference() {
  const panel = $("#tab-reference");
  if (!panel) return;
  if (!state.bundle.reference_workspaces?.length) {
    panel.innerHTML = `<article class="card"><div class="card-head"><h3>Понимание референса</h3><span class="badge">не подключено</span></div><p>Здесь появится hash-verified Reference Workspace после gated анализа и reconstruction study. Референс никогда не становится монтажным source.</p></article>`;
    return;
  }
  if (!state.referenceBundle) {
    panel.innerHTML = `<article class="card"><div class="card-head"><h3>Reference Workspace sealed</h3><span class="badge fail">недоступно</span></div><p>${esc(state.referenceError || "Нужен актуальный revision-bound readiness receipt.")}</p></article>`;
    return;
  }
  const b = state.referenceBundle;
  const w = b.workspace;
  const markers = w.phase_markers;
  const current = b.versions.find((item) => item.current) || b.versions.at(-1);
  const base = b.versions.find((item) => item.id === w.base_version_id) || b.versions[0];
  const latestReview = b.reviews.at(-1);
  const reviewedProposal = latestReview ? b.proposals.find((item) => item.id === latestReview.proposal_id) : null;
  const reviewedRange = proposalRange(reviewedProposal) || {start: markers.twist_start, duration: markers.twist_end - markers.twist_start + 1};
  const versionOptions = (selected) => b.versions.map((version) => `<option value="${version.id}" ${version.id === selected?.id ? "selected" : ""}>${esc(version.message)} · ${short(version.id)}</option>`).join("");
  const pivot = w.pivot_identifiable ? "identified" : "unidentifiable · matrix/translation gauge";
  panel.innerHTML = `
    <article class="reference-hero card">
      <div class="card-head"><div><p class="eyebrow">REFERENCE UNDERSTANDING V2</p><h3>Синхронное сравнение и наблюдаемая модель</h3></div><div class="button-row"><span class="badge good">objective ${esc(b.objective_summary.objective_overall)}</span><span class="badge warn">human review ${b.objective_summary.human_review ? "recorded" : "pending"}</span></div></div>
      <div class="reference-player">
        <video id="candidate-player" controls preload="metadata" src="${referenceArtifactUrl("candidate_media")}"></video>
        <video id="reference-player" class="reference-overlay-video" muted preload="metadata" src="${referenceArtifactUrl("reference_media")}"></video>
        <div class="pivot-overlay ${w.pivot_identifiable ? "identified" : "uncertain"}"><span>pivot</span><b>${w.pivot_identifiable ? "＋" : "∅"}</b><small>${esc(pivot)}</small></div>
        <div class="player-label candidate">Candidate</div><div class="player-label reference">Reference overlay</div>
      </div>
      <label class="opacity-control">Прозрачность reference overlay<input id="reference-opacity" type="range" min="0" max="1" value=".5" step=".01"></label>
      <div class="frame-scrubber"><input id="reference-frame" type="range" min="0" max="${w.frame_count - 1}" value="${state.referenceFrame}" step="1"><span id="reference-frame-label"></span></div>
      <div class="grid two reference-diagnostics"><div><h4>Aligned difference</h4><video id="difference-player" controls muted preload="metadata" src="${referenceArtifactUrl("aligned_difference")}"></video></div><div><h4>Side-by-side</h4><video controls muted preload="metadata" src="${referenceArtifactUrl("side_by_side")}"></video></div></div>
    </article>
    <div class="grid two" style="margin-top:16px">
      <article class="card"><div class="card-head"><h3>Frame overlay</h3><span class="badge">nearest diagnostic</span></div><img id="reference-frame-overlay" class="diagnostic-image" alt="frame overlay"></article>
      <article class="card"><div class="card-head"><h3>Optical-flow residual</h3><span class="badge">uncertainty, not intent</span></div><img id="reference-flow-residual" class="diagnostic-image" alt="optical flow residual"></article>
    </div>
    <article class="card curve-card" style="margin-top:16px">
      <div class="card-head"><h3>Transform / velocity / acceleration / jerk</h3><div class="button-row"><span class="legend reference">Reference</span><span class="legend candidate">Candidate</span></div></div>
      <div class="grid two compact-fields"><label>Канал<select id="reference-channel"><option value="center_x" ${state.referenceChannel === "center_x" ? "selected" : ""}>center x</option><option value="center_y" ${state.referenceChannel === "center_y" ? "selected" : ""}>center y</option><option value="scale" ${state.referenceChannel === "scale" ? "selected" : ""}>scale</option><option value="rotation_degrees" ${state.referenceChannel === "rotation_degrees" ? "selected" : ""}>rotation</option></select></label><label>Производная<select id="reference-derivative"><option value="value" ${state.referenceDerivative === "value" ? "selected" : ""}>transform</option><option value="velocity" ${state.referenceDerivative === "velocity" ? "selected" : ""}>velocity</option><option value="acceleration" ${state.referenceDerivative === "acceleration" ? "selected" : ""}>acceleration</option><option value="jerk" ${state.referenceDerivative === "jerk" ? "selected" : ""}>jerk</option></select></label></div>
      ${motionCurveSvg(b)}
      <div class="phase-pills">${Object.entries(markers).map(([label, frame]) => `<button class="badge" data-action="reference-jump-frame" data-frame="${frame}">${esc(label)} · ${frame}</button>`).join("")}</div>
      <h4>Confidence и uncertainty</h4>${confidenceSvg(b)}
      <div id="reference-sample-detail">${sampleDetailMarkup(b.samples[state.referenceFrame])}</div>
    </article>
    <div class="grid two" style="margin-top:16px">
      <article class="card"><div class="card-head"><h3>Evidence plots</h3><span class="badge">immutable</span></div><div class="plot-grid">${["transform_curves","velocity","acceleration","jerk","pivot","motion_error","derivative_error","phase_contact_sheet"].map((role) => `<figure><img src="${referenceArtifactUrl(role)}" alt="${esc(role)}"><figcaption>${esc(role)}</figcaption></figure>`).join("")}</div></article>
      <article class="card"><div class="card-head"><h3>Вариантные интерпретации</h3><span class="badge">${b.interpretations.length}</span></div><div class="interpretation-list">${b.interpretations.map((item) => `<article class="interpretation ${item.status}"><div class="card-head"><strong>${esc(item.label)}</strong><span class="badge ${item.status === "supported" ? "good" : item.status === "rejected" ? "fail" : "warn"}">${esc(item.status)}</span></div><p>${esc(item.explanation)}</p><small>confidence ${(item.confidence * 100).toFixed(1)}% · ${item.editable_operation_kinds.map(esc).join(", ") || "no added operation"}</small></article>`).join("")}</div>${b.warnings.map((item) => `<p class="uncertainty">⚠ ${esc(item)}</p>`).join("")}</article>
    </div>
    <div class="grid two" style="margin-top:16px">
      <article class="card"><div class="card-head"><h3>Коррекция понимания</h3><span class="badge">proposal only</span></div>
        <label>Редакторская просьба<textarea id="reference-command" rows="3">переход должен мягче подкручиваться и чуть позже оседать</textarea></label><button class="secondary" data-action="reference-language-preview">Получить typed diff</button>
        <details open><summary>Editable tangent handle</summary><div class="mini-form tangent-editor"><div class="grid two compact-fields"><label>Канал<select id="tangent-channel"><option value="rotation">rotation</option><option value="matrix">matrix</option></select></label><label>Anchor frame<input id="tangent-anchor" type="number" value="${markers.twist_peak}" min="${markers.twist_start}" max="${markers.twist_end}"></label><label>Strength<input id="tangent-strength" type="range" min=".05" max="1" step=".05" value=".35"></label><label>Range<input id="tangent-start" type="number" value="${markers.twist_start}"><input id="tangent-end" type="number" value="${markers.twist_end}"></label></div><button class="secondary" data-action="reference-tangent-preview">Preview tangent</button></div></details>
        <details><summary>Phase boundary / pivot hypothesis</summary><div class="mini-form"><div class="grid two compact-fields"><label>Boundary<select id="phase-boundary">${Object.keys(markers).map((item) => `<option value="${item}">${esc(item)}</option>`).join("")}</select></label><label>Δ frames<input id="phase-delta" type="number" min="-24" max="24" value="1"></label></div><div class="button-row"><button class="secondary" data-action="reference-phase-preview">Preview boundary move</button><button class="ghost" data-action="reference-pivot-preview">Preview pivot +0.01</button></div></div></details>
      </article>
      <article class="card"><div class="card-head"><h3>Human correction evidence</h3><span class="badge">${b.human_correction_evidence_ids.length}</span></div><p>Analyzer interpretation остаётся в истории. Каждое решение человека записывается отдельным evidence и может создать только обычную reversible version.</p>${b.human_correction_evidence_ids.map((id) => `<div class="qc-check"><span>${short(id, 18)}</span><span class="badge good">human-correction</span></div>`).join("") || `<p class="muted">Пока нет человеческих коррекций.</p>`}</article>
    </div>
    <article class="card" style="margin-top:16px"><div class="card-head"><h3>Preview typed diff · approve/reject поэлементно</h3><span class="badge">no direct FFmpeg / no direct state write</span></div>${motionProposalMarkup(state.motionProposal)}</article>
    <article class="card" style="margin-top:16px"><div class="card-head"><h3>A / B версий после коррекции</h3><div class="button-row"><span class="badge">synchronized</span><button class="secondary" data-action="reference-render-segment" data-start="${reviewedRange.start}" data-duration="${reviewedRange.duration}">Point rerender affected ${reviewedRange.start}–${reviewedRange.start + reviewedRange.duration - 1}</button></div></div><div class="compare-grid"><div class="compare-pane"><select id="reference-version-select-a">${versionOptions(base)}</select><div id="reference-version-a-pane">${referenceVersionVideo(base, "a")}</div></div><div class="compare-pane"><select id="reference-version-select-b">${versionOptions(current)}</select><div id="reference-version-b-pane">${referenceVersionVideo(current, "b")}</div></div></div></article>`;
  $("#reference-opacity")?.addEventListener("input", (event) => { $("#reference-player").style.opacity = event.target.value; });
  $("#reference-frame")?.addEventListener("input", (event) => updateReferenceFrame(event.target.value));
  $("#reference-channel")?.addEventListener("change", (event) => { state.referenceChannel = event.target.value; renderReference(); });
  $("#reference-derivative")?.addEventListener("change", (event) => { state.referenceDerivative = event.target.value; renderReference(); });
  ["a", "b"].forEach((side) => $(`#reference-version-select-${side}`)?.addEventListener("change", (event) => {
    const version = b.versions.find((item) => item.id === event.target.value);
    $(`#reference-version-${side}-pane`).innerHTML = referenceVersionVideo(version, side);
    bindSynchronizedMedia(["reference-version-player-a", "reference-version-player-b"]);
  }));
  bindSynchronizedMedia(["reference-player", "candidate-player", "difference-player"]);
  bindSynchronizedMedia(["reference-version-player-a", "reference-version-player-b"]);
  updateReferenceFrame(state.referenceFrame, false);
}

function versionOption(version) { return `<option value="${version.id}">${esc(version.message)} · ${short(version.id)}</option>`; }
function videoFor(version, profile = "preview") {
  if (!version) return `<div class="drop-zone muted">Выберите версию</div>`;
  const job = state.bundle.jobs.find((item) => item.version_id === version.id && item.kind === `render.${profile}` && item.status === "succeeded");
  if (!job) return `<div class="drop-zone muted">Для версии ещё нет ${profile}-рендера.</div>`;
  const path = `renders/${version.id}/${profile}/video.mp4`;
  return `<video controls preload="metadata" src="/api/projects/${state.bundle.project.id}/file?path=${encodeURIComponent(path)}"></video>`;
}

function renderVersions() {
  const b = state.bundle;
  const current = currentVersion();
  $("#tab-versions").innerHTML = `<div class="grid two">
    <article class="card"><div class="card-head"><h3>История версий</h3><span class="badge">${b.versions.length}</span></div><div class="version-list">${[...b.versions].reverse().map((version) => `<div class="version ${version.id === b.project.current_version_id ? "current" : ""}"><div class="version-head"><div><strong>${esc(version.message)}</strong><small> ${short(version.id)} · ${seconds(version.timeline.duration_frames, version.timeline.frame_rate)} сек.</small></div>${version.id === b.project.current_version_id ? `<span class="badge good">current</span>` : `<button class="ghost" data-action="revert" data-version="${version.id}">Revert как новую</button>`}</div></div>`).join("") || `<p class="muted">Версий пока нет.</p>`}</div></article>
    <article class="card"><div class="card-head"><h3>Что хранится</h3></div><p>Каждая версия — неизменяемый snapshot edit graph. Patch знает base и inverse; рендер и экспорт могут быть пересозданы.</p><textarea class="code" readonly>${esc(current ? JSON.stringify(current.timeline, null, 2) : "{}")}</textarea></article>
  </div>
  <article class="card" style="margin-top:16px"><div class="card-head"><h3>A / B просмотр</h3><span class="badge">preview</span></div><div class="compare-grid">
    <div class="compare-pane"><select id="compare-a"><option value="">A: версия</option>${b.versions.map(versionOption).join("")}</select><div id="compare-a-video"></div></div>
    <div class="compare-pane"><select id="compare-b"><option value="">B: версия</option>${b.versions.map(versionOption).join("")}</select><div id="compare-b-video"></div></div>
  </div></article>`;
  ["a","b"].forEach((side) => $(`#compare-${side}`)?.addEventListener("change", (event) => {
    const version = b.versions.find((item) => item.id === event.target.value);
    $(`#compare-${side}-video`).innerHTML = videoFor(version);
  }));
}

function latestQc(version, profile) {
  return [...state.bundle.qc].reverse().find((item) => item.version_id === version?.id && item.render_path.includes(`/${profile}/`));
}

function renderDeliver() {
  const b = state.bundle;
  const version = currentVersion();
  const qc = latestQc(version, "preview");
  const exports = version ? b.interchange.filter((item) => item.version_id === version.id) : [];
  $("#tab-deliver").innerHTML = `<div class="grid two">
    <article class="card render-card"><div class="card-head"><h3>Preview</h3><span class="badge">fast · review</span></div>${videoFor(version, "preview")}<div class="button-row"><button class="primary" data-action="render" data-profile="preview" ${version ? "" : "disabled"}>Render preview</button><button class="secondary" data-action="qc" data-profile="preview" ${version ? "" : "disabled"}>QC</button></div></article>
    <article class="card render-card"><div class="card-head"><h3>Final</h3><span class="badge">full resolution</span></div>${videoFor(version, "final")}<div class="button-row"><button class="primary" data-action="render" data-profile="final" ${version ? "" : "disabled"}>Render final</button><button class="secondary" data-action="qc" data-profile="final" ${version ? "" : "disabled"}>QC</button></div></article>
  </div>
  <div class="grid two" style="margin-top:16px"><article class="card"><div class="card-head"><h3>Quality report</h3><span class="badge ${qc?.overall === "pass" ? "good" : qc ? "warn" : ""}">${qc?.overall || "not run"}</span></div><div class="qc-list">${qc?.checks.map((check) => `<div class="qc-check"><span>${esc(check.summary)}</span><span class="badge ${check.status === "pass" ? "good" : check.status === "fail" ? "fail" : "warn"}">${check.status}</span></div>`).join("") || `<p class="muted">Запустите QC после рендера.</p>`}</div><details><summary>Фоновые задачи и receipts</summary>${[...b.jobs].reverse().map((job) => `<div class="qc-check"><span>${esc(job.kind)}<small>${job.error ? esc(job.error) : short(job.id)}</small></span><span class="badge ${job.status === "succeeded" ? "good" : job.status === "failed" ? "fail" : "warn"}">${esc(job.status)}</span></div>`).join("") || `<p class="muted">Задач ещё нет.</p>`}</details></article>
    <article class="card"><div class="card-head"><h3>Editable handoff</h3><span class="badge">derived</span></div><p>Экспорт не заменяет канонический edit graph и всегда сопровождается compatibility report.</p><div class="button-row"><button class="secondary" data-action="export" data-format="kdenlive" ${version ? "" : "disabled"}>Kdenlive / MLT</button><button class="secondary" data-action="export" data-format="otio" ${version ? "" : "disabled"}>OpenTimelineIO</button></div>${exports.map((item) => `<a class="artifact-link" href="/api/projects/${b.project.id}/file?path=${encodeURIComponent(item.output_path)}">${esc(item.format)} · ${esc(item.output_path)}</a>`).join("")}</article></div>`;
}

async function uploadAsset(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  await guarded("Копирую и проверяю исходник…", async () => {
    await api(`/api/projects/${state.bundle.project.id}/assets`, {method:"POST", body:data});
    toast("Исходник добавлен неизменяемо");
    await selectProject(state.bundle.project.id);
  });
}

async function reviseBrief(event) {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const intent = state.bundle.project.intent;
  await guarded("Сохраняю новую ревизию Brief…", async () => {
    await api(`/api/projects/${state.bundle.project.id}/briefs`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({intent:data.get("intent"), scenario:intent.scenario, target_duration_seconds:intent.target_duration_seconds, audience:intent.audience, frame_format:intent.frame_format, tempo:data.get("tempo") || null, mood:data.get("mood") || null, required_elements:intent.required_elements, forbidden_elements:intent.forbidden_elements, sound_requirements:intent.sound_requirements, privacy_mode:intent.privacy_mode, automation_level:intent.automation_level, rationale:data.get("rationale")})});
    toast("Новая ревизия Brief сохранена"); await selectProject(state.bundle.project.id);
  });
}

async function confirmStyle(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  await guarded("Сохраняю явное подтверждение…", async () => {
    await api("/api/style-profiles", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({scope:state.bundle.project.id, explicit_opt_in:true, confirmations:[{preference:data.get("preference"), value:data.get("value")}]})});
    toast("Предпочтение сохранено только по явному подтверждению"); form.reset();
  });
}

async function handleAction(button) {
  const action = button.dataset.action;
  const projectId = state.bundle?.project.id;
  if (!projectId) return;
  if (action === "analyze") await guarded("Извлекаю evidence…", async () => {
    await api(`/api/projects/${projectId}/assets/${button.dataset.asset}/analyze`, {method:"POST", headers:{"Content-Type":"application/json"}, body:"{\"transcribe\":false}"});
    toast("Анализ завершён"); await selectProject(projectId);
  });
  if (action === "anatomy-run") await guarded("Строю Video Anatomy и durable checkpoints…", async () => {
    const assetId = $("#anatomy-asset").value;
    const pinnedFrames = String($("#anatomy-pins").value || "").split(",").map((item) => Number(item.trim())).filter((item) => Number.isInteger(item) && item >= 0);
    const result = await api(`/api/projects/${projectId}/assets/${assetId}/video-anatomy`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({profile:$("#anatomy-profile").value, pinned_frames:pinnedFrames, explicit_provider_opt_in:$("#anatomy-provider-opt-in").checked})});
    state.anatomyPlanId = result.anatomy.plan.id;
    toast(`Video Anatomy: ${result.anatomy.status}; ${result.anatomy.structure.shots.length} shots`);
    await selectProject(projectId);
  });
  if (action === "anatomy-estimate") await guarded("Оцениваю decode, артефакты и свободное место…", async () => {
    const assetId = $("#anatomy-asset").value;
    state.anatomyEstimate = await api(`/api/projects/${projectId}/assets/${assetId}/video-anatomy/estimate`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({profile:$("#anatomy-profile").value})});
    renderAnatomy();
    toast(state.anatomyEstimate.admitted ? "Resource preflight допускает запуск" : "Resource preflight отказал в запуске", !state.anatomyEstimate.admitted);
  });
  if (action === "anatomy-select-shot") {
    document.querySelector(`[data-shot-card="${button.dataset.shot}"]`)?.scrollIntoView({behavior:"smooth", block:"center"});
  }
  if (action === "anatomy-pin-frame") {
    const input = $("#anatomy-pins");
    const values = new Set(String(input.value || "").split(",").map((item) => item.trim()).filter(Boolean));
    values.add(String(button.dataset.frame));
    input.value = [...values].sort((left, right) => Number(left) - Number(right)).join(", ");
    toast(`Frame ${button.dataset.frame} закреплён для следующего прохода`);
  }
  if (action === "anatomy-deepen-shot") await guarded("Углубляю только выбранный shot…", async () => {
    const anatomy = anatomyForView();
    const result = await api(`/api/projects/${projectId}/video-anatomy/${anatomy.plan.id}/shots/${button.dataset.shot}/deepen`, {method:"POST"});
    state.anatomyPlanId = result.anatomy.plan.id;
    toast("Focused motion pass завершён; исходный агрегат сохранён");
    await selectProject(projectId);
  });
  if (action === "anatomy-focused-plans") await guarded("Строю планы только для пробелов и неоднозначностей…", async () => {
    const anatomy = anatomyForView();
    state.anatomyFocusedPlans = await api(`/api/projects/${projectId}/video-anatomy/${anatomy.plan.id}/focused-plans`, {method:"POST"});
    renderAnatomy();
    toast(`${state.anatomyFocusedPlans.length} focused plans; анализ ещё не запущен`);
  });
  if (action === "anatomy-correct-evidence") await guarded("Записываю human correction без перезаписи analyzer evidence…", async () => {
    const payload = JSON.parse($("#anatomy-correction-payload").value);
    const rationale = $("#anatomy-correction-rationale").value.trim();
    if (!rationale) throw new Error("Нужна причина human correction");
    await api(`/api/projects/${projectId}/evidence/corrections`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({asset_id:button.dataset.asset, kind:button.dataset.kind, payload, supersedes:[button.dataset.evidence], rationale})});
    toast("Human correction сохранён отдельным evidence");
    await selectProject(projectId);
  });
  if (action === "anatomy-create-editorial") await guarded("Формирую editorial proposal отдельно от evidence…", async () => {
    const anatomy = anatomyForView();
    await api(`/api/projects/${projectId}/video-anatomy/${anatomy.plan.id}/editorial-proposals`, {method:"POST"});
    toast("Editorial proposal создан; timeline не изменён");
    await selectProject(projectId);
  });
  if (action === "anatomy-create-reconstruction") await guarded("Формирую source-neutral reconstruction proposal…", async () => {
    const anatomy = anatomyForView();
    const targetAssetId = $("#anatomy-target-asset").value || null;
    const baseVersionId = $("#anatomy-base-version").value || null;
    if ((targetAssetId && !baseVersionId) || (!targetAssetId && baseVersionId)) throw new Error("Target asset и base version нужны вместе для patch preview");
    await api(`/api/projects/${projectId}/video-anatomy/${anatomy.plan.id}/reconstruction-proposals`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({target_asset_id:targetAssetId, base_version_id:baseVersionId})});
    toast("Reconstruction proposal создан; reference pixels не встроены");
    await selectProject(projectId);
  });
  if (action === "anatomy-review-proposal") await guarded("Записываю явное человеческое решение…", async () => {
    const proposalId = button.dataset.proposal;
    const reviewer = document.querySelector(`[data-reviewer="${proposalId}"]`).value.trim();
    const rationale = document.querySelector(`[data-rationale="${proposalId}"]`).value.trim();
    if (!reviewer || !rationale) throw new Error("Для review нужны автор и причина");
    const collection = button.dataset.kind === "editorial" ? "video-editorial-proposals" : "video-reconstruction-proposals";
    await api(`/api/projects/${projectId}/${collection}/${proposalId}/review`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({decision:button.dataset.decision, reviewer, rationale})});
    toast(`Proposal ${button.dataset.decision}; canonical timeline не изменён`);
    await selectProject(projectId);
  });
  if (action === "anatomy-accept-reconstruction") await guarded("Провожу approved proposal через обычный reversible patch…", async () => {
    await api(`/api/projects/${projectId}/video-reconstruction-proposals/${button.dataset.proposal}/accept/${button.dataset.review}`, {method:"POST"});
    toast("Создана canonical version с inverse operations");
    await selectProject(projectId);
  });
  if (action === "propose") await guarded("Собираю treatment…", async () => {
    const duration = Number($("#proposal-duration")?.value) || null;
    const options = await api(`/api/projects/${projectId}/treatments/options`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({duration_seconds:duration})});
    toast(`${options.length} treatment-варианта готовы для проверки`); await selectProject(projectId);
  });
  if (action === "accept") await guarded("Создаю неизменяемую версию…", async () => {
    await api(`/api/projects/${projectId}/treatments/${button.dataset.treatment}/accept`, {method:"POST"});
    toast("Treatment принят как новая версия"); await selectProject(projectId);
  });
  if (action === "apply-edited") await guarded("Проверяю и применяю patch…", async () => {
    const treatment = state.bundle.treatments.find((item) => item.id === button.dataset.treatment);
    const operations = JSON.parse($(`#patch-${treatment.id}`).value);
    await api(`/api/projects/${projectId}/patches`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({base_version_id:treatment.base_version_id, operations, rationale:`Edited: ${treatment.patch.rationale}`, message:`Edited treatment: ${treatment.title}`, evidence_refs:treatment.patch.evidence_refs})});
    toast("Изменённый patch стал новой версией"); await selectProject(projectId);
  });
  if (action === "language-preview") await guarded("Преобразую команду в проверяемый diff…", async () => {
    const command = $("#language-command")?.value || "";
    state.languagePreview = await api(`/api/projects/${projectId}/patches/preview-language`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({command, version_id:currentVersion().id})});
    renderEdit(); toast("Diff готов; проект ещё не изменён");
  });
  if (action === "language-apply") await guarded("Применяю подтверждённый typed patch…", async () => {
    const preview = state.languagePreview;
    await api(`/api/projects/${projectId}/patches`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({base_version_id:preview.base_version_id, operations:preview.patch.operations, rationale:preview.patch.rationale, message:`Natural language: ${preview.summary}`, evidence_refs:preview.patch.evidence_refs})});
    state.languagePreview = null; toast("Подтверждённый patch стал новой версией"); await selectProject(projectId);
  });
  if (action === "reference-jump-frame") updateReferenceFrame(Number(button.dataset.frame));
  if (action === "reference-language-preview") await guarded("Разбираю просьбу в независимые motion operations…", async () => {
    const workspace = state.referenceBundle.workspace;
    state.motionProposal = await api(`/api/projects/${projectId}/reference-workspaces/${workspace.id}/corrections/preview-language`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({command:$("#reference-command").value, version_id:currentVersion().id})});
    renderReference(); toast("Typed diff готов; canonical state не изменён");
  });
  if (action === "reference-tangent-preview") await guarded("Строю bounded tangent preview…", async () => {
    const workspace = state.referenceBundle.workspace;
    const start = Number($("#tangent-start").value);
    const end = Number($("#tangent-end").value);
    const operation = {kind:"adjust_tangent", channel:$("#tangent-channel").value, anchor_frame:Number($("#tangent-anchor").value), strength:Number($("#tangent-strength").value), affected_range:{start, duration:end - start + 1}};
    state.motionProposal = await api(`/api/projects/${projectId}/reference-workspaces/${workspace.id}/corrections/preview-operations`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({operations:[operation], version_id:currentVersion().id})});
    renderReference(); toast("Tangent diff готов к отдельному решению");
  });
  if (action === "reference-phase-preview") await guarded("Строю phase-boundary preview…", async () => {
    const workspace = state.referenceBundle.workspace;
    const boundary = $("#phase-boundary").value;
    const original = workspace.phase_markers[boundary];
    const delta = Number($("#phase-delta").value);
    const start = Math.max(0, original - 60);
    const end = Math.min(workspace.frame_count, original + 61);
    const operation = {kind:"move_phase_boundary", boundary, original_frame:original, delta_frames:delta, affected_range:{start, duration:end - start}};
    state.motionProposal = await api(`/api/projects/${projectId}/reference-workspaces/${workspace.id}/corrections/preview-operations`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({operations:[operation], version_id:currentVersion().id})});
    renderReference(); toast("Phase diff готов к отдельному решению");
  });
  if (action === "reference-pivot-preview") await guarded("Проверяю идентифицируемость pivot…", async () => {
    const workspace = state.referenceBundle.workspace;
    const start = workspace.phase_markers.twist_start;
    const end = workspace.phase_markers.twist_end;
    const operation = {kind:"adjust_pivot_curve", delta_x:.01, delta_y:0, affected_range:{start, duration:end - start + 1}};
    state.motionProposal = await api(`/api/projects/${projectId}/reference-workspaces/${workspace.id}/corrections/preview-operations`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({operations:[operation], version_id:currentVersion().id})});
    renderReference(); toast("Pivot-гипотеза проверена без выдумывания параметров");
  });
  if (action === "reference-review") await guarded("Проверяю все approve/reject решения…", async () => {
    const proposal = state.motionProposal;
    const decisions = proposal.items.map((item) => {
      const selected = document.querySelector(`input[name="decision-${item.id}"]:checked`);
      const rationale = document.querySelector(`.decision-rationale[data-correction="${item.id}"]`)?.value.trim();
      if (!selected || !rationale) throw new Error(`Для ${item.operation.kind} нужны решение и причина`);
      return {correction_id:item.id, decision:selected.value, rationale};
    });
    const review = await api(`/api/projects/${projectId}/motion-correction-proposals/${proposal.id}/review`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({decisions})});
    state.motionProposal = null;
    toast(review.outcome === "applied" ? "Одобренные corrections стали новой reversible version" : "Все corrections отклонены; human evidence сохранён");
    await selectProject(projectId);
  });
  if (action === "reference-render-segment") await guarded("Рендерю только затронутый motion-диапазон…", async () => {
    const version = currentVersion();
    await api(`/api/projects/${projectId}/versions/${version.id}/render?profile=preview&start_frame=${button.dataset.start}&duration_frames=${button.dataset.duration}`, {method:"POST"});
    toast("Точечный reference-motion preview создан"); await selectProject(projectId);
  });
  if (action === "timeline-pin" || action === "timeline-toggle") await guarded("Создаю прямую типизированную правку…", async () => {
    const version = currentVersion();
    const field = action === "timeline-pin" ? "pinned" : "enabled";
    const value = button.dataset.value === "true";
    await api(`/api/projects/${projectId}/patches`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({base_version_id:version.id, operations:[{op:"replace", path:`/tracks/${button.dataset.track}/clips/${button.dataset.clip}/${field}`, value}], rationale:`Direct editor ${field} change`, message:`Set clip ${field}=${value}`, evidence_refs:[]})});
    toast("Прямая правка записана новой версией"); await selectProject(projectId);
  });
  if (action === "render-segment") await guarded("Рендерю только затронутый диапазон…", async () => {
    const version = currentVersion();
    await api(`/api/projects/${projectId}/versions/${version.id}/render?profile=preview&start_frame=${button.dataset.start}&duration_frames=${button.dataset.duration}`, {method:"POST"});
    toast("Точечный preview создан"); await selectProject(projectId);
  });
  if (action === "revert") await guarded("Создаю обратную версию…", async () => {
    await api(`/api/projects/${projectId}/versions/${button.dataset.version}/revert`, {method:"POST"});
    toast("Revert записан новой версией"); await selectProject(projectId);
  });
  if (action === "render") await guarded(`Рендер ${button.dataset.profile}…`, async () => {
    const version = currentVersion();
    await api(`/api/projects/${projectId}/versions/${version.id}/render?profile=${button.dataset.profile}`, {method:"POST"});
    toast("Рендер готов"); await selectProject(projectId);
  });
  if (action === "qc") await guarded("Проверяю рендер…", async () => {
    const version = currentVersion();
    await api(`/api/projects/${projectId}/versions/${version.id}/qc?profile=${button.dataset.profile}`, {method:"POST"});
    toast("QC завершён"); await selectProject(projectId);
  });
  if (action === "export") await guarded("Собираю редактируемую передачу…", async () => {
    const version = currentVersion();
    await api(`/api/projects/${projectId}/versions/${version.id}/export/${button.dataset.format}`, {method:"POST"});
    toast("Экспорт создан и проверен"); await selectProject(projectId);
  });
}

function openDialog() { $("#project-dialog").showModal(); }
document.addEventListener("click", (event) => {
  const project = event.target.closest("[data-project]");
  if (project) selectProject(project.dataset.project);
  const action = event.target.closest("[data-action]");
  if (action) handleAction(action);
  if (event.target.closest("#new-project-button") || event.target.closest("#welcome-create")) openDialog();
  if (event.target.closest("[data-close-dialog]")) $("#project-dialog").close();
});

document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => {
  state.tab = button.dataset.tab;
  document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === button));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${state.tab}`));
}));

$("#project-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  await guarded("Создаю проект…", async () => {
    const list = (name) => String(data.get(name) || "").split(",").map((item) => item.trim()).filter(Boolean);
    const payload = {name:data.get("name"), scenario:data.get("scenario"), intent:data.get("intent"), target_duration_seconds:Number(data.get("duration")) || null, audience:data.get("audience") || null, frame_format:data.get("frame_format") || null, tempo:data.get("tempo") || null, mood:data.get("mood") || null, required_elements:list("required_elements"), forbidden_elements:list("forbidden_elements"), sound_requirements:list("sound_requirements"), privacy_mode:data.get("privacy_mode"), automation_level:data.get("automation_level")};
    const project = await api("/api/projects", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
    $("#project-dialog").close(); form.reset(); toast("Проект создан"); await loadProjects(project.id);
  });
});

$("#refresh-button").addEventListener("click", () => guarded("Обновляю состояние…", async () => { await loadProjects(); toast("Состояние обновлено"); }));
await Promise.all([loadHealth(), loadProjects()]);
