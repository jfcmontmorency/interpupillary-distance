import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { toGray, refineCard, sampleEdges } from "./cardDetect.js";
import { calibrate, undistort, projectCard } from "./calib.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const CARD_W = 85.6;   // ISO/IEC 7810 ID-1
const CARD_H = 53.98;
const IRIS_D = 11.7;   // diamètre visible moyen de l'iris (HVID)
const MAX_IMG = 4000;  // limite de taille pour les photos importées

const $ = (s) => document.querySelector(s);

// ---------- géométrie ----------
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
const mul = (a, k) => ({ x: a.x * k, y: a.y * k });
const dot = (a, b) => a.x * b.x + a.y * b.y;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const norm = (a) => mul(a, 1 / Math.hypot(a.x, a.y));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const deg = (r) => (r * 180) / Math.PI;
const median = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const fmt = (v, d = 1) => v.toFixed(d).replace(".", ",");

// ---------- réglages ----------
const DEFAULTS = { mode: "card", distanceCm: 75, deltaMm: 12, rotMm: 10, recTarget: 40 };
const NEAR_CM = 40; // distance de lecture type, pour l'EP de près (détails techniques uniquement)
const settings = (() => {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("ipd-settings") || "{}") }; }
  catch { return { ...DEFAULTS }; }
})();
function saveSettings() {
  try { localStorage.setItem("ipd-settings", JSON.stringify(settings)); } catch {}
}
for (const key of Object.keys(DEFAULTS)) {
  const el = $("#" + key);
  el.value = settings[key];
  el.addEventListener("change", () => {
    settings[key] = key === "mode" ? el.value : parseFloat(el.value) || DEFAULTS[key];
    saveSettings();
    if (editor) { renderEditor(); updateEditResult(); }
    renderResults();
  });
}

/** Facteur qui ramène la mesure brute à l'EP de loin (parallaxe + convergence). */
function farFactor(mode, Dmm = null) {
  // distance mesurée automatiquement (caméra calibrée), sinon celle saisie
  const D = Dmm || settings.distanceCm * 10;
  const extra = (mode === "card" ? settings.deltaMm : 0) + settings.rotMm;
  return (D + extra) / D;
}
/** EP de près à la distance de lecture, à partir de l'EP de loin. */
function nearFromFar(far) {
  const N = NEAR_CM * 10;
  return (far * N) / (N + settings.rotMm);
}

// ---------- détection ----------
let landmarker = null;
let lastTs = 0;
const nextTs = () => (lastTs = Math.max(performance.now(), lastTs + 1));

async function initLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });
  try { landmarker = await FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch { landmarker = await FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
}

function analyze(result, W, H) {
  if (!result?.faceLandmarks?.length) return null;
  const lm = result.faceLandmarks[0];
  const P = (i) => ({ x: lm[i].x * W, y: lm[i].y * H });
  // version 3D (z de MediaPipe : même échelle que x)
  const P3 = (i) => [lm[i].x * W, lm[i].y * H, lm[i].z * W];
  const avg3 = (ids) => [0, 1, 2].map((k) => ids.reduce((acc, i) => acc + P3(i)[k], 0) / ids.length);
  const iris = (c, ring) => {
    const pts = ring.map(P);
    const ringC = mul(pts.reduce(add, { x: 0, y: 0 }), 1 / pts.length);
    const ring3 = avg3(ring), c3 = P3(c);
    return {
      center: mid(P(c), ringC),
      center3: [0, 1, 2].map((k) => (c3[k] + ring3[k]) / 2),
      diam: (dist(pts[0], pts[2]) + dist(pts[1], pts[3])) / 2,
    };
  };
  const a = iris(468, [469, 470, 471, 472]);
  const b = iris(473, [474, 475, 476, 477]);
  // OD = œil droit du sujet = à gauche dans l'image non inversée
  const [od, og] = a.center.x < b.center.x ? [a, b] : [b, a];

  let yaw = 0, pitch = 0, roll = 0;
  const m = result.facialTransformationMatrixes?.[0]?.data;
  if (m) {
    const R = (i, j) => m[j * 4 + i];
    yaw = deg(Math.asin(clamp(-R(2, 0), -1, 1)));
    pitch = deg(Math.atan2(R(2, 1), R(2, 2)));
    roll = deg(Math.atan2(R(1, 0), R(0, 0)));
  }
  // ouverture des paupières rapportée au diamètre d'iris (≈ 0,8 œil ouvert, < 0,2 pendant un clignement)
  const irisD = (od.diam + og.diam) / 2;
  const openness = Math.min(dist(P(159), P(145)), dist(P(386), P(374))) / irisD;
  // position du nez entre les deux pupilles (0 = OD, 1 = OG), calculée en 3D : une tête
  // légèrement tournée décale le nez (en avant du plan des yeux) mais ne change pas ce rapport
  const B3 = P3(168);
  const u3 = [0, 1, 2].map((k) => og.center3[k] - od.center3[k]);
  const b3 = [0, 1, 2].map((k) => B3[k] - od.center3[k]);
  const t3 = clamp((b3[0] * u3[0] + b3[1] * u3[1] + b3[2] * u3[2]) / (u3[0] ** 2 + u3[1] ** 2 + u3[2] ** 2), 0, 1);
  const bridge = P(168);
  const u2 = sub(og.center, od.center);
  const t2 = clamp(dot(sub(bridge, od.center), u2) / dot(u2, u2), 0, 1);
  return { od, og, bridge, yaw, pitch, roll, openness, t3, tCorr: t3 - t2 };
}

/**
 * Échelle tirée de la carte, robuste à une carte pivotée ou inclinée. Une carte pivotée
 * (gauche/droite) paraît moins large, une carte inclinée (haut/bas) moins haute : le raccourci
 * ne fait que réduire une dimension, donc la vraie échelle est donnée par la plus grande.
 * tilt = raccourcissement relatif de l'autre dimension (≈ 1 − cos de l'angle).
 */
function cardScale(c) {
  const kW = cardWidth(c) / CARD_W;
  const kH = cardHeight(c) / CARD_H;
  const hi = Math.max(kW, kH), lo = Math.min(kW, kH);
  const tilt = 1 - lo / hi;
  // carte face à la caméra (écart dans le bruit) : moyenne pondérée par la taille en pixels
  const k = tilt < 0.003 ? (kW * CARD_W + kH * CARD_H) / (CARD_W + CARD_H) : hi;
  return { mmPerPx: 1 / k, tilt, tiltDeg: deg(Math.acos(1 - tilt)) };
}
// Le front est incliné vers l'arrière (souvent 10–20°) : une carte posée dessus paraît moins haute.
// L'échelle tirée de la dimension non raccourcie corrige ce cas ; on ne rejette qu'au-delà de ≈ 30°.
const MAX_CARD_TILT = 0.14;
const CARD_ASPECT_TOL = 0.15; // tolérance sur les proportions apparentes (carte inclinée sur le front)

function irisEstimate(a) {
  const raw = (dist(a.od.center, a.og.center) / ((a.od.diam + a.og.diam) / 2)) * IRIS_D;
  return raw * farFactor("iris");
}

function poseIssues(a) {
  const out = [];
  if (Math.abs(a.yaw) > 3) out.push("tournez-vous face à l'écran");
  if (Math.abs(a.pitch) > 10) out.push("gardez la tête droite");
  if (Math.abs(a.roll) > 6) out.push("ne penchez pas la tête");
  return out;
}

// ---------- caméra ----------
const video = $("#video");
const overlay = $("#overlay");
const octx = overlay.getContext("2d");
let liveRunning = false;
let lastVideoTime = -1;
let lastAnalysis = null;
const liveSamples = [];
let burst = null; // { remaining, best, bestScore }

// ---------- navigation par étapes ----------
function setView(name) {
  document.body.dataset.view = name;
  if (name !== "measure" && rec.active) stopRecording();
  if (name !== "edit" && video.srcObject) setStatus("Caméra active");
  window.scrollTo(0, 0);
}
function goMeasure() {
  setView("measure");
  startCamera();
}
$("#startBtn").addEventListener("click", goMeasure);
document.querySelectorAll("[data-go]").forEach((b) =>
  b.addEventListener("click", () => {
    const v = b.dataset.go;
    if (v === "measure") goMeasure();
    else if (v === "results") showResults();
    else setView(v);
  })
);
$("#captureBtn").addEventListener("click", captureWithTimer);
$("#switchCamBtn").addEventListener("click", async () => {
  if (rec.active) stopRecording();
  video.srcObject?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  facing = facing === "user" ? "environment" : "user";
  $("#switchCamBtn").textContent = facing === "user" ? "Utiliser la caméra arrière" : "Utiliser la caméra avant";
  await startCamera();
});

let facing = "user"; // "user" = caméra avant (selfie), "environment" = caméra arrière

async function startCamera() {
  if (video.srcObject) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    // effet miroir seulement pour la caméra avant
    $("#videoWrap").classList.toggle("rear", facing === "environment");
    refreshCamCal();
    lastVideoTime = -1;
    liveCard.corners = null;
    // le bouton de changement n'apparaît que s'il y a plusieurs caméras
    try {
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      $("#switchCamBtn").hidden = cams.length < 2;
    } catch {}
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    setStatus("Caméra active");
    // la boucle démarre tout de suite (la calibration n'a pas besoin du modèle du visage)
    $("#hud").hidden = false;
    if (!liveRunning) { liveRunning = true; requestAnimationFrame(liveLoop); }
    await modelReady;
    $("#captureBtn").disabled = false;
    $("#recBtn").disabled = false;
  } catch (e) {
    setStatus("Caméra indisponible : " + e.message);
  }
}

function liveLoop() {
  if (!liveRunning) return;
  if ((landmarker || calib.active) && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    if (calib.active) {
      calibStep();
      requestAnimationFrame(liveLoop);
      return;
    }
    const a = analyze(landmarker.detectForVideo(video, nextTs()), video.videoWidth, video.videoHeight);
    lastAnalysis = a;
    irisSmooth = smoothIris(a);
    if (settings.mode === "card" && !burst) liveCardStep(a);
    drawOverlay(a);
    updateLive(a);
    const q = liveQuality(a);
    showQuality(q);
    if (!rec.active) $("#hudCount").textContent = liveDistText();
    if (rec.active) recordStep(a, q);
    if (burst) burstStep(a);
  }
  requestAnimationFrame(liveLoop);
}

// ---------- voyant et enregistrement automatique ----------

/** État global de la mesure en direct : bad (rouge), warn (orange), ok (vert). */
function liveQuality(a) {
  if (!a) return { level: "bad", msg: "Je ne vous vois pas" };
  if (Math.abs(a.yaw) > 3) return { level: "warn", msg: "Tournez-vous face à l'écran" };
  if (Math.abs(a.pitch) > 10 || Math.abs(a.roll) > 6) return { level: "warn", msg: "Gardez la tête droite" };
  if (a.openness < 0.3) return { level: "warn", msg: "Gardez les yeux ouverts" };
  if (settings.mode === "card") {
    // la carte reste « vue » malgré un ou deux ratés ponctuels ; chaque mesure est de toute façon
    // revalidée en pleine résolution avant d'être enregistrée
    if (!liveCard.corners) return { level: "warn", msg: "Placez la carte dans le cadre orange" };
    if (cardScale(liveCard.corners).tilt > MAX_CARD_TILT) return { level: "warn", msg: "Remettez la carte bien à plat" };
  }
  return { level: "ok", msg: rec.active ? "Parfait, ne bougez plus" : "Tout est bon !" };
}

let lastLevel = null;
/** Le contour de la vidéo sert de voyant ; le message s'affiche en haut de la vidéo. */
function showQuality(q) {
  if (q.level !== lastLevel) {
    const w = $("#videoWrap");
    w.classList.remove("bad", "warn", "ok");
    w.classList.add(q.level);
    lastLevel = q.level;
  }
  $("#hudMsg").textContent = q.msg;
}

const rec = { active: false, count: 0, okSince: null, nextAt: 0, audio: null };
const fullCanvas = document.createElement("canvas");

function beep(freq = 880, dur = 0.06, when = 0) {
  const ac = rec.audio;
  if (!ac) return;
  const o = ac.createOscillator(), g = ac.createGain();
  o.frequency.value = freq;
  const t0 = ac.currentTime + when;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(0.25, t0 + 0.005);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(ac.destination);
  o.start(t0);
  o.stop(t0 + dur + 0.02);
}

$("#recBtn").addEventListener("click", () => {
  if (calib.active) return;
  if (rec.active) return stopRecording();
  try { rec.audio ??= new (window.AudioContext || window.webkitAudioContext)(); rec.audio.resume(); } catch {}
  measurements.length = 0;
  renderResults();
  Object.assign(rec, { active: true, count: 0, okSince: null, nextAt: 0 });
  $("#recBtn").textContent = "■ Arrêter";
  $("#videoWrap").classList.add("recording");
  updateProgress();
});

function stopRecording() {
  rec.active = false;
  $("#recBtn").textContent = "● Démarrer la mesure";
  $("#videoWrap").classList.remove("recording");
  updateProgress();
}

function updateProgress() {
  const target = settings.recTarget;
  const pct = rec.active || rec.count ? Math.min(100, (100 * rec.count) / target) : 0;
  $("#recBar").style.width = pct + "%";
  $("#hudCount").textContent = rec.active ? `${rec.count} / ${target}` : "";
}

/** Mesure en pleine résolution sur l'image courante ; null si la carte n'est pas retrouvée précisément. */
function measureLive(a) {
  let mmPerPx;
  if (settings.mode === "card") {
    fullCanvas.width = video.videoWidth;
    fullCanvas.height = video.videoHeight;
    fullCanvas.getContext("2d", { willReadFrequently: true }).drawImage(video, 0, 0);
    const r = findCard(toGray(fullCanvas), [
      { corners: liveCard.corners, frac: 0.05 },
      { corners: liveCard.corners, frac: 0.12 },
    ], expectedCardW(a));
    if (r.error) return null;
    // caméra calibrée : coins corrigés de la distorsion de l'objectif
    const sc = cardScale(r.corners.map((p) => undistort(camCal, p)));
    if (sc.tilt > MAX_CARD_TILT) return null;
    mmPerPx = sc.mmPerPx;
  } else {
    mmPerPx = IRIS_D / ((a.od.diam + a.og.diam) / 2);
  }
  // les pupilles d'une tête tournée de θ paraissent rapprochées d'un facteur cos θ
  const yawCos = Math.cos((a.yaw * Math.PI) / 180);
  const od = undistort(camCal, a.od.center), og = undistort(camCal, a.og.center);
  return {
    mode: settings.mode,
    source: "auto",
    raw: (dist(od, og) * mmPerPx) / yawCos,
    t: a.t3,
    yaw: a.yaw,
    // distance caméra → carte : focale (px) × mm par px
    // distance figée au moment de la mesure : mesurée (caméra calibrée) ou saisie ; changer le
    // réglage ensuite ne doit pas réécrire les mesures déjà prises
    ...measDistance(settings.mode === "card" && camCal?.f ? camCal.f * mmPerPx : null),
  };
}

function recordStep(a, q) {
  const now = performance.now();
  if (q.level !== "ok") { rec.okSince = null; return; }
  rec.okSince ??= now;
  // on attend que la position soit stable, puis une mesure toutes les 250 ms au plus
  if (now - rec.okSince < 300 || now < rec.nextAt) return;
  rec.nextAt = now + 250;
  const s = measureLive(a);
  if (!s) return;
  measurements.push(s);
  rec.count++;
  beep();
  const w = $("#videoWrap");
  w.classList.add("flash");
  setTimeout(() => w.classList.remove("flash"), 120);
  updateProgress();
  if (rec.count >= settings.recTarget) finishRecording();
}

function finishRecording() {
  stopRecording();
  beep(660, 0.12);
  beep(990, 0.18, 0.15);
  showResults();
}

function showResults() {
  renderResults(true);
  setView("results");
}
function hideResults() {
  rec.count = 0;
  updateProgress();
  goMeasure();
}
$("#backBtn").addEventListener("click", hideResults);
$("#newBtn").addEventListener("click", () => {
  measurements.length = 0;
  renderResults();
  hideResults();
});

// ---------- carte : estimation, validation, suivi en direct ----------

/** Position attendue de la carte (plaquée sur le front, bord bas ~25 mm au-dessus des pupilles). */
function guessCard(od, og, pxPerMm) {
  const u = norm(sub(og, od));
  const v = { x: -u.y, y: u.x }; // perpendiculaire, vers le bas de l'image
  const center = add(mid(od, og), mul(v, -(25 + CARD_H / 2) * pxPerMm));
  const hw = (CARD_W / 2) * pxPerMm, hh = (CARD_H / 2) * pxPerMm;
  const corner = (sx, sy) => add(center, add(mul(u, sx * hw), mul(v, sy * hh)));
  return [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
}

const cardWidth = (c) => (dist(c[0], c[1]) + dist(c[3], c[2])) / 2;
const cardHeight = (c) => (dist(c[0], c[3]) + dist(c[1], c[2])) / 2;

/**
 * Message d'erreur, ou null si le résultat ressemble bien à une carte ID-1.
 * Les critères visent aussi à rejeter les faux positifs (limite front/cheveux, sourcils) :
 * une vraie carte a des bords parfaitement droits, continus, et une taille cohérente
 * avec l'échelle du visage (expectedW = largeur attendue en px d'après l'iris).
 */
function cardError(r, expectedW = null) {
  if (r.error) return r.error;
  const [a, b, c, d] = r.corners;
  const aspectErr = cardWidth(r.corners) / ((dist(a, d) + dist(b, c)) / 2) / (CARD_W / CARD_H) - 1;
  if (Math.abs(aspectErr) > CARD_ASPECT_TOL) return `proportions incohérentes (${fmt(aspectErr * 100)} %)`;
  // 50 % : un doigt peut couvrir une partie d'un bord
  if (r.coverage < 0.5) return "bords trop peu visibles ou pas rectilignes";
  if (r.maxRms > 0.8) return "bords pas assez rectilignes (cheveux, sourcils ?)";
  if (expectedW) {
    const ratio = cardWidth(r.corners) / expectedW;
    // large tolérance : le diamètre d'iris de MediaPipe est souvent biaisé de 10–20 %
    if (ratio < 0.7 || ratio > 1.4) return `taille incohérente avec le visage (×${fmt(ratio, 2)})`;
  }
  return null;
}

/** Largeur de carte attendue (px) d'après le diamètre d'iris. */
const irisHist = [];
/** Diamètre d'iris lissé (médiane des ~2 dernières secondes) ; les clignements sont ignorés. */
function smoothIris(a) {
  if (!a) return null;
  const d = (a.od.diam + a.og.diam) / 2;
  const med = irisHist.length >= 5 ? median(irisHist) : d;
  if (irisHist.length < 5 || Math.abs(d / med - 1) < 0.1) {
    irisHist.push(d);
    if (irisHist.length > 60) irisHist.shift();
  }
  return irisHist.length >= 5 ? median(irisHist) : d;
}
let irisSmooth = null;
const expectedCardW = (a) => (a && irisSmooth ? (irisSmooth / IRIS_D) * CARD_W : null);

/** Essaie plusieurs estimations initiales [{ corners, frac }] ; renvoie le premier résultat valide. */
function findCard(gray, guesses, expectedW = null) {
  let lastErr = "carte introuvable";
  for (const g of guesses) {
    const r = refineCard(gray, g.corners, Math.max(8, g.frac * cardWidth(g.corners)), { aspectTol: CARD_ASPECT_TOL });
    const err = cardError(r, expectedW);
    if (!err) return r;
    lastErr = err;
  }
  return { error: lastErr };
}

const liveCard = { next: 0, corners: null, guide: null, misses: 0, diag: [], canvas: document.createElement("canvas") };

function liveCardStep(a) {
  liveCard.guide = a ? guessCard(a.od.center, a.og.center, (a.od.diam + a.og.diam) / 2 / IRIS_D) : null;
  const now = performance.now();
  if (now < liveCard.next) return;
  liveCard.next = now + 200;
  // détection sur une image réduite (rapide) : c'est un retour visuel, la capture est traitée en pleine résolution
  const W = video.videoWidth, H = video.videoHeight, s = Math.min(1, 1280 / W);
  const c = liveCard.canvas;
  c.width = Math.round(W * s);
  c.height = Math.round(H * s);
  c.getContext("2d", { willReadFrequently: true }).drawImage(video, 0, 0, c.width, c.height);
  const gray = toGray(c);
  const down = (pts) => pts.map((p) => mul(p, s));
  const guesses = [];
  if (liveCard.corners) guesses.push({ corners: down(liveCard.corners), frac: 0.12 });
  if (liveCard.guide) {
    guesses.push({ corners: down(liveCard.guide), frac: 0.18 });
    guesses.push({ corners: down(liveCard.guide), frac: 0.35 });
  }
  const ew = expectedCardW(a);
  const r = guesses.length ? findCard(gray, guesses, ew && ew * s) : { error: "pas de visage" };
  if (!r.error) {
    liveCard.corners = r.corners.map((p) => mul(p, 1 / s));
    liveCard.misses = 0;
  } else if (++liveCard.misses >= 3) {
    liveCard.corners = null;
  }
  if (a) {
    liveCard.diag.push(r.error || null);
    if (liveCard.diag.length > 25) liveCard.diag.shift();
  }
}

/** Résumé des 5 dernières secondes : taux de réussite et cause de refus la plus fréquente. */
function cardDiagText() {
  const d = liveCard.diag;
  if (d.length < 5) return "";
  const ok = d.filter((e) => !e).length;
  const counts = new Map();
  for (const e of d) if (e) {
    const key = e.replace(/\s*\(.*\)\s*$/, "");
    counts.set(key, { n: (counts.get(key)?.n || 0) + 1, last: e });
  }
  const top = [...counts.values()].sort((x, y) => y.n - x.n)[0];
  return `Diagnostic : carte reconnue sur ${Math.round((100 * ok) / d.length)} % des images` +
    (top ? ` · refus le plus fréquent : ${top.last}` : "");
}

/** Distance caméra → carte estimée en direct (caméra calibrée seulement). */
function liveDistText() {
  if (!camCal?.f || !liveCard.corners || settings.mode !== "card") return "";
  const mm = camCal.f * cardScale(liveCard.corners.map((p) => undistort(camCal, p))).mmPerPx;
  return `≈ ${Math.round(mm / 10)} cm`;
}

function drawQuad(pts, color, dash) {
  octx.strokeStyle = color;
  octx.setLineDash(dash);
  octx.beginPath();
  octx.moveTo(pts[0].x, pts[0].y);
  for (const p of [...pts.slice(1), pts[0]]) octx.lineTo(p.x, p.y);
  octx.stroke();
  octx.setLineDash([]);
}

function drawOverlay(a) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!a) return;
  const lw = Math.max(1.5, overlay.width / 800);
  octx.lineWidth = lw;
  for (const [eye, col] of [[a.od, "#e5484d"], [a.og, "#2b8aeb"]]) {
    octx.strokeStyle = col;
    octx.beginPath();
    octx.arc(eye.center.x, eye.center.y, eye.diam / 2, 0, Math.PI * 2);
    octx.stroke();
    octx.fillStyle = col;
    octx.beginPath();
    octx.arc(eye.center.x, eye.center.y, lw * 1.5, 0, Math.PI * 2);
    octx.fill();
  }
  octx.strokeStyle = "rgba(255,255,255,0.7)";
  octx.setLineDash([6, 6]);
  octx.beginPath();
  octx.moveTo(a.od.center.x, a.od.center.y);
  octx.lineTo(a.og.center.x, a.og.center.y);
  octx.stroke();
  octx.setLineDash([]);
  if (settings.mode === "card") {
    if (liveCard.corners) {
      octx.lineWidth = lw * 2;
      drawQuad(liveCard.corners, "#2fbf71", []);
    } else if (liveCard.guide) {
      drawQuad(liveCard.guide, "rgba(245,165,36,0.9)", [lw * 6, lw * 4]);
    }
  }
}

/** Infos techniques (menu « Autres options ») ; les consignes, elles, sont sur la vidéo. */
function updateLive(a) {
  $("#cardDiag").textContent = settings.mode === "card" ? cardDiagText() : "";
  if (!a || poseIssues(a).length) return;
  liveSamples.push(irisEstimate(a));
  if (liveSamples.length > 90) liveSamples.shift();
  if (liveSamples.length >= 10) $("#liveVal").textContent = `${fmt(median(liveSamples))} mm`;
}

function captureWithTimer() {
  const cd = $("#countdown");
  let n = 3;
  cd.hidden = false;
  cd.textContent = n;
  $("#captureBtn").disabled = true;
  const t = setInterval(() => {
    n--;
    if (n > 0) { cd.textContent = n; return; }
    clearInterval(t);
    cd.hidden = true;
    // rafale : on garde l'image la plus frontale parmi ~12 images
    burst = { remaining: 12, best: null, bestScore: Infinity };
  }, 1000);
}

function burstStep(a) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  const score = a ? Math.abs(a.yaw) + Math.abs(a.pitch) / 3 : 1e6;
  if (score < burst.bestScore) { burst.best = c; burst.bestScore = score; }
  if (--burst.remaining <= 0) {
    const img = burst.best;
    burst = null;
    $("#captureBtn").disabled = false;
    // la position de carte suivie en direct sert de point de départ sur la capture
    openEditor(img, liveCard.corners, camCal);
  }
}

$("#fileInput").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  await modelReady;
  const bmp = await createImageBitmap(file);
  const k = Math.min(1, MAX_IMG / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * k);
  c.height = Math.round(bmp.height * k);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  openEditor(c);
});

// ---------- calibration de la caméra ----------
// Mémorisée par caméra et par résolution (localStorage) : à refaire seulement si l'on change de caméra.
const CAL_KEY = "ipd-calib";
let camCal = null; // calibration de la caméra active, ou null

function camKey() {
  const t = video.srcObject?.getVideoTracks?.()[0];
  if (!t) return null;
  return `${t.label || t.getSettings().deviceId || "caméra"}|${video.videoWidth}x${video.videoHeight}`;
}
function loadCals() {
  try { return JSON.parse(localStorage.getItem(CAL_KEY) || "{}"); } catch { return {}; }
}
function refreshCamCal() {
  const k = camKey();
  camCal = k ? loadCals()[k] || null : null;
  updateCalibInfo();
}
function updateCalibInfo() {
  if (!$("#calibInfo")) return; // ligne retirée de l'étape 1 (calibration expérimentale)
  const n = Object.keys(loadCals()).length;
  $("#calibInfo").innerHTML = n
    ? "✓ Caméra calibrée sur cet appareil : la distance sera mesurée automatiquement."
    : "Facultatif : calibrer la caméra (1 min) corrige la déformation de l'objectif et mesure la distance pour vous.";
  $("#calibStartBtn").textContent = n ? "Recalibrer" : "Calibrer ma caméra";
}

// Calibration libre : on place la carte une fois au centre, puis on la bouge à sa guise.
// L'app la suit et capture d'elle-même chaque vue qui apporte quelque chose de nouveau :
// un coin de l'image encore vide, une rotation (gauche/droite) ou une inclinaison (haut/bas).
const NEEDS = ["center", "TL", "TR", "BR", "BL", "yaw+", "yaw-", "pitch+", "pitch-"];
const calib = { active: false, views: [], done: new Set(), track: null, seen: null, stable: 0, next: 0, lost: 0, frame: null, state: "acquire" };

/** Cadre de départ au centre (taille d'une carte tenue à ~35 cm d'une webcam ordinaire). */
function centerFrame(W, H) {
  const f0 = W / 2 / Math.tan((35 * Math.PI) / 180);
  const Z = (f0 * 85.6) / (0.24 * W);
  return projectCard(f0, W / 2, H / 2, 0, 0, Z);
}

/** Déformation en trapèze : rY ≠ 0 si la carte pivote (gauche/droite), rP ≠ 0 si elle bascule (haut/bas). */
function tiltShape(c) {
  return { rY: dist(c[0], c[3]) / dist(c[1], c[2]) - 1, rP: dist(c[0], c[1]) / dist(c[3], c[2]) - 1 };
}

/** Ce qu'une vue apporte : zone de l'image et inclinaisons. */
function viewKinds(c) {
  const W = video.videoWidth, H = video.videoHeight;
  const m = mul(c.reduce(add, { x: 0, y: 0 }), 0.25);
  const nx = m.x / W - 0.5, ny = m.y / H - 0.5;
  const sh = tiltShape(c);
  const kinds = [];
  const flat = Math.abs(sh.rY) < 0.04 && Math.abs(sh.rP) < 0.04;
  if (flat && Math.abs(nx) < 0.15 && Math.abs(ny) < 0.15) kinds.push("center");
  if (Math.abs(nx) > 0.17 && Math.abs(ny) > 0.1) kinds.push((ny < 0 ? "T" : "B") + (nx < 0 ? "L" : "R"));
  if (Math.abs(sh.rY) >= 0.05) kinds.push(sh.rY > 0 ? "yaw+" : "yaw-");
  if (Math.abs(sh.rP) >= 0.05) kinds.push(sh.rP > 0 ? "pitch+" : "pitch-");
  return kinds;
}

async function startCalibration() {
  await startCamera();
  if (!video.srcObject || !video.videoWidth) return;
  if (rec.active) stopRecording();
  document.querySelectorAll("details.pop[open]").forEach((d) => (d.open = false));
  Object.assign(calib, { active: true, views: [], done: new Set(), track: null, seen: null, stable: 0, next: 0, lost: 0, state: "acquire",
    frame: centerFrame(video.videoWidth, video.videoHeight) });
  try { rec.audio ??= new (window.AudioContext || window.webkitAudioContext)(); rec.audio.resume(); } catch {}
  document.body.classList.add("calibrating");
  $("#recBar").style.width = "0%";
}

function stopCalibration() {
  calib.active = false;
  document.body.classList.remove("calibrating");
  $("#recBar").style.width = "0%";
  octx.clearRect(0, 0, overlay.width, overlay.height);
}

const calibCanvas = document.createElement("canvas");

function calibStep() {
  const W = video.videoWidth, H = video.videoHeight;
  const now = performance.now();
  if (now >= calib.next) {
    calib.next = now + 100;
    // suivi rapide sur une image réduite
    const s = Math.min(1, 960 / W);
    calibCanvas.width = Math.round(W * s);
    calibCanvas.height = Math.round(H * s);
    calibCanvas.getContext("2d", { willReadFrequently: true }).drawImage(video, 0, 0, calibCanvas.width, calibCanvas.height);
    const gray = toGray(calibCanvas);
    const down = (pts) => pts.map((p) => mul(p, s));
    // 1) suivi autour de la position précédente ; 2) recherche large là où la carte a été vue
    // pour la dernière fois (mouvement rapide) ; 3) cadre de départ au centre
    const guesses = [[calib.track, 0.3], [calib.seen, 0.8], [calib.frame, 0.5]].filter(([g]) => g);
    // carte perdue : on la cherche aussi, à tour de rôle, dans les coins encore à faire
    if (!calib.track) {
      const spots = ["TL", "TR", "BR", "BL"].filter((k) => !calib.done.has(k));
      if (spots.length) {
        const k = spots[(calib.scan = (calib.scan || 0) + 1) % spots.length];
        const shape = calib.seen || calib.frame;
        const m = mul(shape.reduce(add, { x: 0, y: 0 }), 0.25);
        const to = { x: W * (k[1] === "L" ? 0.2 : 0.8), y: H * (k[0] === "T" ? 0.24 : 0.76) };
        guesses.push([shape.map((p) => add(p, sub(to, m))), 0.6]);
      }
    }
    let found = null;
    for (const [g, frac] of guesses) {
      const r = refineCard(gray, down(g), Math.max(8, frac * cardWidth(g) * s), { aspectTol: 0.6, maxAngle: 30 });
      // tolérant : les doigts cachent souvent une partie des bords
      if (!r.error && r.coverage >= 0.3 && r.maxRms < 2.5) { found = r.corners.map((p) => mul(p, 1 / s)); break; }
    }
    if (!found) {
      if (++calib.lost > 4) Object.assign(calib, { track: null, stable: 0 });
    } else {
      calib.lost = 0;
      // immobile = déplacement < 2,5 % de la largeur de la carte entre deux analyses
      const still = calib.track && Math.max(...found.map((p, i) => dist(p, calib.track[i]))) < 0.025 * cardWidth(found);
      calib.stable = still ? calib.stable + 1 : 0;
      calib.track = found;
      calib.seen = found;
      const fresh = viewKinds(found).filter((k) => !calib.done.has(k));
      calib.state = fresh.length ? (calib.stable >= 2 ? "capture" : "hold") : "move";
      if (fresh.length && calib.stable >= 2) captureCalibView(found, fresh);
      if (calib.done.size >= NEEDS.length) return finishCalibration();
    }
  }
  drawCalibOverlay();
  const missing = NEEDS.filter((k) => !calib.done.has(k));
  const next = missing[0] || "";
  // perdue depuis peu (mouvement rapide) ou depuis longtemps (> 2 s) : consignes différentes
  const lostLong = !calib.seen || calib.lost >= 20;
  const hint = !calib.track ? (lostLong ? "Placez la carte dans le cadre orange, au centre" : "Bougez la carte un peu plus lentement")
    : calib.state === "hold" ? "Ne bougez plus…"
    : next === "center" ? "Tenez la carte au centre, bien face à la caméra"
    : next.length === 2 ? "Amenez la carte vers un coin orange"
    : next.startsWith("yaw") ? "Faites pivoter la carte vers la gauche ou la droite"
    : "Basculez le haut de la carte vers l'avant ou l'arrière";
  showQuality({ level: !calib.track ? (lostLong ? "bad" : "warn") : calib.state === "move" ? "warn" : "ok", msg: hint });
  $("#hudCount").textContent = `${calib.done.size} / ${NEEDS.length}`;
}

/** Capture en pleine résolution : coins précis + points de bord pour la distorsion. */
function captureCalibView(guess, kinds) {
  const W = video.videoWidth, H = video.videoHeight;
  fullCanvas.width = W;
  fullCanvas.height = H;
  fullCanvas.getContext("2d", { willReadFrequently: true }).drawImage(video, 0, 0);
  const gray = toGray(fullCanvas);
  const r = refineCard(gray, guess, Math.max(10, 0.06 * cardWidth(guess)), { aspectTol: 0.6, maxAngle: 10 });
  if (r.error || r.coverage < 0.3) return;
  calib.views.push({ corners: r.corners, edges: sampleEdges(gray, r.corners, 5) });
  kinds.forEach((k) => calib.done.add(k));
  calib.stable = 0;
  beep();
  const w = $("#videoWrap");
  w.classList.add("flash");
  setTimeout(() => w.classList.remove("flash"), 120);
  $("#recBar").style.width = (100 * calib.done.size) / NEEDS.length + "%";
}

function drawCalibOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  const W = overlay.width, H = overlay.height;
  const lw = Math.max(2, W / 600);
  // repères de coins : orange à faire, vert fait
  const L = 0.07 * W, ix = 0.05 * W, iy = 0.07 * H;
  const corners = { TL: [ix, iy, 1, 1], TR: [W - ix, iy, -1, 1], BR: [W - ix, H - iy, -1, -1], BL: [ix, H - iy, 1, -1] };
  for (const [k, [x, y, sx, sy]] of Object.entries(corners)) {
    octx.strokeStyle = calib.done.has(k) ? "#2fbf71" : "rgba(245,165,36,0.95)";
    octx.lineWidth = lw * 2.5;
    octx.beginPath();
    octx.moveTo(x, y + sy * L);
    octx.lineTo(x, y);
    octx.lineTo(x + sx * L, y);
    octx.stroke();
  }
  octx.lineWidth = lw;
  if (!calib.track && (!calib.seen || calib.lost >= 20)) {
    drawQuad(calib.frame, "rgba(245,165,36,0.95)", [lw * 6, lw * 4]);
  } else if (calib.track) {
    octx.lineWidth = lw * 1.5;
    drawQuad(calib.track, calib.state === "move" ? "rgba(255,255,255,0.85)" : "#2fbf71", []);
  }
}

function finishCalibration() {
  const W = video.videoWidth, H = video.videoHeight;
  const res = calibrate(calib.views, W, H);
  stopCalibration();
  beep(660, 0.12);
  beep(990, 0.18, 0.15);
  const focalOk = res.f && res.hfov > 30 && res.hfov < 130;
  const cal = { cx: res.cx, cy: res.cy, R: res.R, k1: res.k1, f: focalOk ? res.f : null, W, H, hfov: focalOk ? res.hfov : null, date: Date.now() };
  const all = loadCals();
  all[camKey()] = cal;
  try { localStorage.setItem(CAL_KEY, JSON.stringify(all)); } catch {}
  refreshCamCal();
  // déformation relative aux coins de l'image (ρ = 1)
  const edgePct = Math.abs(res.k1) * 100;
  $("#calibResult").innerHTML = `
    <div class="calib-stats">
      <div><span>Champ de vision</span>${focalOk ? fmt(res.hfov, 0) + "°" : "—"}</div>
      <div><span>Déformation de l'objectif</span>${fmt(edgePct, 1)} %</div>
    </div>
    <p>${focalOk
      ? "La distance à la caméra sera désormais <b>mesurée automatiquement</b> : plus besoin de mètre ruban."
      : "La déformation est corrigée, mais la focale n'a pas pu être déterminée : la distance saisie reste utilisée. Recommencez en inclinant davantage la carte."}</p>
    <p class="muted small">Calibration mémorisée pour cette caméra (${W}×${H}).</p>`;
  $("#popCalib").showModal();
}

$("#calibBtn").addEventListener("click", startCalibration);
$("#calibCancelBtn").addEventListener("click", stopCalibration);
$("#calibStartBtn")?.addEventListener("click", () => {
  setView("measure");
  startCalibration();
});

// ---------- éditeur ----------
const editCanvas = $("#editCanvas");
const ectx = editCanvas.getContext("2d");
const loupe = $("#loupe");
const lctx = loupe.getContext("2d");
let editor = null;

const LABELS = {
  od: "pupille OD (votre œil droit)",
  og: "pupille OG (votre œil gauche)",
  bridge: "milieu du nez",
  c0: "carte – coin haut gauche",
  c1: "carte – coin haut droit",
  c2: "carte – coin bas droit",
  c3: "carte – coin bas gauche",
};

function openEditor(img, cardHint = null, cal = null) {
  const W = img.width, H = img.height;
  const a = analyze(landmarker.detectForVideo(img, nextTs()), W, H);
  let od, og, bridge, irisDiam = null, pose = null;
  if (a) {
    od = a.od.center; og = a.og.center; bridge = a.bridge;
    irisDiam = (a.od.diam + a.og.diam) / 2;
    pose = a;
  } else {
    od = { x: W * 0.42, y: H * 0.5 }; og = { x: W * 0.58, y: H * 0.5 }; bridge = mid(od, og);
    setStatus("Visage non trouvé : placez les repères");
  }
  const faceGuess = guessCard(od, og, irisDiam ? irisDiam / IRIS_D : dist(od, og) / 63);
  const start = cardHint || faceGuess;

  editor = {
    img, W, H, irisDiam, pose, faceGuess, cardHint, cal,
    markers: { od, og, bridge, c0: start[0], c1: start[1], c2: start[2], c3: start[3] },
    selected: "od",
    drag: null,
    gray: null,
    edgePts: null,
  };
  editCanvas.width = W;
  editCanvas.height = H;
  setView("edit");
  $("#cardStatus").textContent = "";
  if (settings.mode === "card") {
    const g = [];
    if (cardHint) g.push({ corners: cardHint, frac: 0.08 });
    g.push({ corners: faceGuess, frac: 0.15 }, { corners: faceGuess, frac: 0.35 });
    detectCard(g);
  }
  renderEditor();
  updateEditResult();
}

const CORNERS = ["c0", "c1", "c2", "c3"];

/** Détection automatique des bords à partir d'une liste d'estimations initiales. */
function detectCard(guesses) {
  const st = $("#cardStatus");
  const m = editor.markers;
  if (!editor.gray) editor.gray = toGray(editor.img);
  const r = findCard(editor.gray, guesses, editor.irisDiam ? (editor.irisDiam / IRIS_D) * CARD_W : null);
  if (r.error) {
    st.className = "note warn";
    st.textContent = `Détection impossible : ${r.error}. Faites glisser les 4 coins orange à peu près sur ceux de la carte, puis cliquez « Détecter les bords ». Vérifiez aussi que la carte est entière dans l'image, à plat, et qu'elle se distingue bien de la peau ou des cheveux.`;
    return false;
  }
  CORNERS.forEach((k, i) => (m[k] = r.corners[i]));
  editor.edgePts = r.points;
  st.className = "note ok";
  st.textContent = `Bords détectés : ${r.points.length} points, résidu ${fmt(r.rms, 2)} px.`;
  return true;
}

$("#detectBtn").addEventListener("click", () => {
  if (!editor) return;
  if (settings.mode !== "card") { $("#cardStatus").textContent = "Disponible en mode carte."; return; }
  const current = CORNERS.map((k) => editor.markers[k]);
  detectCard([{ corners: current, frac: 0.1 }, { corners: current, frac: 0.25 }]);
  renderEditor();
  updateEditResult();
});

/** Toute retouche manuelle d'un coin invalide les points de bord affichés. */
function markerMoved(key) {
  if (CORNERS.includes(key) && editor.edgePts) {
    editor.edgePts = null;
    $("#cardStatus").className = "note";
    $("#cardStatus").textContent = "Coins modifiés à la main.";
  }
}

function visibleKeys() {
  const keys = ["od", "og", "bridge"];
  if (settings.mode === "card") keys.push("c0", "c1", "c2", "c3");
  return keys;
}

/** Dessine la scène ; `px` = taille d'un pixel écran en unités image. */
function drawScene(ctx, px) {
  const { img, markers: m, selected } = editor;
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = px * 1.5;

  if (settings.mode === "card") {
    ctx.strokeStyle = "#f5a524";
    ctx.beginPath();
    ctx.moveTo(m.c0.x, m.c0.y);
    for (const k of ["c1", "c2", "c3", "c0"]) ctx.lineTo(m[k].x, m[k].y);
    ctx.stroke();
    if (editor.edgePts) {
      ctx.fillStyle = "#22d3ee";
      for (const q of editor.edgePts) ctx.fillRect(q.x - px, q.y - px, px * 2, px * 2);
    }
  } else if (editor.irisDiam) {
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    for (const k of ["od", "og"]) {
      ctx.beginPath();
      ctx.arc(m[k].x, m[k].y, editor.irisDiam / 2, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.setLineDash([px * 6, px * 6]);
  ctx.beginPath();
  ctx.moveTo(m.od.x, m.od.y);
  ctx.lineTo(m.og.x, m.og.y);
  ctx.stroke();
  ctx.setLineDash([]);

  for (const k of visibleKeys()) {
    const p = m[k];
    const col = k === "od" ? "#e5484d" : k === "og" ? "#2b8aeb" : k === "bridge" ? "#2fbf71" : "#f5a524";
    const r = px * (k === selected ? 9 : 7);
    ctx.strokeStyle = col;
    ctx.lineWidth = px * (k === selected ? 2 : 1.5);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.moveTo(p.x - r * 1.6, p.y); ctx.lineTo(p.x - r * 0.4, p.y);
    ctx.moveTo(p.x + r * 0.4, p.y); ctx.lineTo(p.x + r * 1.6, p.y);
    ctx.moveTo(p.x, p.y - r * 1.6); ctx.lineTo(p.x, p.y - r * 0.4);
    ctx.moveTo(p.x, p.y + r * 0.4); ctx.lineTo(p.x, p.y + r * 1.6);
    ctx.stroke();
  }
}

function pxScale() {
  const w = editCanvas.getBoundingClientRect().width || editor.W;
  return editor.W / w;
}

function renderEditor() {
  if (!editor) return;
  ectx.setTransform(1, 0, 0, 1, 0, 0);
  drawScene(ectx, pxScale());
  renderLoupe();
  $("#selName").textContent = editor.selected ? LABELS[editor.selected] : "aucun";
}

function renderLoupe() {
  const p = editor.markers[editor.selected] || mid(editor.markers.od, editor.markers.og);
  const Z = 8;
  const S = loupe.width;
  lctx.setTransform(1, 0, 0, 1, 0, 0);
  lctx.fillStyle = "#000";
  lctx.fillRect(0, 0, S, S);
  lctx.imageSmoothingEnabled = false;
  lctx.setTransform(Z, 0, 0, Z, S / 2 - Z * p.x, S / 2 - Z * p.y);
  drawScene(lctx, 1 / Z);
}

function toImg(e) {
  const r = editCanvas.getBoundingClientRect();
  return { x: ((e.clientX - r.left) * editor.W) / r.width, y: ((e.clientY - r.top) * editor.H) / r.height };
}

editCanvas.addEventListener("pointerdown", (e) => {
  if (!editor) return;
  const p = toImg(e);
  const reach = 30 * pxScale();
  let best = null, bd = Infinity;
  for (const k of visibleKeys()) {
    const d = dist(p, editor.markers[k]);
    if (d < bd) { bd = d; best = k; }
  }
  if (best && bd <= reach) {
    editor.selected = best;
    editor.drag = { key: best, off: sub(editor.markers[best], p) };
    editCanvas.setPointerCapture(e.pointerId);
  }
  renderEditor();
});
editCanvas.addEventListener("pointermove", (e) => {
  if (!editor?.drag) return;
  const { key, off } = editor.drag;
  const p = add(toImg(e), off);
  editor.markers[key] = { x: clamp(p.x, 0, editor.W), y: clamp(p.y, 0, editor.H) };
  markerMoved(key);
  renderEditor();
  updateEditResult();
});
const endDrag = () => { if (editor) editor.drag = null; };
editCanvas.addEventListener("pointerup", endDrag);
editCanvas.addEventListener("pointercancel", endDrag);

function nudge(dx, dy, step) {
  if (!editor?.selected) return;
  const p = editor.markers[editor.selected];
  editor.markers[editor.selected] = { x: p.x + dx * step, y: p.y + dy * step };
  markerMoved(editor.selected);
  renderEditor();
  updateEditResult();
}
document.querySelectorAll(".nudge button").forEach((b) =>
  b.addEventListener("click", () => nudge(+b.dataset.dx, +b.dataset.dy, parseFloat($("#step").value)))
);
document.addEventListener("keydown", (e) => {
  if (!editor || document.body.dataset.view !== "edit") return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const dirs = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
  if (e.key === "Tab" && !e.shiftKey) {
    const keys = visibleKeys();
    editor.selected = keys[(keys.indexOf(editor.selected) + 1) % keys.length];
    renderEditor();
    e.preventDefault();
    return;
  }
  if (!dirs[e.key]) return;
  e.preventDefault();
  nudge(...dirs[e.key], e.shiftKey ? 0.1 : 1);
});
window.addEventListener("resize", renderEditor);

/** Calcul pour la capture en cours. */
function computeEditor() {
  const { markers: m0, irisDiam, pose, cal } = editor;
  // photo prise avec une caméra calibrée : repères corrigés de la distorsion
  const m = Object.fromEntries(Object.entries(m0).map(([k, p]) => [k, undistort(cal, p)]));
  const mode = settings.mode;
  const warnings = [];
  let mmPerPx;
  if (mode === "card") {
    const corners = [m.c0, m.c1, m.c2, m.c3];
    const sc = cardScale(corners);
    mmPerPx = sc.mmPerPx;
    if (sc.tilt > MAX_CARD_TILT) warnings.push("La carte semble pivotée ou mal placée : vérifiez les coins.");
    const top = dist(m.c0, m.c1), bottom = dist(m.c3, m.c2);
    if (Math.abs(top - bottom) / cardWidth(corners) > 0.02) warnings.push("La carte ne semble pas bien à plat face à la caméra.");
  } else {
    if (!irisDiam) return { error: "Le mode iris nécessite une détection automatique du visage." };
    mmPerPx = IRIS_D / irisDiam;
  }
  if (pose) {
    for (const i of poseIssues(pose)) warnings.push("Position : " + i + ".");
  }
  const yawCos = pose ? Math.cos((pose.yaw * Math.PI) / 180) : 1;
  const raw = (dist(m.od, m.og) * mmPerPx) / yawCos;
  const u = sub(m.og, m.od);
  // part du nez mesurée sur les repères, corrigée de la rotation de la tête (estimée en 3D)
  const t = clamp(dot(sub(m.bridge, m.od), u) / dot(u, u) + (pose ? pose.tCorr : 0), 0, 1);
  const { D, Dauto } = measDistance(mode === "card" && cal?.f ? cal.f * mmPerPx : null);
  return { mode, raw, t, yaw: pose ? pose.yaw : null, warnings, D, Dauto };
}

/** Distance à enregistrer avec une mesure : mesurée si disponible, sinon celle saisie à cet instant. */
function measDistance(measured) {
  return measured ? { D: measured, Dauto: true } : { D: settings.distanceCm * 10, Dauto: false };
}

function derive(meas) {
  const far = meas.raw * farFactor(meas.mode, meas.D);
  return { far, odMono: far * meas.t, ogMono: far * (1 - meas.t), near: nearFromFar(far) };
}

function updateEditResult() {
  const el = $("#editResult");
  const r = computeEditor();
  if (r.error) { el.innerHTML = `<p class="warn">${r.error}</p>`; $("#addBtn").disabled = true; return; }
  $("#addBtn").disabled = false;
  const d = derive(r);
  el.innerHTML = `
    <div class="big">${fmt(d.far, 1)} mm</div>
    <div>Œil droit ${fmt(d.odMono)} · œil gauche ${fmt(d.ogMono)} mm</div>
    ${r.warnings.map((w) => `<p class="warn">${w}</p>`).join("")}
    <div class="muted small">Mesure brute ${fmt(r.raw, 2)} mm × correction ${fmt(farFactor(r.mode, r.D), 4)}${r.Dauto ? ` · distance mesurée ${fmt(r.D / 10, 0)} cm` : ""}</div>`;
}

$("#addBtn").addEventListener("click", () => {
  const r = computeEditor();
  if (r.error) return;
  measurements.push({ ...r, source: "manuel" });
  editor = null;
  showResults();
});
$("#cancelBtn").addEventListener("click", () => { closeEditor(); goMeasure(); });
function closeEditor() {
  editor = null;
}

// ---------- résultats ----------
const measurements = [];
const T95 = [0, 12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.20, 2.18, 2.16, 2.14, 2.13];
const tCrit = (df) => (df < T95.length ? T95[df] : 1.96);

function stats(arr) {
  const n = arr.length;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
  return { n, mean, sd, ci: n > 1 ? (tCrit(n - 1) * sd) / Math.sqrt(n) : NaN, med: median(arr) };
}

let summaryText = "";

/** Rejet des valeurs aberrantes : au-delà de 3 écarts absolus médians normalisés de la médiane. */
function inlierMask(values) {
  if (values.length < 5) return values.map(() => true);
  const med = median(values);
  const mad = 1.4826 * median(values.map((v) => Math.abs(v - med)));
  const lim = Math.max(3 * mad, 0.3);
  return values.map((v) => Math.abs(v - med) <= lim);
}

/** Pas « rond » pour les graduations (0,1 / 0,2 / 0,5 / 1 / 2 …). */
function niceStep(range, target) {
  const raw = range / target, p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * p).find((st) => st >= raw) || 10 * p;
}

/**
 * Graphique des mesures dans l'ordre de capture : un point par mesure, la moyenne en pointillé,
 * ± 1 écart type en bande pâle. Les valeurs écartées sont en gris ; celles hors cadre sont
 * plaquées au bord avec un triangle. animate = apparition progressive (< 1,5 s).
 */
function runChart(values, mask, mean, sd, animate) {
  const n = values.length;
  const Wv = 640, Hv = 220, L = 46, R = 14, T = 14, B = 28;
  const kept = values.filter((_, i) => mask[i]);
  const spread = Math.max(sd * 3, 0.6);
  let lo = Math.min(...kept, mean - spread), hi = Math.max(...kept, mean + spread);
  const step = niceStep(hi - lo, 5);
  lo = Math.floor(lo / step) * step;
  hi = Math.ceil(hi / step) * step;
  const X = (i) => L + (n === 1 ? (Wv - L - R) / 2 : (i * (Wv - L - R)) / (n - 1));
  const Y = (v) => T + ((hi - v) / (hi - lo)) * (Hv - T - B);
  const dec = step < 1 ? 1 : 0;
  // durée totale ≤ 1,5 s : points en ~1 s, puis la moyenne
  const per = Math.min(40, 1000 / n);
  let svg = `<svg viewBox="0 0 ${Wv} ${Hv}" class="${animate ? "anim" : ""}" role="img" aria-label="Mesures successives et leur moyenne">`;
  for (let v = lo; v <= hi + 1e-9; v += step) {
    svg += `<line class="grid" x1="${L}" x2="${Wv - R}" y1="${Y(v)}" y2="${Y(v)}"/>`;
    svg += `<text class="axis" x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${fmt(v, dec)}</text>`;
  }
  const xStep = Math.max(1, niceStep(n, 6));
  for (let k = 1; k <= n; k++) {
    if (k === 1 || k === n || k % xStep === 0) {
      svg += `<text class="axis" x="${X(k - 1)}" y="${Hv - 8}" text-anchor="middle">${k}</text>`;
    }
  }
  if (sd > 0) {
    const y1 = Y(Math.min(hi, mean + sd)), y2 = Y(Math.max(lo, mean - sd));
    svg += `<rect class="band" x="${L}" y="${y1}" width="${Wv - L - R}" height="${y2 - y1}" style="animation-delay:${n * per}ms"/>`;
  }
  values.forEach((v, i) => {
    const out = v > hi || v < lo;
    const y = Y(Math.min(hi, Math.max(lo, v)));
    const cls = mask[i] ? "pt" : "pt off";
    const d = `style="animation-delay:${Math.round(i * per)}ms"`;
    svg += out
      ? `<path class="${cls}" d="M${X(i) - 5} ${y + (v > hi ? 5 : -5)} L${X(i)} ${y} L${X(i) + 5} ${y + (v > hi ? 5 : -5)}Z" ${d}/>`
      : `<circle class="${cls}" cx="${X(i)}" cy="${y}" r="${n > 60 ? 3 : 4}" ${d}/>`;
  });
  const ym = Y(mean);
  svg += `<line class="mean" x1="${L}" x2="${Wv - R}" y1="${ym}" y2="${ym}" style="animation-delay:${n * per}ms"/>`;
  svg += `<text class="mean-lab" x="${Wv - R - 4}" y="${ym - 6}" text-anchor="end" style="animation-delay:${n * per + 200}ms">moyenne ${fmt(mean, 1)} mm</text>`;
  return svg + "</svg>";
}

function renderResults(animate = false) {
  const tbody = $("#table tbody");
  tbody.innerHTML = "";
  const has = measurements.length > 0;
  $("#stepResults").disabled = !has;
  $("#detail").hidden = !has;
  $("#copyBtn").disabled = !has;
  $("#clearBtn").disabled = !has;
  $("#hist").innerHTML = "";
  $("#chart").hidden = true;
  if (!has) {
    $("#summary").innerHTML = `<p class="muted">Aucune mesure pour l'instant.</p>`;
    $("#techStats").innerHTML = "";
    summaryText = "";
    seriesId = null; // la prochaine mesure ouvrira un nouvel essai
    renderHistory();
    return;
  }
  const ds = measurements.map(derive);
  const fars = ds.map((d) => d.far);
  const mask = inlierMask(fars);
  const kept = ds.filter((_, i) => mask[i]);
  const nOut = ds.length - kept.length;

  measurements.forEach((m, i) => {
    const d = ds[i];
    const tr = document.createElement("tr");
    if (!mask[i]) tr.className = "out";
    tr.innerHTML = `<td>${i + 1}</td><td>${m.source || "manuel"} · ${m.mode === "card" ? "carte" : "iris"}</td>
      <td>${fmt(m.raw, 2)}</td><td><strong>${fmt(d.far, 2)}</strong></td>
      <td>${fmt(d.odMono)}</td><td>${fmt(d.ogMono)}</td>
      <td>${m.yaw == null ? "—" : fmt(m.yaw) + "°"}</td>
      <td><button aria-label="Supprimer">✕</button></td>`;
    tr.querySelector("button").addEventListener("click", () => { measurements.splice(i, 1); renderResults(); });
    tbody.appendChild(tr);
  });

  const far = stats(kept.map((d) => d.far));
  const odS = stats(kept.map((d) => d.odMono));
  const ogS = stats(kept.map((d) => d.ogMono));
  const near = nearFromFar(far.mean);
  // l'essai en cours est enregistré (ou mis à jour) dans l'historique de session
  seriesId ??= Date.now();
  // distance de l'essai = celle de ses mesures (anciennes mesures sans distance : réglage actuel)
  const Ds = measurements.map((m) => m.D || settings.distanceCm * 10);
  const Dmean = Ds.reduce((acc, v) => acc + v, 0) / Ds.length;
  const Dauto = measurements.some((m) => m.Dauto);
  upsertHistory({ id: seriesId, mean: far.mean, sd: far.sd, n: far.n, od: odS.mean, og: ogS.mean, dist: Math.round(Dmean / 10), auto: Dauto, mode: measurements[0].mode });
  renderHistory();
  // Résultat principal : avec plusieurs essais, la moyenne des essais (plus fiable qu'un seul,
  // car chaque essai a son petit biais : position de la carte, de la tête…)
  const trials = loadHistory().filter((e) => e.mode === settings.mode);
  let q, main, od, og, subTxt = "";
  if (trials.length >= 2) {
    const ts = trialSummary(trials);
    main = ts.mean; od = ts.od; og = ts.og; q = ts.q;
    subTxt = `<p class="muted small">Moyenne de ${ts.n} essais · ${ts.detail} · cet essai : ${fmt(far.mean, 1)} mm</p>`;
    summaryText = `Écart pupillaire : ${fmt(main, 1)} mm (moyenne de ${ts.n} essais, ± ${fmt(ts.sd, 2)} mm)`;
  } else {
    main = far.mean; od = odS.mean; og = ogS.mean;
    if (far.n < 5) q = ["fair", "Peu de mesures : refaites une série pour plus de sûreté"];
    else if (far.sd <= 0.5) q = ["good", "✓ Mesure régulière"];
    else if (far.sd <= 1) q = ["fair", "Mesure correcte : une nouvelle série peut l'affiner"];
    else q = ["poor", "Mesures dispersées : recommencez en restant immobile"];
    subTxt = `<p class="muted small">Faites 2 ou 3 essais (en reprenant la carte) : le résultat deviendra leur moyenne, plus fiable.</p>`;
    summaryText = `Écart pupillaire : ${fmt(main, 1)} mm`;
  }
  summaryText += `\nŒil droit : ${fmt(od)} mm · œil gauche : ${fmt(og)} mm`;
  $("#summary").innerHTML = `
    <div class="big">${fmt(main, 1)} <small>mm</small></div>
    <div class="quality ${q[0]}">${q[1]}</div>
    <div class="eyes">
      <div>Œil droit<b>${fmt(od)} mm</b></div>
      <div>Œil gauche<b>${fmt(og)} mm</b></div>
    </div>
    <p class="muted">C'est la valeur « EP » ou « PD » à indiquer pour des lunettes de vue.</p>
    ${subTxt}`;
  $("#techStats").innerHTML = `
    <div class="stats">
      <div><span>Mesures retenues</span>${far.n}${nOut ? ` (${nOut} écartée${nOut > 1 ? "s" : ""})` : ""}</div>
      <div><span>IC 95 % de cet essai</span>${far.n > 1 ? "± " + fmt(far.ci, 2) + " mm" : "—"}</div>
      <div><span>Écart-type</span>${far.n > 1 ? fmt(far.sd, 2) + " mm" : "—"}</div>
      <div><span>Médiane</span>${fmt(far.med, 2)} mm</div>
      <div><span>Distance caméra</span>${fmt(Dmean / 10, 0)} cm (${Dauto ? "mesurée" : "saisie"})</div>
      <div><span>EP de près, lecture à ${NEAR_CM} cm</span>${fmt(near, 2)} mm</div>
    </div>
    <p class="muted small">Valeurs aberrantes écartées au-delà de 3 écarts absolus médians. L'intervalle de confiance de l'essai ne couvre que la variation entre ses mesures ; chaque essai a en plus un petit biais propre (position de la carte et de la tête), d'où l'intérêt de faire la moyenne de plusieurs essais.</p>`;
  $("#chart").hidden = fars.length < 2;
  if (fars.length >= 2) $("#hist").innerHTML = runChart(fars, mask, far.mean, far.sd, animate);
}

$("#copyBtn").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(summaryText); setStatus("Résultat copié"); }
  catch { setStatus("Copie impossible"); }
});
$("#clearBtn").addEventListener("click", () => {
  if (confirm("Effacer toutes les mesures ?")) { measurements.length = 0; renderResults(); }
});

function setStatus(t) { $("#status").textContent = t; }

// ---------- historique des essais (session du navigateur) ----------
const HIST_KEY = "ipd-history";
let seriesId = null;
function loadHistory() {
  try { return JSON.parse(sessionStorage.getItem(HIST_KEY) || "[]"); } catch { return []; }
}
function saveHistory(list) {
  try { sessionStorage.setItem(HIST_KEY, JSON.stringify(list.slice(-30))); } catch {}
}
function upsertHistory(entry) {
  const list = loadHistory();
  const i = list.findIndex((e) => e.id === entry.id);
  if (i >= 0) list[i] = entry; else list.push(entry);
  saveHistory(list);
}

/**
 * Synthèse des essais : un essai très éloigné des autres (au-delà de 1 mm de la médiane, ou de
 * 3 écarts absolus médians si la dispersion est plus grande) est écarté, comme pour les mesures.
 * Le verdict repose sur l'écart type des essais retenus, moins sensible qu'un écart max − min.
 */
function trialSummary(list) {
  const ms = list.map((e) => e.mean);
  let keep = ms.map(() => true);
  if (ms.length >= 4) {
    const med = median(ms);
    const lim = Math.max(1, 3 * 1.4826 * median(ms.map((v) => Math.abs(v - med))));
    keep = ms.map((v) => Math.abs(v - med) <= lim);
  }
  const kept = list.filter((_, i) => keep[i]);
  const avg = (f) => kept.reduce((acc, e) => acc + f(e), 0) / kept.length;
  const mean = avg((e) => e.mean);
  const sd = kept.length > 1 ? Math.sqrt(kept.reduce((acc, e) => acc + (e.mean - mean) ** 2, 0) / (kept.length - 1)) : 0;
  const n = kept.length, nOut = list.length - n;
  const q = sd <= 0.4 ? ["good", `✓ ${n} essais concordants`]
    : sd <= 0.8 ? ["fair", `${n} essais assez concordants : un essai de plus peut affiner`]
    : ["poor", "Essais variables : vérifiez la distance saisie et la position de la carte"];
  const detail = `dispersion entre essais : ± ${fmt(sd, 2)} mm` + (nOut ? ` · ${nOut} essai${nOut > 1 ? "s" : ""} écarté${nOut > 1 ? "s" : ""}` : "");
  return { keep, mean, sd, n, nOut, od: avg((e) => e.od), og: avg((e) => e.og), q, detail };
}

function renderHistory() {
  const list = loadHistory();
  $("#history").hidden = list.length === 0;
  const body = $("#histBody");
  body.innerHTML = "";
  const ts = list.length >= 2 ? trialSummary(list) : null;
  list.forEach((e, i) => {
    const tr = document.createElement("tr");
    if (e.id === seriesId) tr.className = "current";
    if (ts && !ts.keep[i]) tr.classList.add("out");
    const time = new Date(e.id).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
    tr.innerHTML = `<td>${i + 1}${e.id === seriesId ? ' <span class="tag">actuel</span>' : ""}${ts && !ts.keep[i] ? ' <span class="tag off">écarté</span>' : ""}</td><td>${time}</td>
      <td><b>${fmt(e.mean, 1)} mm</b>${e.n > 1 ? ` <span class="muted">± ${fmt(e.sd, 2)}</span>` : ""}</td>
      <td>${fmt(e.od)} / ${fmt(e.og)}</td><td>${e.dist} cm${e.auto ? " · auto" : ""}${e.mode === "iris" ? " · sans carte" : ""}</td><td>${e.n}</td>`;
    body.appendChild(tr);
  });
  // cohérence entre essais
  const sum = $("#histSummary");
  if (list.length < 2) {
    sum.innerHTML = `<p class="muted small">Refaites une ou deux séries (en reprenant la carte) pour vérifier que les résultats concordent.</p>`;
    return;
  }
  sum.innerHTML = `<div class="history-sum">
      <span class="quality ${ts.q[0]}">${ts.q[1]}</span>
      <span>Moyenne : <b>${fmt(ts.mean, 1)} mm</b></span>
      <span class="muted">${ts.detail}</span>
    </div>`;
}

$("#histReset").addEventListener("click", () => {
  if (!confirm("Effacer l'historique de vos essais ?")) return;
  saveHistory([]);
  seriesId = measurements.length ? null : seriesId;
  renderHistory();
});
renderHistory();
updateCalibInfo();

// ---------- aide et menus ----------
document.querySelectorAll("[data-pop]").forEach((b) =>
  b.addEventListener("click", () => document.getElementById(b.dataset.pop).showModal())
);
// un clic ailleurs ferme les petits menus ; choisir une option aussi
document.addEventListener("click", (e) => {
  document.querySelectorAll("details.pop[open]").forEach((d) => {
    if (!d.contains(e.target) || e.target.closest(".menu-body")) d.open = false;
  });
});

// ---------- démarrage ----------
const modelReady = initLandmarker()
  .then(() => setStatus(video.srcObject ? "Caméra active" : "Prêt"))
  .catch((e) => setStatus("Échec du chargement du modèle : " + e.message));

// accès de test : http://localhost:8123/?debug
if (new URLSearchParams(location.search).has("debug")) {
  window.__ipd = { measurements, showResults, renderResults, rec, findCard, cardError, calib, getCamCal: () => camCal, refreshCamCal };
}

// aide (?) : un toucher l'ouvre ou la ferme (utile sur mobile), sans activer le champ associé
document.addEventListener("click", (e) => {
  const h = e.target.closest(".help");
  document.querySelectorAll(".help.open").forEach((o) => o !== h && o.classList.remove("open"));
  if (h) { e.preventDefault(); h.classList.toggle("open"); }
});

// garde l'info-bulle dans l'écran : décalage horizontal calculé à l'ouverture
function placeTip(h) {
  const r = h.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const w = Math.min(270, vw * 0.8);
  let left = -10;
  if (r.left + left + w > vw - 8) left = vw - 8 - w - r.left;
  if (r.left + left < 8) left = 8 - r.left;
  h.style.setProperty("--tip-left", left + "px");
}
document.querySelectorAll(".help").forEach((h) => {
  ["mouseenter", "focus", "click"].forEach((ev) => h.addEventListener(ev, () => placeTip(h)));
});
