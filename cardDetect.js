// Détection sous-pixel des bords d'une carte rectangulaire à partir d'une estimation grossière de ses coins.
//
// Pour chacun des 4 bords : on échantillonne des profils d'intensité perpendiculaires au bord,
// on y repère les pics de gradient (précision sous-pixel par interpolation parabolique),
// puis on ajuste une droite robuste (RANSAC + moindres carrés totaux pondérés).
// Les coins sont les intersections des droites : les coins arrondis de la carte
// (rayon ≈ 3 mm) n'interviennent donc pas, et l'on retrouve les dimensions nominales 85,60 × 53,98 mm.

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
const mul = (a, k) => ({ x: a.x * k, y: a.y * k });
const len = (a) => Math.hypot(a.x, a.y);
const norm = (a) => mul(a, 1 / len(a));
const cross = (a, b) => a.x * b.y - a.y * b.x;
const dot = (a, b) => a.x * b.x + a.y * b.y;

/** Image en niveaux de gris (Float32Array). */
export function toGray(canvas) {
  const W = canvas.width, H = canvas.height;
  const d = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, W, H).data;
  const g = new Float32Array(W * H);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * d[j] + 0.587 * d[j + 1] + 0.114 * d[j + 2];
  return { g, W, H };
}

function bilinear({ g, W, H }, x, y) {
  if (x < 0 || y < 0 || x > W - 2 || y > H - 2) return NaN;
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * W + x0;
  return (g[i] * (1 - fx) + g[i + 1] * fx) * (1 - fy) + (g[i + W] * (1 - fx) + g[i + W + 1] * fx) * fy;
}

/** Candidats de bord le long du segment a→b, cherchés à ±radius selon la normale n. */
function edgeCandidates(img, a, b, n, radius, nSamples, lo = 0.15) {
  const out = [];
  const ab = sub(b, a);
  const R = Math.ceil(radius) + 2;
  // demi-largeur de bande le long du bord pour moyenner le bruit
  const t = norm(ab);
  for (let s = 0; s < nSamples; s++) {
    const f = lo + ((1 - 2 * lo) * s) / (nSamples - 1);
    const p = add(a, mul(ab, f));
    const v = [];
    for (let k = -R; k <= R; k++) {
      let acc = 0, cnt = 0;
      for (const o of [-1, 0, 1]) {
        const val = bilinear(img, p.x + n.x * k + t.x * o, p.y + n.y * k + t.y * o);
        if (!Number.isNaN(val)) { acc += val; cnt++; }
      }
      v.push(cnt ? acc / cnt : NaN);
    }
    if (v.some(Number.isNaN)) continue;
    const sm = v.map((_, i) => (i === 0 || i === v.length - 1 ? v[i] : (v[i - 1] + 2 * v[i] + v[i + 1]) / 4));
    const d = sm.map((_, i) => (i === 0 || i === sm.length - 1 ? 0 : (sm[i + 1] - sm[i - 1]) / 2));
    const peaks = [];
    for (let i = 2; i < d.length - 2; i++) {
      const c = Math.abs(d[i]);
      if (c < 3 || c < Math.abs(d[i - 1]) || c < Math.abs(d[i + 1])) continue;
      const l = Math.abs(d[i - 1]), r = Math.abs(d[i + 1]);
      const den = l - 2 * c + r;
      const off = den < 0 ? (0.5 * (l - r)) / den : 0;
      peaks.push({ k: i - R + off, w: c, sign: Math.sign(d[i]) });
    }
    peaks.sort((x, y) => y.w - x.w);
    for (const pk of peaks.slice(0, 3)) {
      if (Math.abs(pk.k) > radius) continue;
      out.push({ ...add(p, mul(n, pk.k)), w: pk.w, sign: pk.sign, s });
    }
  }
  return out;
}

/** Droite par moindres carrés totaux pondérés. */
function tls(pts) {
  let sw = 0, mx = 0, my = 0;
  for (const p of pts) { sw += p.w; mx += p.w * p.x; my += p.w * p.y; }
  mx /= sw; my /= sw;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) {
    const dx = p.x - mx, dy = p.y - my;
    sxx += p.w * dx * dx; sxy += p.w * dx * dy; syy += p.w * dy * dy;
  }
  const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  return { p: { x: mx, y: my }, d: { x: Math.cos(ang), y: Math.sin(ang) } };
}
const lineDist = (L, q) => Math.abs(cross(L.d, sub(q, L.p)));

/** Ajustement robuste d'une droite aux candidats ; dir = direction attendue du bord. */
function fitEdge(cands, dir, tol, maxAngleDeg) {
  if (cands.length < 6) return null;
  const sinMax = Math.sin((maxAngleDeg * Math.PI) / 180);
  let best = null, bestScore = 0;
  const iters = Math.min(400, cands.length * cands.length);
  for (let it = 0; it < iters; it++) {
    const a = cands[(Math.random() * cands.length) | 0];
    const b = cands[(Math.random() * cands.length) | 0];
    if (a.s === b.s || a.sign !== b.sign) continue;
    const dd = sub(b, a);
    if (len(dd) < 1e-6) continue;
    const L = { p: a, d: norm(dd) };
    if (Math.abs(cross(L.d, dir)) > sinMax) continue;
    // un seul point (le plus fort) par profil
    const perS = new Map();
    for (const c of cands) {
      if (c.sign !== a.sign || lineDist(L, c) > tol) continue;
      const prev = perS.get(c.s);
      if (!prev || c.w > prev.w) perS.set(c.s, c);
    }
    let score = 0;
    for (const c of perS.values()) score += c.w;
    if (score > bestScore) { bestScore = score; best = [...perS.values()]; }
  }
  if (!best || best.length < 6) return null;
  let L = tls(best);
  // deux passes de réajustement en rejetant les points éloignés
  let inl = best;
  for (let k = 0; k < 2; k++) {
    const sign = best[0].sign;
    const perS = new Map();
    for (const c of cands) {
      if (c.sign !== sign || lineDist(L, c) > tol) continue;
      const prev = perS.get(c.s);
      if (!prev || lineDist(L, c) < lineDist(L, prev)) perS.set(c.s, c);
    }
    inl = [...perS.values()];
    if (inl.length < 6) return null;
    L = tls(inl);
  }
  const rms = Math.sqrt(inl.reduce((a, c) => a + lineDist(L, c) ** 2, 0) / inl.length);
  return { ...L, inliers: inl, rms };
}

/** Jusqu'à `max` droites distinctes (hypothèses) pour un même bord. */
function edgeHypotheses(cands, dir, tol, maxAngleDeg, nSamples, max = 4) {
  const out = [];
  let rest = cands;
  for (let h = 0; h < max && rest.length >= 6; h++) {
    const L = fitEdge(rest, dir, tol, maxAngleDeg);
    if (!L) break;
    L.coverage = L.inliers.length / nSamples;
    if (L.coverage >= 0.35) out.push(L);
    rest = rest.filter((c) => lineDist(L, c) > tol * 2);
  }
  return out;
}

const ASPECT = 85.6 / 53.98;

function intersect(L1, L2) {
  const den = cross(L1.d, L2.d);
  if (Math.abs(den) < 1e-9) return null;
  const t = cross(sub(L2.p, L1.p), L2.d) / den;
  return add(L1.p, mul(L1.d, t));
}

function quad(lines) {
  const out = [];
  for (let e = 0; e < 4; e++) {
    const c = intersect(lines[(e + 3) % 4], lines[e]);
    if (!c) return null;
    out.push(c);
  }
  return out;
}

function shape(c) {
  const d = (p, q) => len(sub(p, q));
  const w = (d(c[0], c[1]) + d(c[3], c[2])) / 2;
  const h = (d(c[0], c[3]) + d(c[1], c[2])) / 2;
  return { w, h, aspectErr: w / h / ASPECT - 1 };
}

/**
 * Une passe de détection. Avec `multi`, plusieurs droites candidates par bord sont
 * combinées et l'on retient le quadrilatère aux bonnes proportions — ce qui écarte
 * les lignes imprimées à l'intérieur de la carte, les sourcils, les cheveux…
 */
function pass(img, corners, radius, tol, nSamples, maxAngle, multi, aspectTol = 0.04) {
  const center = mul(corners.reduce(add, { x: 0, y: 0 }), 0.25);
  const hyps = [];
  for (let e = 0; e < 4; e++) {
    const a = corners[e], b = corners[(e + 1) % 4];
    const dir = norm(sub(b, a));
    let n = { x: -dir.y, y: dir.x };
    if (dot(n, sub(a, center)) < 0) n = mul(n, -1); // normale sortante
    const cands = edgeCandidates(img, a, b, n, radius, nSamples);
    const H = edgeHypotheses(cands, dir, tol, maxAngle, nSamples, multi ? 4 : 1);
    if (!H.length) return { error: `bord ${["haut", "droit", "bas", "gauche"][e]} introuvable` };
    hyps.push(H);
  }
  let best = null;
  for (const t of hyps[0]) for (const r of hyps[1]) for (const b of hyps[2]) for (const l of hyps[3]) {
    const lines = [t, r, b, l];
    const c = quad(lines);
    if (!c) continue;
    const s = shape(c);
    if (multi && Math.abs(s.aspectErr) > aspectTol) continue;
    const cov = t.coverage + r.coverage + b.coverage + l.coverage;
    // priorité au soutien des bords, puis au quadrilatère le plus grand (bord extérieur de la carte)
    const score = Math.round(cov * 10) * 1e9 + s.w * s.h;
    if (!best || score > best.score) best = { score, corners: c, lines };
  }
  if (!best) return { error: "aucun rectangle aux proportions d'une carte" };
  return best;
}

/**
 * Affine les 4 coins (ordre : haut-gauche, haut-droit, bas-droit, bas-gauche).
 * radius : rayon de recherche initial en pixels.
 * Retourne { corners, rms, coverage, points } ou { error }.
 */
export function refineCard(img, corners, radius, opts = {}) {
  const nSamples = 48;
  // aspectTol : tolérance sur les proportions (élargie pour la calibration, où la carte est inclinée)
  let r = pass(img, corners, radius, 2, nSamples, opts.maxAngle ?? 10, true, opts.aspectTol ?? 0.04);
  if (r.error) return r;
  // passes fines autour des droites trouvées
  for (const [rad, tol] of [[5, 1.2], [3, 0.8]]) {
    const r2 = pass(img, r.corners, rad, tol, nSamples, 3, false);
    if (r2.error) break;
    r = r2;
  }
  const rms = Math.sqrt(r.lines.reduce((a, L) => a + L.rms ** 2, 0) / 4);
  const coverage = Math.min(...r.lines.map((L) => L.coverage));
  const maxRms = Math.max(...r.lines.map((L) => L.rms));
  return { corners: r.corners, rms, maxRms, coverage, points: r.lines.flatMap((L) => L.inliers.map(({ x, y }) => ({ x, y }))) };
}

/**
 * Points de bord bruts, sans imposer qu'ils soient alignés (pour mesurer la distorsion) :
 * le pic de gradient le plus fort de chaque profil, autour du bord estimé, du signe majoritaire.
 */
export function sampleEdges(img, corners, radius = 4, nSamples = 96) {
  const center = mul(corners.reduce(add, { x: 0, y: 0 }), 0.25);
  const edges = [];
  for (let e = 0; e < 4; e++) {
    const a = corners[e], b = corners[(e + 1) % 4];
    const dir = norm(sub(b, a));
    let n = { x: -dir.y, y: dir.x };
    if (dot(n, sub(a, center)) < 0) n = mul(n, -1);
    const cands = edgeCandidates(img, a, b, n, radius, nSamples, 0.1);
    const best = new Map();
    for (const c of cands) if (!best.has(c.s) || c.w > best.get(c.s).w) best.set(c.s, c);
    const pts = [...best.values()];
    const sign = Math.sign(pts.reduce((acc, p) => acc + p.sign, 0)) || 1;
    edges.push(pts.filter((p) => p.sign === sign).map(({ x, y }) => ({ x, y })));
  }
  return edges;
}
