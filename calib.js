// Calibration de la caméra à partir de plusieurs vues d'une carte ID-1 tenue à la main.
//
// 1. Distorsion radiale (méthode « fil à plomb ») : les bords d'une carte sont parfaitement
//    droits ; on cherche le coefficient k1 qui redresse au mieux tous les bords observés.
//    Modèle de correction : p' = c + (p − c)·(1 + k1·ρ²), ρ = distance au centre / demi-diagonale.
// 2. Focale (méthode de Zhang simplifiée) : pour chaque vue, l'homographie carte → image
//    impose deux contraintes sur la focale (centre optique au milieu de l'image, pixels carrés).
//    Les vues inclinées sont celles qui la déterminent.

export const CARD_W = 85.6;
export const CARD_H = 53.98;

/** Corrige la distorsion d'un point (coordonnées image en pixels). */
export function undistort(cal, p) {
  if (!cal) return p;
  const dx = p.x - cal.cx, dy = p.y - cal.cy;
  const r2 = (dx * dx + dy * dy) / (cal.R * cal.R);
  const s = 1 + cal.k1 * r2;
  return { x: cal.cx + dx * s, y: cal.cy + dy * s };
}

/** Droite par moindres carrés totaux : résidu RMS et étendue des points le long de la droite. */
function lineFit(pts) {
  const n = pts.length;
  let mx = 0, my = 0;
  for (const p of pts) { mx += p.x; my += p.y; }
  mx /= n; my /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) {
    const dx = p.x - mx, dy = p.y - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  const a = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const ux = Math.cos(a), uy = Math.sin(a);
  let r2 = 0, lo = Infinity, hi = -Infinity;
  for (const p of pts) {
    const dx = p.x - mx, dy = p.y - my;
    const along = dx * ux + dy * uy, across = -dx * uy + dy * ux;
    r2 += across * across;
    lo = Math.min(lo, along); hi = Math.max(hi, along);
  }
  return { rms: Math.sqrt(r2 / n), len: hi - lo };
}

/** Courbure résiduelle moyenne des bords (sans unité : résidu / longueur du bord). */
export function straightness(views, cal) {
  let cost = 0, n = 0;
  for (const v of views) {
    for (const edge of v.edges) {
      if (edge.length < 10) continue;
      const { rms, len } = lineFit(edge.map((p) => undistort(cal, p)));
      if (len <= 0) continue;
      cost += (rms / len) ** 2;
      n++;
    }
  }
  return n ? Math.sqrt(cost / n) : Infinity;
}

/** Minimisation 1D par section dorée. */
function golden(f, a, b, iters = 60) {
  const g = (Math.sqrt(5) - 1) / 2;
  let c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
  for (let i = 0; i < iters; i++) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
    else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
  }
  return (a + b) / 2;
}

/** Résolution d'un système linéaire (élimination de Gauss avec pivot partiel). */
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-12) return null;
    for (let r = c + 1; r < n; r++) {
      const k = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
    }
  }
  const x = new Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let j = r + 1; j < n; j++) s -= M[r][j] * x[j];
    x[r] = s / M[r][r];
  }
  return x;
}

/** Homographie (h33 = 1) envoyant 4 points src sur 4 points dst. */
export function homography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i], { x: u, y: v } = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  const h = solve(A, b);
  return h && [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

const CARD_PTS = [
  { x: -CARD_W / 2, y: -CARD_H / 2 }, { x: CARD_W / 2, y: -CARD_H / 2 },
  { x: CARD_W / 2, y: CARD_H / 2 }, { x: -CARD_W / 2, y: CARD_H / 2 },
];

/**
 * Focale en pixels à partir des coins (déjà corrigés de la distorsion) de plusieurs vues.
 * Chaque homographie H = K·[r1 r2 t] donne : r1·r2 = 0 et |r1| = |r2|, avec K = diag(f, f, 1).
 */
export function solveFocal(views, cal) {
  const S = 1 / 1000; // mise à l'échelle pour le conditionnement numérique
  let sAA = 0, sAB = 0, used = 0;
  for (const v of views) {
    const dst = v.corners.map((p) => {
      const q = undistort(cal, p);
      return { x: (q.x - cal.cx) * S, y: (q.y - cal.cy) * S };
    });
    const H = homography(CARD_PTS, dst);
    if (!H) continue;
    const [h00, h01] = [H[0][0], H[0][1]], [h10, h11] = [H[1][0], H[1][1]], [h20, h21] = [H[2][0], H[2][1]];
    // a = 1/f² (en unités mises à l'échelle) ; deux équations linéaires a·A + B = 0
    const rows = [
      [h00 * h01 + h10 * h11, h20 * h21],
      [h00 * h00 + h10 * h10 - h01 * h01 - h11 * h11, h20 * h20 - h21 * h21],
    ];
    for (const [A, B] of rows) {
      // normalisation : chaque équation pèse selon l'information qu'elle apporte
      const w = 1 / (h00 * h00 + h10 * h10 + h01 * h01 + h11 * h11);
      sAA += A * A * w * w; sAB += A * B * w * w;
    }
    used++;
  }
  if (used < 3 || sAA <= 0) return null;
  const a = -sAB / sAA;
  if (!(a > 0)) return null;
  return 1 / Math.sqrt(a) / S;
}

/** Calibration complète. views : [{ corners: [4 points], edges: [[points] × 4] }]. */
export function calibrate(views, W, H) {
  const base = { cx: W / 2, cy: H / 2, R: Math.hypot(W, H) / 2, k1: 0 };
  const before = straightness(views, base);
  const k1 = golden((k) => straightness(views, { ...base, k1: k }), -0.5, 0.5);
  const cal = { ...base, k1 };
  const after = straightness(views, cal);
  const f = solveFocal(views, cal);
  const hfov = f ? (2 * Math.atan(W / 2 / f) * 180) / Math.PI : null;
  return { ...cal, W, H, f, hfov, before, after, n: views.length };
}

/** Projection d'une carte posée en (X, Y, Z) mm, tournée (lacet, tangage en degrés) : cibles et tests. */
export function projectCard(f, cx, cy, X, Y, Z, yawDeg = 0, pitchDeg = 0, w = CARD_W, h = CARD_H) {
  const a = (yawDeg * Math.PI) / 180, b = (pitchDeg * Math.PI) / 180;
  return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => {
    let x = (sx * w) / 2, y = (sy * h) / 2, z = 0;
    [x, z] = [x * Math.cos(a) + z * Math.sin(a), -x * Math.sin(a) + z * Math.cos(a)];
    [y, z] = [y * Math.cos(b) - z * Math.sin(b), y * Math.sin(b) + z * Math.cos(b)];
    return { x: cx + (f * (x + X)) / (Z + z), y: cy + (f * (y + Y)) / (Z + z) };
  });
}
