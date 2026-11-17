# Mesure d'écart pupillaire (EP)

Application web sans compilation qui mesure l'écart pupillaire avec la caméra et une carte au format bancaire.

**En ligne : https://jfcmontmorency.github.io/interpupillary-distance/** (fonctionne aussi sur téléphone, caméra avant ou arrière).

## Lancer en local

```bash
node serve.js 8123
```

Puis ouvrir http://localhost:8123 (la caméra exige `localhost` ou HTTPS). `?debug` expose quelques fonctions de test dans `window.__ipd`.

## Principe

1. **Détection** : MediaPipe Face Landmarker localise le centre de chaque iris (moyenne du point central et du centre des 4 points du contour), la rotation de la tête et l'ouverture des paupières (les clignements sont ignorés).
2. **Échelle** : une carte ID‑1 (85,60 × 53,98 mm) plaquée sur le front.
   - *Détection des bords* ([cardDetect.js](cardDetect.js)) : 48 profils d'intensité perpendiculaires à chaque bord, pics de gradient au sous-pixel, droite robuste (RANSAC + moindres carrés totaux). Plusieurs droites candidates par bord sont combinées et l'on garde le rectangle aux proportions d'une carte. Les coins sont les intersections des droites : les coins arrondis n'interviennent pas.
   - *Carte pivotée ou inclinée* : le raccourci ne réduit qu'une dimension, donc l'échelle retenue est la plus grande des deux (largeur / 85,60 et hauteur / 53,98). Au-delà d'environ 15°, la mesure est rejetée.
   - *Anti-faux positifs* (limite front/cheveux, sourcils) : proportions à ±4 %, bords continus et rectilignes, largeur cohérente avec l'échelle de l'iris.
   - *Sans carte* : diamètre d'iris moyen de 11,7 mm (±2 mm).
3. **Corrections** (EP de loin) : `EP = brute / cos(lacet) × (D + δ + r) / D`
   - `D` : distance caméra → carte ;
   - `δ` (12 mm) : retrait des pupilles derrière la carte (parallaxe) ;
   - `r` (10 mm) : pupille → centre de rotation de l'œil (convergence vers l'objectif).
4. **Œil droit / gauche** : position du nez entre les pupilles calculée en 3D, donc insensible à une légère rotation de la tête.
5. **Statistiques** : rejet des mesures aberrantes (médiane ± 3 MAD). Avec plusieurs essais (historique de session), le résultat principal est la moyenne des essais, un essai isolé (> 1 mm de la médiane) étant écarté.

**Enregistrement automatique** : le contour de la vidéo sert de voyant (rouge : pas de visage, orange : à corriger, vert : mesure). En vert, une mesure est prise au plus toutes les 250 ms, en pleine résolution, avec un bip.

Tout le traitement se fait dans le navigateur ; seuls la bibliothèque et le modèle sont téléchargés (jsDelivr, Google Storage). Aucune image n'est envoyée.
