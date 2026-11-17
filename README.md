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
   - *Carte pivotée ou inclinée* : le raccourci ne réduit qu'une dimension, donc l'échelle retenue est la plus grande des deux (largeur / 85,60 et hauteur / 53,98). Le front étant incliné vers l'arrière, une carte posée dessus paraît moins haute : ce cas est corrigé, et seule une carte inclinée de plus d'environ 30° est rejetée.
   - *Anti-faux positifs* (limite front/cheveux, sourcils) : proportions à ±4 %, bords continus et rectilignes, largeur cohérente avec l'échelle de l'iris.
   - *Sans carte* : diamètre d'iris moyen de 11,7 mm (±2 mm).
3. **Corrections** (EP de loin) : `EP = brute / cos(lacet) × (D + δ + r) / D`
   - `D` : distance caméra → carte ;
   - `δ` (12 mm) : retrait des pupilles derrière la carte (parallaxe) ;
   - `r` (10 mm) : pupille → centre de rotation de l'œil (convergence vers l'objectif).
4. **Œil droit / gauche** : position du nez entre les pupilles calculée en 3D, donc insensible à une légère rotation de la tête.
5. **Statistiques** : rejet des mesures aberrantes (médiane ± 3 MAD). Avec plusieurs essais (historique de session), le résultat principal est la moyenne des essais, un essai isolé (> 1 mm de la médiane) étant écarté.

**Calibration de la caméra** (expérimentale, dans « Autres options » ; [calib.js](calib.js), mémorisée par caméra) : la carte tenue à la main est placée au centre puis bougée librement ; l'app la suit et capture d'elle-même chaque vue qui apporte une information nouvelle (4 coins de l'image, rotation gauche/droite, bascule haut/bas). Les points de bord masqués par les doigts sont écartés (ajustement parabolique robuste). La distorsion radiale `k1` est estimée par la méthode du fil à plomb (les bords doivent redevenir droits), puis la focale par les contraintes des homographies carte → image (méthode de Zhang simplifiée). Les mesures sont ensuite corrigées de la distorsion et la distance caméra → carte est calculée (`D = f × mm/px`). Sur données synthétiques : focale à 0,2 % près, distance à moins de 1 %.

**Enregistrement automatique** : le contour de la vidéo sert de voyant (rouge : pas de visage, orange : à corriger, vert : mesure). En vert, une mesure est prise au plus toutes les 250 ms, en pleine résolution, avec un bip.

Tout le traitement se fait dans le navigateur ; seuls la bibliothèque et le modèle sont téléchargés (jsDelivr, Google Storage). Aucune image n'est envoyée.
