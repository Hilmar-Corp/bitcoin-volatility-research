# HC-BTCVOL-001 — Benchmark scientifique et institutionnel (FR)

**Statut :** dossier de recherche documentaire et de préparation éditoriale, version du 20 octobre 2026.  
**Important :** les institutions ci-dessous sont des *références externes*, pas des évaluateurs, sponsors, validateurs ou partenaires de cette étude. Ce dossier ne modifie ni les expériences ni l'inférence enregistrée avant l'ouverture du holdout.

## 1. Mission : deux niveaux de publication complémentaires

**Working paper scientifique** : expliciter la question de recherche, la littérature, la construction des données, la causalité temporelle des prévisions, les estimateurs, les tests, les résultats et les limitations, avec équations et annexes reproductibles.

**Note de recherche destinée aux investisseurs institutionnels** : expliciter l'intérêt pour la mesure/prévision du risque, les hypothèses, la portée réelle et les défaillances possibles, avec un résumé lisible et les mêmes résultats numériques. La note ne transforme pas la précision de prévision en performance d'investissement.

Deux versions intégrales **FR + EN**, mêmes figures, chiffres, période, tableaux et réserves. La version anglaise peut être diffusée comme working paper ; la version française n'est pas une plaquette commerciale.

## 2. Références institutionnelles vérifiées

| Référence | Document source | Enseignement éditorial / scientifique pour HilmarCorp |
|---|---|---|
| MIT Sloan Finance | https://mitsloan.mit.edu/faculty/academic-groups/finance/research | Question identifiable, contribution vérifiable, protocole et limites, pas seulement graphiques |
| Cambridge Judge Working Papers | https://www.jbs.cam.ac.uk/faculty-research/publications/working-papers/ | Working papers explicitement préliminaires ; séparer étude, bibliographie et revue par les pairs |
| ENSAE Paris / CREST / École polytechnique | https://www.ensae.fr/recherche/centre-de-recherche-en-economie-et-statistique-crest | Rigueur quantitative et confrontation des modèles aux observations |
| École polytechnique / CMAP MathsFi | https://portail.polytechnique.edu/cmap/en/research/probability/mathsfi | Mathématiques financières, modélisation du risque, optimisation numérique, microstructure |
| NYU Stern V-Lab | https://vlab.stern.nyu.edu/about | Benchmark pertinent en volatilité/risk forecasting ; distinguer expérience historique de suivi temps réel |
| BlackRock Investment Institute | https://www.blackrock.com/institutions/en-global/institutional-insights/thought-leadership/capital-market-assumptions | Présenter dates, méthodologies, hypothèses, incertitudes, limites et champ d'application |
| Amundi Investment Institute | https://research-center.amundi.com/publications/working-paper | Working papers de recherche financière avec abstract et application à l'investissement |
| Amundi : surfaces de volatilité implicite (2026) | https://research-center.amundi.com/article/modeling-and-forecasting-large-unbalanced-option-implied-volatility-surfaces | Exemple *distinct* de notre objet : prévision séquentielle, benchmark explicite, application empirique |
| Amundi / CREST : DCC-GARCH (2016) | https://research-center.amundi.com/article/stationarity-dynamic-conditional-correlation-models | Exemple de pont recherche mathématique / buy-side ; ne pas confondre GARCH univarié et DCC multivarié |
| Federal Reserve SR 26-2 (17 avril 2026) | https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm | Inspirer l'inventaire de modèle, validation conceptuelle, contrôle des résultats, monitoring, gouvernance ; **non applicable comme certification de cette étude** |
| BIS / BCBS Stress Testing Principles (2018) | https://www.bis.org/publications/201810-guidelines-stress-testing-principles | Qualité des données, documentation, hypothèses, scénarios et gouvernance, si un usage de stress testing est envisagé |

**Vigilance réglementaire 2026 :** SR 26-2 remplace SR 11-7. Ne pas décrire SR 11-7 comme l'instruction actuelle ; ne pas assimiler cette recherche à un modèle réglementaire validé.

## 3. Bibliographie scientifique fondamentale — références primaires

1. **Corsi (2009)**, *A Simple Approximate Long-Memory Model of Realized Volatility*. *Journal of Financial Econometrics*. https://doi.org/10.1093/jjfinec/nbp001. Référence au modèle HAR-RV original. **Notre variante utilise des moyennes de log-RV**, ce qui n'est pas identique au modèle original à agrégation arithmétique.
2. **Patton (2011)**, *Volatility forecast comparison using imperfect volatility proxies*. *Journal of Econometrics*. https://doi.org/10.1016/j.jeconom.2010.03.034. Rôle des proxys imparfaits et pertinence du QLIKE pour la comparaison de variances conditionnelles sous ses hypothèses.
3. **Diebold & Mariano (1995)**, *Comparing Predictive Accuracy*. *Journal of Business & Economic Statistics*. https://doi.org/10.1080/07350015.1995.10524599. Test de différence de pertes de prévision.
4. **Newey & West (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*. *Econometrica*. https://doi.org/10.2307/1913610. Variance de long terme et corrélation sérielle.
5. **Holm (1979)**, *A Simple Sequentially Rejective Multiple Test Procedure*. *Scandinavian Journal of Statistics*. https://www.jstor.org/stable/4615733. Correction des deux tests primaires.
6. **Bollerslev (1986)**, *Generalized Autoregressive Conditional Heteroskedasticity*. *Journal of Econometrics*. https://doi.org/10.1016/0304-4076(86)90063-1. Modèle GARCH.
7. **Andersen, Bollerslev, Diebold & Labys (2003)**, *Modeling and Forecasting Realized Volatility*. *Econometrica*. https://doi.org/10.1111/1468-0262.00418. Fondements des mesures réalisées.
8. **Duan (1983)**, *Smearing Estimate: A Nonparametric Retransformation Method*. *JASA*. https://doi.org/10.1080/01621459.1983.10478017. Retransformation log-RV.
9. **Volatility prediction comparison via robust volatility proxies: An empirical deviation perspective (2024)**, *Journal of Econometrics* 239(2), 105633. https://doi.org/10.1016/j.jeconom.2023.105633. Travail directement lié à la stabilité des comparaisons sur **Bitcoin** ; ses méthodes supplémentaires ne font pas partie de notre protocole préenregistré.

Les liens donnent accès aux pages institutionnelles ou des éditeurs ; le texte intégral peut nécessiter un abonnement. Les résultats empiriques de ces articles **ne sont pas des métriques comparables** à celles de HC-BTCVOL-001 sans réplication sur le même échantillon.

## 4. Éléments empiriques non négociables — chiffres gelés

- Instrument : **Binance Spot BTCUSDT**, rendement logarithmique, bougies **5 minutes**, journées **UTC**, période globale **2020-01-01 → 2026-09-30**.
- Données : **82 archives**, **709 920** horodatages attendus, **709 458** bougies observées, **462** absences, **16 journées RV invalides** sans interpolation.
- Sept modèles : rolling variance 30j ; EWMA daily ; ARCH(1) ; GARCH(1,1) ; rolling RV 30j ; EWMA-RV ; variante **log-HAR-RV(1,7,30)**.
- Holdout **2025-01-01 → 2026-09-30**, **638** journées communes, *first look* unique après gel.
- QLIKE = \`log(h_t) + RV_t / h_t\` ; score inférieur = meilleure prévision. **Les QLIKE moyennes négatives sont admissibles** et dépendent des unités de variance. Ne pas interpréter les gains en rendement financier ou pourcentage de profit.
- HAR vs EWMA-RV : **gain QLIKE +0,08827622** ; **p de Holm 0,00514178** ; IC bootstrap supplémentaire **[0,04149298 ; 0,15364046]**.
- GARCH vs EWMA daily : **gain +0,03848423** ; **p de Holm 0,16063413** ; différence **non significative** dans ce protocole.
- Post-holdout : l'avantage HAR reste descriptivement positif sous exclusions extrêmes mais se réduit ; par exemple **+0,01913281** après exclusion des 5 % plus fortes différences absolues de pertes. **Cela ne prouve pas de significativité sur le sous-échantillon.**
- Audit second solveur : **1 276 / 1 276** réestimations ARCH/GARCH SLSQP convergées, zéro alerte numérique. **Pas d'audit externe.**

## 5. Grille de recevabilité académique et buy-side

| Critère de lecture | Statut actuel | Ce que le manuscrit doit rendre explicite |
|---|---|---|
| Protocole / anti-look-ahead | Documenté et gelé | Horaires de formation de l'information, fenêtres, coupure validation/holdout |
| Construction RV et anomalies | Documentée | Formule exacte, bordures de jours, absence de comblement des lacunes |
| Estimation des paramètres | Validée numériquement | Définition des récurrences, initialisation, vraisemblance, hypothèses |
| Comparateurs sur échantillon commun | Acquis | Tableau complet à N=638, famille quotidienne vs intraday |
| Tests confirmatoires | Acquis | Seuls les deux contrastes préenregistrés, DM-HAC lag 7, Holm |
| Sensibilité HAC/bootstrap/extrêmes | Étudiée après holdout | Séparer clairement diagnostics exploratoires et décision gelée |
| Incertitude et généralisation | Limitations documentées | Une venue, un actif, une période ; risque proxy/bruit microstructure |
| Données brutes totalement réexécutables depuis le seul dépôt | **Non** | Source Binance et SHA des 82 ZIP, limites d'accès et droit de redistribution |
| CI reproductible depuis les artefacts de sortie | Oui | Version Python, environnement figé, commandes et empreintes |
| Validation externe | **Non** | Décrire l'absence de revue par les pairs ou d'attestation institutionnelle |
| Valeur économique / trading / portefeuille | **Non étudiée** | Ne pas promettre d'alpha ou d'amélioration du Sharpe d'une stratégie |

## 6. Structure cible des deux manuscrits (FR et EN)

1. Page de titre, déclaration de portée et dates de version.
2. **Résumé / Abstract** : question, données, méthode, test préenregistré, résultat limité et réserve.
3. **Introduction / Introduction** : importance de la prévision de variance BTC et contribution empirique mesurable.
4. **Littérature / Related literature** : RV, HAR, GARCH, bruit du proxy, comparaison des prévisions, spécificité BTC.
5. **Données et construction des variables / Data and measurement** : extraction, qualité, exclusions, formule RV, UTC.
6. **Modèles et estimation / Models and estimation** : sept modèles, récurrences, log-HAR exact, QMLE, Duan.
7. **Plan expérimental / Experimental design** : séparation chronologique, fenêtres 730j, absence de réoptimisation OOS.
8. **Inférence / Statistical inference** : QLIKE, perte paire, DM-HAC, Holm, bootstrap.
9. **Résultats / Results** : QLIKE et deux tests ; contrôle visuel, résultat qui n'est *pas* significatif.
10. **Robustesses *post-holdout* / Post-holdout diagnostics** : taux d'influence extrême, HAC, sous-périodes, second solveur.
11. **Limites et implications / Limitations and interpretation** : biais, transférabilité, distinction risque / performance de trading.
12. **Conclusion / Conclusion** : uniquement la portée testée.
13. **Références / References**.
14. **Annexes / Appendices** : détails mathématiques, tableaux SHA, audit source, valeurs des paramètres, instructions CI.

**Livrables distincts :** working paper FR, working paper EN, note institutionnelle FR (~2-4 pages), note institutionnelle EN (~2-4 pages). Les quatre doivent renvoyer aux **mêmes** versions Git figées, pas à des résultats remaniés.

## 7. Contrôle éditorial de publication

**Go** seulement si : les tableaux correspondent aux artefacts vérifiés ; toutes les figures indiquent unités et périodes ; les équations reproduisent le code réel ; aucun post-holdout n'est annoncé comme confirmatoire ; les sources sont vérifiées à leur URL originale ; les versions FR/EN concordent ; les données externes ne sont pas présentées comme licenciées par Apache-2.0 ; un relecteur externe éventuel est nommé seulement après accord.

**Interdits :** « BlackRock approved », « Amundi validated », « MIT grade certified », « profitable strategy demonstrated », « HAR significantly beats all seven models », « out-of-sample 100% public raw-data reproduction ». Aucun de ces énoncés n'est établi par la recherche actuelle.

**Statut final de ce document : BENCHMARK ET CAHIER DES CHARGES, PAS MANUSCRIT.**
