# Modélisation (VaR) : Approche Hybride ARIMA & Quantile Boosting

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Time Series](https://img.shields.io/badge/Time_Series-Econometrics-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-XGBoost-orange.svg)
![Quant Finance](https://img.shields.io/badge/Quantitative-Finance-red.svg)

## Description du Projet
Ce dépôt présente une architecture de modélisation hybride combinant économétrie traditionnelle et algorithmes de Machine Learning non-linéaires. L'objectif est d'estimer le risque extrême (Value at Risk à 95%) d'un portefeuille d'actions.

* **Actif étudié :** Portefeuille équipondéré (1/N) de 6 majors européennes du secteur de l'énergie.
* **Période d'analyse :** 2015 – 2026.
* **Variables exogènes (Macroéconomie à T-1) :** Cours du Brent, paire EUR/USD, indice EuroStoxx 50, indice de volatilité VIX.

---

## Méthodologie et Architecture Mathématique

L'approche repose sur un modèle en deux étapes (Two-Stage Hybrid Model), séparant l'espérance conditionnelle du rendement de sa volatilité extrême.

### 1. Filtrage de la dynamique linéaire : Modèle ARIMA
La première étape consiste à modéliser l'autocorrélation linéaire des rendements journaliers. L'analyse préalable (Test ADF p-value = 0.000, lecture des corrélogrammes ACF/PACF) a justifié l'utilisation d'un modèle parcimonieux **ARIMA(1,0,1)** :

$$r_t = c + \phi_1 r_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$$

Où $r_t$ est le rendement au temps $t$, et $\epsilon_t$ le résidu du modèle.

### 2. Diagnostic des Résidus et Justification Non-Linéaire
L'analyse des résidus de l'ARIMA ($\epsilon_t$) sur l'échantillon d'entraînement révèle les caractéristiques classiques des séries financières à haute fréquence :
* **Queues de distribution épaisses (Fat Tails) :** Kurtosis = 13.52, Test de Jarque-Bera (p-value < 0.05).
* **Hétéroscédasticité conditionnelle :** Test ARCH-LM de Engle (p-value < 0.05).

Ces résultats prouvent que l'ARIMA laisse intacte la structure complexe des risques extrêmes, justifiant le passage à une modélisation non-linéaire sur les résidus.

### 3. Modélisation du Risque : Gradient Boosting & Régression Quantile
Pour estimer la Value at Risk, l'approche classique de prédiction de la moyenne (minimisation de l'erreur quadratique RMSE) est abandonnée au profit d'une **Régression Quantile**.

L'algorithme *Gradient Boosting Regressor* est paramétré pour minimiser la *Pinball Loss* ciblée sur le 5ème centile ($\alpha = 0.05$) :

$$L_{\alpha}(y, \hat{y}) = \max(\alpha(y - \hat{y}), (\alpha - 1)(y - \hat{y}))$$

Le modèle apprend ainsi les relations complexes entre les variables macroéconomiques et les chocs résiduels pour tracer un seuil de risque dynamique. 
L'estimation finale de la VaR s'exprime par : 

$$VaR_{t, 95\%} = \hat{r}_{t, ARIMA} + \hat{\epsilon}_{t, Quantile}$$

---

## Résultats et Backtesting (Out-of-Sample)

Le modèle hybride a été testé sur des données non vues lors de l'entraînement (échantillon de test de 577 jours). L'évaluation de la performance se concentre sur les statistiques de couverture du risque :

* **Taux de dépassement (Hit Rate) :** 2.95% (Pour une cible théorique fixée à 5.00%).
* **VaR Moyenne anticipée :** -2.35%
* **VaR Maximale (Pire choc anticipé) :** -6.24%

Le modèle démontre une capacité robuste à s'ajuster dynamiquement à la volatilité. L'analyse de l'explicabilité (Feature Importance) indique que le modèle s'appuie majoritairement sur les chocs du **Brent** et de la parité **EUR/USD** pour ajuster son filet de sécurité.

---

## Structure du Dépôt

Le projet est divisé en trois notebooks modulaires, respectant un workflow d'analyse séquentiel :

1. `01_Data_Preparation.ipynb` : Collecte, nettoyage des séries temporelles, calcul des rendements et construction du portefeuille équipondéré.
2. `02_ARIMA_Modelling.ipynb` : Tests de stationnarité (ADF), identification (ACF/PACF), entraînement du modèle linéaire et diagnostics économétriques des résidus (Jarque-Bera, ARCH-LM).
3. `03_Quantile_Boosting_VaR.ipynb` : Intégration des variables macroéconomiques, démonstration des limites du RMSE, et implémentation du modèle XGBoost Quantile pour l'estimation finale de la VaR.
