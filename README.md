# Black-Friday-Sales-Analytics-and-Machine-Learning
End-to-end Data Science project analyzing 100,000 Black Friday transactions through EDA, customer segmentation, predictive modeling, and machine learning to generate actionable business insights.

# Deep Analysis of Black Friday Sales with Machine Learning

## Objectif du projet

Ce projet vise à analyser un dataset de transactions Black Friday afin de comprendre les comportements d'achat des clients, identifier les catégories et segments les plus rentables, mesurer l'impact des remises, puis construire des modèles de Machine Learning capables de prédire le montant d'achat et d'identifier les clients à forte valeur.

---

## 1. Présentation du dataset

Le dataset contient **100 000 transactions** et **18 variables** décrivant les clients, les produits, les prix, les remises, les quantités achetées, les moyens de paiement et les informations temporelles.

Chaque ligne représente une transaction réalisée par un client pendant une période commerciale liée au Black Friday.

Les données couvrent plusieurs dimensions :

- Informations clients : âge, genre, ville, segment client
- Informations produits : identifiant produit, catégorie
- Informations transactionnelles : quantité, prix initial, remise, prix final, montant d'achat
- Informations temporelles : date, heure, jour de semaine, week-end, période Black Friday

---

## 2. Qualité des données

L'inspection initiale montre que le dataset est propre et directement exploitable.

Aucune valeur manquante n'a été détectée.  
Aucune ligne dupliquée n'a été identifiée.  
Les types de variables sont globalement cohérents avec leur signification métier.

La colonne `purchase_date` a été convertie en format date afin de créer de nouvelles variables temporelles comme le mois, le jour, la semaine et le jour de semaine.

Cette étape confirme que les données sont suffisamment fiables pour réaliser une analyse exploratoire rigoureuse et entraîner des modèles de Machine Learning.

---

## 3. Analyse des clients

L'analyse démographique montre que les groupes d'âge **36-45 ans**, **18-25 ans** et **26-35 ans** génèrent les revenus les plus élevés.

Le groupe **36-45 ans** apparaît comme le segment le plus rentable avec plus de **7,19 millions de dollars** de chiffre d'affaires. Cela peut s'expliquer par un pouvoir d'achat plus élevé et une forte sensibilité aux promotions.

Les groupes **18-25 ans** et **26-35 ans** représentent également des segments très importants, probablement plus sensibles aux campagnes digitales et aux offres promotionnelles.

Du point de vue marketing, ces trois groupes constituent les principales cibles à privilégier pour les campagnes Black Friday.

---

## 4. Analyse des segments clients

Les clients sont répartis en plusieurs segments : `New`, `Returning`, `Loyal` et `VIP`.

Les **Returning Customers** génèrent le chiffre d'affaires total le plus élevé avec environ **12,13 millions de dollars**. Cela montre que la base de clients existants joue un rôle central dans la performance commerciale.

Les **New Customers** représentent également un volume important de revenus, ce qui indique que le Black Friday constitue un levier efficace d'acquisition client.

Les **VIP Customers**, malgré un revenu total plus faible, affichent le revenu moyen par transaction le plus élevé. Cela signifie qu'ils achètent moins fréquemment, mais avec des paniers plus importants.

Cette analyse montre qu'il existe deux leviers business complémentaires : le volume généré par les clients récurrents et la valeur élevée générée par les clients VIP.

---

## 5. Analyse des produits

L'analyse produit permet d'identifier les catégories et les produits qui contribuent le plus au chiffre d'affaires.

Certaines catégories concentrent une part importante des ventes, ce qui en fait des catégories stratégiques pour la gestion des stocks, les campagnes marketing et l'optimisation des promotions.

Les produits les plus vendus ne sont pas nécessairement les plus rentables. Un produit peut générer beaucoup de ventes avec un faible prix unitaire, tandis qu'un produit premium peut générer un chiffre d'affaires élevé avec un volume plus faible.

Cette distinction est essentielle pour piloter une stratégie commerciale efficace.

---

## 6. Analyse des prix et des quantités

Les distributions des prix montrent une offre diversifiée, allant de produits accessibles à des produits plus premium.

La variable `purchase_amount` présente une distribution asymétrique, ce qui est fréquent dans les données e-commerce : la majorité des transactions ont un montant modéré, tandis qu'une minorité de transactions représente des paniers très élevés.

Les quantités achetées sont généralement faibles à modérées, ce qui indique un comportement d'achat principalement individuel. Les achats en grande quantité apparaissent comme des cas particuliers, potentiellement liés à des achats groupés ou à des clients à forte valeur.

---

## 7. Analyse du chiffre d'affaires

L'analyse des ventes montre que le chiffre d'affaires est fortement influencé par le prix final, la quantité achetée, la période d'achat et le segment client.

Les analyses temporelles permettent d'identifier les jours et heures où les ventes sont les plus importantes. Ces informations sont utiles pour optimiser les campagnes publicitaires, les promotions flash, la gestion des stocks et les ressources logistiques.

Le Black Friday agit comme un accélérateur de consommation, en concentrant les achats sur une période courte mais très intense.

---

## 8. Analyse des remises

L'analyse des remises permet d'évaluer l'impact réel des promotions sur les ventes.

Les réductions jouent un rôle important dans la stimulation de la demande, mais elles ne constituent pas le seul facteur explicatif du chiffre d'affaires.

Le comportement d'achat dépend également de la catégorie de produit, du profil client, du prix final, de la quantité achetée et du contexte temporel.

L'objectif business n'est donc pas simplement d'appliquer les remises les plus élevées, mais d'identifier le niveau optimal de réduction permettant de maximiser les ventes tout en préservant la rentabilité.

---

## 9. Détection des valeurs extrêmes

La méthode IQR a permis d'identifier plusieurs valeurs extrêmes sur les variables de prix, quantité et montant d'achat.

Ces valeurs ne doivent pas être supprimées automatiquement, car dans un contexte Black Friday, elles peuvent représenter des comportements réels : achats premium, gros paniers, achats groupés ou clients VIP.

Ces observations extrêmes sont donc importantes pour comprendre les transactions à forte valeur.

---

## 10. Modélisation Machine Learning : Régression

L'objectif de la régression est de prédire la variable `purchase_amount`, c'est-à-dire le montant total dépensé lors d'une transaction.

Plusieurs variables ont été utilisées, notamment :

- `original_price`
- `discount_pct`
- `quantity`
- `final_price`
- `age_group`
- `gender`
- `customer_segment`
- `product_category`
- `payment_method`
- `city`
- `purchase_hour`
- `is_weekend`
- `discount_amount`
- `effective_discount_rate`
- `revenue_per_item`
- `customer_purchase_freq`
- `is_bulk_purchase`
- `purchase_month`
- `purchase_weekday`

Les données ont été divisées en 80 % pour l'entraînement et 20 % pour le test.

### Résultats des modèles de régression

| Modèle | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 134.39 | 275.60 | 0.7877 |
| Gradient Boosting | 8.15 | 17.27 | 0.9992 |
| Random Forest | 0.28 | 3.33 | 1.0000 |

La régression linéaire obtient une performance correcte avec un R² de **0.7877**, ce qui signifie qu'elle explique environ 78,77 % de la variance du montant d'achat.

Les modèles d'ensemble, notamment Random Forest et Gradient Boosting, obtiennent des performances nettement supérieures. Cela montre que la relation entre les variables explicatives et le montant d'achat est fortement non linéaire.

Cependant, la performance quasi parfaite du Random Forest doit être interprétée avec prudence. Certaines variables comme `final_price`, `quantity` et `revenue_per_item` sont directement liées à la cible `purchase_amount`.

Il existe donc un risque de **data leakage**, car le modèle dispose d'informations très proches de la variable à prédire.

---

## 11. Importance des variables

Les variables les plus importantes sont :

| Variable | Importance |
|---|---:|
| revenue_per_item | 0.37 |
| final_price | 0.34 |
| quantity | 0.21 |
| is_bulk_purchase | 0.09 |
| original_price | ≈ 0.00 |

Cette hiérarchie est cohérente avec la logique métier, car le montant d'achat dépend directement du prix final et de la quantité achetée.

La variable `revenue_per_item` est la plus influente, suivie de `final_price` et `quantity`.

Le faible poids de `original_price` montre qu'une fois le prix final connu, le prix initial apporte peu d'information supplémentaire au modèle.

---

## 12. Classification des clients à forte valeur

Une seconde tâche de Machine Learning consiste à identifier les clients à forte valeur, appelés `high_value_customer`.

Le seuil retenu est d'environ **1 785,48 dollars** de dépenses cumulées.

La répartition des classes est la suivante :

- Classe 0 : 74 997 clients non high-value
- Classe 1 : 25 003 clients high-value

Le dataset est donc modérément déséquilibré, avec environ 75 % de clients ordinaires et 25 % de clients à forte valeur.

### Résultats de la régression logistique

| Métrique | Score |
|---|---:|
| Accuracy | 0.8067 |
| Precision | 0.6918 |
| Recall | 0.4093 |
| F1-score | 0.5143 |

Le modèle obtient une accuracy d'environ **80,67 %**, ce qui semble correct au premier regard.

Cependant, le recall de la classe high-value est faible. Le modèle ne détecte qu'environ 41 % des vrais clients à forte valeur.

Cela signifie que plusieurs clients importants ne sont pas correctement identifiés, ce qui peut être problématique dans une stratégie marketing ciblée.

Le modèle reconnaît mieux les clients ordinaires que les clients premium.

---

## 13. Segmentation client avec K-Means

Une segmentation non supervisée a été réalisée avec K-Means afin d'identifier différents profils clients.

Quatre clusters ont été obtenus.

### Cluster 2 : clients premium

Ce cluster présente les dépenses les plus élevées, avec un total moyen d'environ **4 078 dollars** et un panier moyen élevé.

Il représente les clients à très forte valeur.

### Cluster 0 : clients fidèles

Ce segment présente une fréquence d'achat élevée et un montant total intéressant.

Ce sont des clients réguliers qu'il faut fidéliser.

### Cluster 1 : acheteurs occasionnels

Ces clients ont une dépense moyenne plus modérée et une fréquence d'achat plus faible.

Ils représentent un potentiel de développement.

### Cluster 3 : petits consommateurs

Ce segment regroupe les clients les moins engagés, avec un niveau de dépense plus faible.

Ils peuvent être ciblés par des campagnes d'activation ou de relance.

---

## 14. Recommandations business

À partir des résultats obtenus, plusieurs recommandations peuvent être formulées :

- Renforcer les campagnes marketing sur les groupes d'âge les plus rentables.
- Fidéliser les Returning Customers, qui génèrent le plus de chiffre d'affaires total.
- Proposer des offres exclusives aux VIP Customers afin d'augmenter leur fréquence d'achat.
- Optimiser les remises au lieu d'appliquer systématiquement les réductions les plus fortes.
- Surveiller les produits et catégories stratégiques pour éviter les ruptures de stock.
- Utiliser la segmentation client pour personnaliser les campagnes commerciales.
- Améliorer la détection des clients high-value avec des modèles plus adaptés ou un meilleur équilibrage des classes.

---

## 15. Conclusion générale

Ce projet met en œuvre une démarche complète de Data Science appliquée à des données transactionnelles Black Friday.

L'analyse exploratoire a permis de comprendre les comportements d'achat selon les profils clients, les segments, les produits, les prix, les remises et les périodes d'achat.

Les modèles de Machine Learning ont permis de prédire le montant des transactions et d'identifier les clients à forte valeur.

Les résultats montrent que les variables liées au prix final, à la quantité achetée et au comportement client sont les principaux moteurs du chiffre d'affaires.

La segmentation client apporte une lecture stratégique supplémentaire en distinguant les clients premium, les clients fidèles, les acheteurs occasionnels et les petits consommateurs.

Ce projet démontre comment la Data Science peut transformer des données brutes de vente en insights exploitables pour la prise de décision business, l'optimisation marketing, la gestion des stocks et la personnalisation de l'expérience client.
