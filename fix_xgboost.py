
import os
path = "src/models/xgboost/xgboost_regressor.py"
with open(path, "r", encoding="mbcs") as f: # Read what powershell likely wrote
    content = f.read()

# Replace broken characters
content = content.replace("Modèle", "Modele").replace("l`évaluateur", "l evaluateur").replace("saison_Eté", "saison_Ete")
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

