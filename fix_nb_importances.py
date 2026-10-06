
import json
import glob

for path in glob.glob("e:/Projects/Studies/ElectricityConsumption/notebooks/*.ipynb"):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            for i, line in enumerate(source):
                if "importance_features(modele, colonnes_features" in line:
                    source[i] = line.replace("importance_features(modele, colonnes_features", "evaluator.plot_feature_importances(modele, colonnes_features")
                if "importance_features" in line and "import" in line:
                    source[i] = line.replace(", importance_features", "").replace("importance_features,", "").replace("importance_features", "")
            cell["source"] = source
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

