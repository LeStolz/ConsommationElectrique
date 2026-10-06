
import json
import glob

for path in glob.glob("e:/Projects/Studies/ElectricityConsumption/notebooks/*.ipynb"):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            for i, line in enumerate(source):
                if "evaluer(modele, X_val, y_val, horizon_h=val[\"horizon_h\"])" in line:
                    source.insert(i, line.replace("resultats_val, pred_val = evaluer(modele, X_val, y_val, horizon_h=val[\"horizon_h\"])", "pred_val = modele.predict(X_val)"))
                    source[i+1] = line.replace("resultats_val, pred_val = evaluer(modele, X_val, y_val, horizon_h=val[\"horizon_h\"])", "resultats_val = evaluator.evaluate_metrics(y_val, pred_val, horizon_h=val[\"horizon_h\"])")
                elif "evaluer(modele, X_train, y_train)" in line:
                    source.insert(i, line.replace("resultats_train, _ = evaluer(modele, X_train, y_train)", "pred_train = modele.predict(X_train)"))
                    source[i+1] = line.replace("resultats_train, _ = evaluer(modele, X_train, y_train)", "resultats_train = evaluator.evaluate_metrics(y_train, pred_train)")
                elif "evaluer(modele, X_test, y_test, horizon_h=test[\"horizon_h\"])" in line:
                    source.insert(i, line.replace("resultats_test, pred_test = evaluer(modele, X_test, y_test, horizon_h=test[\"horizon_h\"])", "pred_test = modele.predict(X_test)"))
                    source[i+1] = line.replace("resultats_test, pred_test = evaluer(modele, X_test, y_test, horizon_h=test[\"horizon_h\"])", "resultats_test = evaluator.evaluate_metrics(y_test, pred_test, horizon_h=test[\"horizon_h\"])")
                elif "evaluer(modele, X_val, y_val)" in line:
                    source.insert(i, line.replace("resultats_val, _ = evaluer(modele, X_val, y_val)", "pred_val = modele.predict(X_val)"))
                    source[i+1] = line.replace("resultats_val, _ = evaluer(modele, X_val, y_val)", "resultats_val = evaluator.evaluate_metrics(y_val, pred_val)")
                
                if "evaluer" in line and "import" in line:
                    source[i] = line.replace(", evaluer", "").replace("evaluer,", "").replace("evaluer", "")
            cell["source"] = source
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

