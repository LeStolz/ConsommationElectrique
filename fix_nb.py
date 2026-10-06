
import json
path = "e:/Projects/Studies/ElectricityConsumption/notebooks/02_evaluation.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "freq=chr(39)" in line:
                source[i] = line.replace("freq=chr(39)+\"10D\"+chr(39)", "freq=\"10D\"")
                source[i] = source[i].replace("freq=chr(39)+\"1D\"+chr(39)", "freq=\"1D\"")
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

