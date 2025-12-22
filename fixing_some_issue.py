import nbformat

notebook_path = r"C:\Users\abood\OneDrive\Dokumente\UNI slides\Grad project\Github-weather\Weather-prediction\Final.ipynb"
nb = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)

# Remove widgets metadata
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, notebook_path)
print("Cleaned and saved:", notebook_path)