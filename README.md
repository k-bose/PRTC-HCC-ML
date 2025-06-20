# PRTC-HCC-ML
Reproducing MLs models from the Research Paper "Predicting Response to Transarterial Chemoembolization in Hepatocellualar Carcinoma Using Machine Learning Models"

# Environment
1. Create and activate Python Virtual Environment
    ```bash
    $ python3.11 -m venv .venv
    $ source .venv/bin/activate
    ```
2. Install supporting packages
    ```bash
    $ pip install "numpy<2.0"
    $ pip install versioneer
    ```
3. Install PyRadiomics
    ```bash
    $ pip install pyradiomics --no-build-isolation
    ```
4. Install rest of the packages
    ```bash
    $ pip install -r requirements.txt
    ```