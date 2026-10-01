# Mwanza multi-stage healthcare allocation: interactive demo

DSAI 6220 Advanced Machine Learning, Group 1 (NM-AIST).

The app reads the tables produced by the Colab notebook and re-runs the two
optimization stages live, so the Year-1 and Year-2 budgets can be changed with sliders.

## Files
- `app.py`: the Streamlit app
- `requirements.txt`: Python packages
- `data/mwanza_tiles_access.csv`: one row per 640 m tile (CNN label, population, distance)
- `data/mwanza_health_facilities_osm.csv`: the cleaned health facilities

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
