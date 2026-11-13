# Submission Checklist

## 1. Review the completed analysis

- Read `MODEL_REPORT.md`
- Confirm the app title and explanatory language fit your course expectations
- If you want to reproduce the tables, run `python export_results.py`

## 2. Test locally

From the project folder, run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

In the browser, verify:

- Store selection changes the forecast and metrics
- A one-month selection works
- A continuous multi-month selection works
- The 80% band can be shown and hidden
- All three tabs open
- The forecast CSV download works

## 3. Put the project in GitHub

1. Create a public repository named `bakery-forecast-lab`
2. Open the downloaded project folder
3. Upload the **contents** of the folder to the repository root
4. Confirm these paths exist in GitHub:

   ```text
   app.py
   modeling.py
   requirements.txt
   data/bakerysales.csv
   .streamlit/config.toml
   ```

5. Commit to the `main` branch

Do not place every file inside an extra nested folder. Streamlit should be able
to find `app.py` at the repository root.

## 4. Deploy with Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with the GitHub account that owns the repository
3. Select **Create app**
4. Choose the `bakery-forecast-lab` repository
5. Select branch `main`
6. Set the main file path to `app.py`
7. Select **Deploy**

If the repository is missing, open Streamlit's GitHub authorization settings and
grant access to this repository. If the app fails during installation, open the
deployment logs and confirm `requirements.txt` is at the repository root.

## 5. Final public-link check

Open the Streamlit URL in a private/incognito browser window. This verifies that
your professor can access the app without your GitHub or Streamlit login.

## 6. Submit

Submit the public Streamlit URL. If the assignment permits supporting files, also
include the GitHub repository URL and `MODEL_REPORT.md`.

