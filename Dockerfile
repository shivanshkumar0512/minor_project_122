# SEAF dashboard - Hugging Face Spaces (Docker SDK, free CPU) or any container host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    SEAF_RUNTIME_DIR=/tmp/seaf

# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN chown user:user /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=user . .

# Artifacts are committed; build them only if missing (never at container start).
RUN test -f artifacts/finance/bundle.joblib -a -f artifacts/healthcare/bundle.joblib \
         -a -f artifacts/recruitment/bundle.joblib \
    || python scripts/train_all.py

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/_stcore/health', timeout=4)"

# XSRF/CORS are relaxed because HF serves the app inside an iframe on another origin
# (otherwise st.file_uploader on the Admin page returns 403).
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.enableXsrfProtection=false --server.enableCORS=false"]
