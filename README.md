# MLOps Experiment 8: Iris Flower Classifier

End-to-end MLOps project by **E056 Nisarg Shah** (BTI DS, Sem 11, E2 batch).

| Stage | Tool | Files |
| --- | --- | --- |
| Model | scikit-learn Random Forest | `src/train.py` |
| Serving API | FastAPI | `src/api.py` |
| Version control | Git, GitHub | this repository |
| CI/CD | GitHub Actions | `.github/workflows/ci.yml` |
| Container | Docker | `Dockerfile` |
| Orchestration | Kubernetes (kind) | `k8s/` |
| Monitoring | Prometheus, Grafana | `k8s/prometheus.yaml`, `k8s/grafana.yaml`, `monitoring/` |
| Deployment | Streamlit Community Cloud | `streamlit_app.py` |

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train
pytest -v
streamlit run streamlit_app.py
```

## Docker and Kubernetes

```bash
docker build -t iris-api:1.0 .
kind create cluster --config k8s/kind-config.yaml
kind load docker-image iris-api:1.0 --name mlops-exp8
kubectl apply -f k8s/app.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl create configmap grafana-dashboard -n mlops --from-file=monitoring/iris-dashboard.json
kubectl apply -f k8s/grafana.yaml
python scripts/generate_traffic.py
```

| Service | URL |
| --- | --- |
| API docs | http://localhost:30080/docs |
| Prometheus | http://localhost:30090 |
| Grafana | http://localhost:30300 |
