# 🫀 Heart Disease Risk Prediction: End-to-End MLOps Pipeline

[![MLOps CI/CD Pipeline](https://github.com/username/heart-disease-mlops/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/username/heart-disease-mlops/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-manifests-326ce5.svg)](https://kubernetes.io/)

Production-grade, end-to-end Machine Learning Operations (MLOps) system built on the **UCI Heart Disease Dataset (Cleveland)**. This repository implements automated data ingestion, exploratory data analysis, Scikit-Learn pipeline engineering, multi-model hyperparameter tuning with **MLflow**, automated **GitHub Actions CI/CD** testing, **FastAPI** containerized serving, **Kubernetes** cluster orchestration, and **Prometheus** observability.

---

## 🏛️ System Architecture

```text
       +-------------------------+
       |   UCI Machine Learning  |
       |      Repository         |
       +------------+------------+
                    |
                    v
       +-------------------------+
       |  Automated Acquisition  |
       |  & EDA (src/eda.py)     |
       +------------+------------+
                    |
                    v
       +-------------------------+
       | Scikit-Learn Pipeline   |
       | Imputation + Scaler +   |
       | OneHotEncoder           |
       +------------+------------+
                    |
                    v
       +-------------------------+         +-------------------------+
       |  GridSearchCV Tuning    | <-----> |   MLflow Tracking UI    |
       | LogReg vs Random Forest |         | (Params, Metrics, Plots)|
       +------------+------------+         +-------------------------+
                    |
                    v
       +-------------------------+
       | Packaged Model Pipeline |
       | (models/*.pkl)          |
       +------------+------------+
                    |
                    v
       +-------------------------+         +-------------------------+
       |  GitHub Actions CI/CD   | ------> |      Docker Engine      |
       | (Lint, Train, Pytest)   |         | (heart-disease-api)     |
       +-------------------------+         +------------+------------+
                                                        |
                                                        v
                                           +-------------------------+
                                           |   Kubernetes Cluster    |
                                           | (2 Replicas, HPA, Svc)  |
                                           +------------+------------+
                                                        |
                                           +------------+------------+
                                           |                         |
                                           v                         v
                                +--------------------+    +--------------------+
                                |  FastAPI Endpoints |    | Prometheus Metrics |
                                |  (/predict, /docs) |    | (/metrics @ 8000)  |
                                +--------------------+    +--------------------+
 