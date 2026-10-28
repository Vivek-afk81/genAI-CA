# SMS Spam Detection API

A machine learning project that classifies SMS messages as **Spam** or **Ham (Legitimate)** using TF-IDF text vectorization and a Random Forest classifier. The trained model is deployed as a REST API using FastAPI, containerized with Docker, and tested automatically using GitHub Actions.

## Features

- SMS spam classification using machine learning
- TF-IDF text feature extraction
- Random Forest classification model
- FastAPI REST API for real-time predictions
- Docker-based application deployment
- Automated API tests using Pytest
- Continuous Integration (CI) using GitHub Actions

## Tech Stack

- Python 3.11
- Pandas
- Scikit-learn 1.6.1
- Joblib
- FastAPI
- Uvicorn
- Docker
- Pytest
- GitHub Actions
- Google Colab for model training

## Project Structure

```text
genAI-CA/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   └── main.py
├── tests/
│   └── test_api.py
├── model.joblib
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── Readme.md
```

## Dataset

**Dataset:** [SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The dataset contains labeled SMS messages with two important columns:

- `v1`: Message label (`spam` or `ham`)
- `v2`: SMS message text

The downloaded dataset used in this project contains 5,572 messages:

- Ham messages: 4,825
- Spam messages: 747

The dataset was loaded and inspected in Google Colab before model training.

## Machine Learning Workflow

1. Load and inspect the SMS dataset.
2. Select the message text and target label.
3. Split the data into training and testing sets using an 80:20 ratio.
4. Convert text into numerical features using `TfidfVectorizer`.
5. Train a `RandomForestClassifier`.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Save the complete trained pipeline as `model.joblib`.
8. Load the saved pipeline in FastAPI for predictions.

### Model Performance

The Random Forest model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 97.13% |
| Spam Precision | 100% |
| Spam Recall | 78.52% |
| Spam F1-score | 88% |

**Confusion Matrix**

Rows represent actual labels; columns represent predicted labels.

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 966 | 0 |
| Spam | 32 | 117 |

The model correctly classified 966 ham messages and 117 spam messages in the test set. It missed 32 spam messages.

These are results from the project's initial held-out evaluation; further independent evaluation may be needed before real-world use.

## Installation and Local Setup

### Prerequisites

- Python 3.11
- Git
- Docker Desktop (for containerized execution)

### 1. Clone the repository

```bash
git clone https://github.com/Vivek-afk81/genAI-CA.git
cd genAI-CA
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the FastAPI application

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

- Home: http://localhost:8000/
- Health check: http://localhost:8000/health
- Interactive API documentation: http://localhost:8000/docs

## API Endpoints

### 1. Home

**Endpoint:** `GET /`

Returns a message confirming that the API is running.

### 2. Health Check

**Endpoint:** `GET /health`

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Predict SMS Classification

**Endpoint:** `POST /predict`

Example request:

```json
{
  "message": "Congratulations! You have won a free prize. Call now!"
}
```

Example response:

```json
{
  "message": "Congratulations! You have won a free prize. Call now!",
  "prediction": "spam",
  "is_spam": true
}
```

A legitimate message should return `"prediction": "ham"` and `"is_spam": false`.

Use the Swagger UI at `/docs` to interactively test the endpoints.

## Docker Deployment

### 1. Build the Docker image

Run this command from the repository root:

```bash
docker build -t sms-spam-api .
```

### 2. Run the container

```bash
docker run --name sms-spam-container -p 8000:8000 sms-spam-api
```

### 3. Test the containerized API

Open:

http://localhost:8000/docs

Submit a request to `POST /predict` to test the model.

To stop the container, press `Ctrl+C` in the terminal running it. To remove the stopped container, run:

```bash
docker rm sms-spam-container
```

## Automated Testing

The project uses Pytest and FastAPI's `TestClient` to verify API functionality.

Run the tests locally:

```bash
python -m pytest -v
```

The test suite checks:

- Home endpoint availability
- Model health endpoint
- Spam prediction
- Rejection of empty messages

The four tests passed during local verification.

## Continuous Integration — GitHub Actions

The workflow is defined in:

```text
.github/workflows/ci.yml
```

It runs automatically when code is pushed to, or a pull request targets, the `main` or `master` branch.

The pipeline:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs project dependencies.
4. Executes the automated API tests.

**CI Status:** The GitHub Actions workflow completed successfully, with the test job passing.

View the repository and workflow history:

- [GitHub Repository](https://github.com/Vivek-afk81/genAI-CA)
- [GitHub Actions Runs](https://github.com/Vivek-afk81/genAI-CA/actions)

## Limitations

- The model can miss spam messages, as reflected in the test results.
- The dataset may not represent current spam patterns.
- The project is an educational demonstration and is not a production-grade spam filtering service.
- The current CI pipeline tests the API; it does not automatically retrain the model.

## Future Improvements

- Experiment with alternative classifiers and hyperparameter tuning.
- Improve spam recall while monitoring false positives.
- Add more comprehensive input validation and API tests.
- Add Docker image building to the CI workflow.
- Introduce model versioning and monitoring.
- Evaluate the model on newer SMS datasets.

## Author

**Vivek-afk81**

GitHub: [@Vivek-afk81](https://github.com/Vivek-afk81)

---

This project was developed as a machine learning deployment assignment covering model training, serialization, REST API development, containerization, automated testing, and continuous integration.
