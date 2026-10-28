from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_spam_prediction():
    response = client.post(
        "/predict",
        json={
            "message": (
                "Congratulations! You have won a free prize. "
                "Call now!"
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"


def test_empty_message():
    response = client.post(
        "/predict",
        json={"message": ""},
    )

    assert response.status_code == 422