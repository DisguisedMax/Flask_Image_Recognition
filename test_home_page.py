"""Tests for the Flask home page."""


def test_home_page_loads(client):
    """GET / should return the upload page successfully."""
    response = client.get("/")
    assert response.status_code == 200
