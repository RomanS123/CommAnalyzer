import pytest
from unittest.mock import patch
from app.youtube_api import YouTubeAPI

@patch("app.youtube_api.requests.get")  # Используем абсолютный путь
def test_get_comments(mock_get):
    mock_response = {
        "items": [
            {"snippet": {"topLevelComment": {"snippet": {"textDisplay": "Nice video!"}}, "totalReplyCount": 0}}
        ],
        "nextPageToken": None
    }
    mock_get.return_value.json.return_value = mock_response

    api = YouTubeAPI(api_key="fake_key")
    comments = api.get_comments("https://youtube.com/watch?v=12345")
    assert len(comments) == 1
    assert comments[0] == "Nice video!"
