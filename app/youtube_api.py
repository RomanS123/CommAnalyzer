import requests
import re

class YouTubeAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_comments(self, video_url):
        video_id = self.extract_video_id(video_url)

        RESULTS_AM = 100
        curr_results = RESULTS_AM
        nextPageToken = None
        comments = list()

        base_url = "https://www.googleapis.com/youtube/v3/commentThreads"

        while curr_results == RESULTS_AM:
            try:

                params = {
                    "part": "snippet,replies",
                    "videoId": video_id,
                    "key": self.api_key,
                    "maxResults": RESULTS_AM,
                    "pageToken": nextPageToken,
                    "textFormat": "plainText"
                }

                response = requests.get(base_url, params=params)
                data = response.json()

                for item in data["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comments.append(re.sub(r'@@\w+', '', comment))

                    if item["snippet"]["totalReplyCount"] > 0:

                        # Собираем ответы
                        if "replies" in item:
                            for reply in item["replies"]["comments"]:
                                reply_text = reply["snippet"]["textDisplay"]
                                comments.append(
                                    re.sub(r'@@\w+', '', reply_text))

                    nextPageToken = data.get("nextPageToken", "")
                    curr_results = len(data["items"])

            except Exception as e:
                print(f"Error: {e}")
                break
        return comments

    @staticmethod
    def extract_video_id(video_url):
        return video_url.split("v=")[-1]
