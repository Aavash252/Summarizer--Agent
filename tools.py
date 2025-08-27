import requests
from bs4 import BeautifulSoup

def get_text_from_url(url: str) -> str:
    """Fetches the clean text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        publish_date = None

        # 1. look for <time> tags with datetime attr
        time_tag = soup.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            publish_date = time_tag["datetime"]
        elif time_tag:
            publish_date = time_tag.get_text(strip=True)

        # 2. meta tags (common in news sites)
        if not publish_date:
            meta_date = soup.find("meta", {"property": "article:published_time"})
            if meta_date and meta_date.has_attr("content"):
                publish_date = meta_date["content"]

        if not publish_date:
            meta_date = soup.find("meta", {"name": "date"})
            if meta_date and meta_date.has_attr("content"):
                publish_date = meta_date["content"]
    
        paragraphs = soup.find_all('p')
       
        title = soup.title.string if soup.title else "Untitled"
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return {"title": title,"date": publish_date if publish_date else "Date not found", "text": article_text}
    except requests.RequestException as e:
        return f"Error fetching URL: {e}"