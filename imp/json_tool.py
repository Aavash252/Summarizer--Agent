from newspaper import Article
import logging

# Suppress unwanted log messages from newspaper
logging.getLogger('newspaper').setLevel(logging.CRITICAL)

def get_article_data(url: str) -> dict:
    """
    Uses newspaper3k to extract clean text and metadata from a URL.
    Returns a dictionary with 'text' and 'title'.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        if not article.text:
            return {"error": "Could not extract meaningful article text from the URL."}

        return {
            "text": article.text,
            "title": article.title
        }
    except Exception as e:
        return {"error": f"An error occurred while processing the URL: {e}"}