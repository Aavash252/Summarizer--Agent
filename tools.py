import os
import requests
from bs4 import BeautifulSoup

def get_text_from_url(url: str) -> str:
    """Fetches the clean text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        publish_date = None

        time_tag = soup.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            publish_date = time_tag["datetime"]
        elif time_tag:
            publish_date = time_tag.get_text(strip=True)

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


def get_text_from_file(filepath: str) -> str:
    """Reads and returns the text content from a local file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found at {filepath}")
        return ""
    except Exception as e:
        print(f"Warning: Could not read file at {filepath}: {e}")
        return ""
    
def write_summary_to_file(filepaths: list, summary_points: list, output_filename: str = "summary_report.txt"):
    """
    Formats the analysis results and writes them to a text file.
    """
    print(f"\nWriting summary to {output_filename}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("--- FOUR-NEWS SUMMARY REPORT ---\n")
            
            for i, p in enumerate(summary_points, 1):
                f.write(f"\n--- Article {i}---\n")
                
                if isinstance(p, dict):
                    point = p.get('key_point', 'N/A')
                    category = p.get('category', 'N/A')
                    sentiment = p.get('sentiment', 'N/A')
                else: 
                    point = p.key_point
                    category = p.category
                    sentiment = p.sentiment
                
                f.write(f"- Point:     {point}\n")
                f.write(f"- Category:  {category}\n")
                f.write(f"- Sentiment: {sentiment}\n")

        print(f"Successfully saved report to {output_filename}")
    except Exception as e:
        print(f"Error: Could not write to file. {e}")

def write_summary_to_file(summary_list: list, source_type: str, output_filename: str = "summary_report.txt"):
    """Formats the analysis results and writes them to a text file."""
    print(f"\nWriting summary to {output_filename}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
           
            f.write(f"--- NEWS ANALYSIS REPORT ({len(summary_list)} ARTICLES) (Source: {source_type.upper()}) ---\n")
            
            for i, p in enumerate(summary_list, 1):
                source_display = os.path.basename(p.get('source', 'N/A'))
                f.write(f"\n--- Article {i} (Source: {source_display}) ---\n")
                f.write(f"- Point:     {p.get('key_point', 'N/A')}\n")
                f.write(f"- Category:  {p.get('category', 'N/A')}\n")
                f.write(f"- Sentiment: {p.get('sentiment', 'N/A')}\n")

        print(f"Successfully saved report to {output_filename}")
    except Exception as e:
        print(f"Error: Could not write to file. {e}")