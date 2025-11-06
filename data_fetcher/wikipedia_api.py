import requests
import pandas as pd

def fetch_page_views(lang: str, article: str, start_date: str, end_date: str) -> pd.DataFrame or None:
    """
    Fetches daily page views for a Wikipedia article using the Wikimedia Analytics API.
    
    Args:
        lang (str): Wikipedia language code (e.g., 'en', 'ar', 'fa').
        article (str): Page title (e.g., 'Artificial_intelligence').
        start_date (str): Start date in YYYYMMDD format.
        end_date (str): End date in YYYYMMDD format.
        
    Returns:
        pd.DataFrame or None: DataFrame with 'date' and 'views' columns, or None on failure.
    """
    # Wikimedia Analytics API endpoint for page views
    BASE_URL = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/all-agents/{article}/daily/{start_date}/{end_date}"
    )
    
    # MediaWiki requires a User-Agent header
    # NOTE: Replace 'your_email@example.com' with an actual email for production use
    headers = {
        'User-Agent': 'MyWebTrafficForecaster (your_email@example.com)'
    }

    try:
        response = requests.get(BASE_URL, headers=headers)
        response.raise_for_status() 
        data = response.json()
        
        # Extract and format the items
        views = pd.DataFrame(data['items'])
        views['date'] = pd.to_datetime(views['timestamp'].str[:-2], format='%Y%m%d')
        views = views[['date', 'views']].set_index('date')
        views.index.name = lang.upper() # Set index name for plotting clarity
        return views
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None
    except KeyError:
        print("Error: API response format unexpected. Check if the article name is correct and dates are valid.")
        return None