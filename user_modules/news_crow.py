import core
import asyncio
import re

class NewsCrow(core.module.Module):
    """
    A professional RSS news aggregator. 
    Pulls categorized news from major global sources.
    """

    settings = {}
    dependencies = ["feedparser", "requests"]

    # Built-in directory of reliable RSS feeds
    FEED_DIRECTORY = {
        "world": [
            "http://feeds.bbci.co.uk/news/world/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
        ],
        "politics": [
            "https://rss.politico.com/politics-news.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml"
        ],
        "technology": [
            "https://techcrunch.com/feed/",
            "https://www.wired.com/feed/rss"
        ],
        "cybersecurity": [
            "https://feeds.feedburner.com/TheHackersNews",
            "https://www.bleepingcomputer.com/feed/"
        ],
        "business": [
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?profile=120000000&id=10000115",
            "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml"
        ],
        "science": [
            "https://www.sciencedaily.com/rss/all.xml"
        ],
        "gaming": [
            "https://feeds.ign.com/ign/news"
        ]
    }

    def _strip_html(self, text: str) -> str:
        """Helper to remove HTML tags from summaries for a cleaner AI read."""
        if not text:
            return "No summary provided."
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        # Remove extra whitespace and newlines
        return " ".join(text.split())

    async def get_news(self, category: str = None, custom_url: str = None):
        """
        Fetches the latest news articles. You must provide EITHER a category OR a custom_url.
        
        Valid categories: 'world', 'politics', 'technology', 'cybersecurity', 'business', 'science', 'gaming'.
        
        Args:
            category: The topic to search for (use one from the valid list above).
            custom_url: A specific RSS feed URL if the user asks for a site not in the categories.
        """
        if not category and not custom_url:
            return "Error: You must provide either a 'category' or a 'custom_url'."

        def _fetch():
            # Import inside to dodge the module loader
            import feedparser
            import requests

            urls_to_fetch = []
            report_title = ""

            if custom_url:
                urls_to_fetch = [custom_url]
                report_title = f"CUSTOM FEED: {custom_url}"
            else:
                cat_lower = category.lower()
                if cat_lower not in self.FEED_DIRECTORY:
                    valid_cats = ", ".join(self.FEED_DIRECTORY.keys())
                    return f"Error: Invalid category '{category}'. Valid options are: {valid_cats}"
                
                urls_to_fetch = self.FEED_DIRECTORY[cat_lower]
                report_title = f"CATEGORY: {category.upper()}"

            report = f"NEWS BRIEFING | {report_title}\n"
            report += "=" * 50 + "\n\n"

            found_articles = 0

            for url in urls_to_fetch:
                try:
                    # Use requests with a timeout so broken feeds don't hang the bot
                    resp = requests.get(url, timeout=10)
                    feed = feedparser.parse(resp.content)
                    
                    source_name = feed.feed.get('title', url)
                    
                    # Grab the top 5 articles from each source
                    for entry in feed.entries[:5]:
                        title = entry.get('title', 'Untitled')
                        link = entry.get('link', 'No link')
                        published = entry.get('published', 'Unknown date')
                        
                        # Get summary and clean the HTML out of it
                        raw_summary = entry.get('summary', '')
                        summary = self._strip_html(raw_summary)
                        
                        report += f"SOURCE:   {source_name}\n"
                        report += f"TITLE:    {title}\n"
                        report += f"DATE:     {published}\n"
                        report += f"LINK:     {link}\n"
                        report += f"SUMMARY:  {summary[:400]}...\n"
                        report += "-" * 40 + "\n\n"
                        
                        found_articles += 1

                except Exception as e:
                    report += f"[Failed to fetch from {url}: {str(e)}]\n\n"

            if found_articles == 0:
                return "No articles could be retrieved at this time."

            return report

        return await asyncio.to_thread(_fetch)
