from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
import openai
import os
from dotenv import load_dotenv
import httpx
from datetime import datetime, timedelta
import urllib.parse
import asyncio
from typing import Optional, Dict, Any
import json
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from validation import (
   validate_stocks_request,
   validate_news_request,
   validate_finnhub_request,
   validate_search_request,
   TimeoutMiddleware
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
   raise RuntimeError("OPENAI_API_KEY is not set in environment")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Redis setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Cache TTLs in seconds
CACHE_SECONDS = {
   "news": 24 * 3600,
   "finnhub": 24 * 3600,
   "search": 2 * 3600,
   "stocks": 360,
}

CACHE_TIMES = {k: v / 3600 for k, v in CACHE_SECONDS.items()}

def get_cached_data(key: str, cache_type: str):
   try:
       data = redis_client.get(key)
       if data:
           print(f"Cache HIT for {cache_type} key: {key}")
           return json.loads(data)
       print(f"Cache MISS for {cache_type} key: {key}")
       return None
   except Exception as e:
       print(f"Redis get error: {e}")
       return None

def set_cached_data(key: str, data, cache_type: str):
   try:
       ttl = CACHE_SECONDS.get(cache_type, 3600)
       redis_client.setex(key, ttl, json.dumps(data))
       print(f"Cache SET for {cache_type} key: {key} (TTL: {ttl}s)")
   except Exception as e:
       print(f"Redis set error: {e}")

app = FastAPI(
   title="News Sentiment API",
   max_request_size=1_000_000
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

FRONTEND_ORIGINS = [
   "https://scout-reels.vercel.app",
]

app.add_middleware(
   CORSMiddleware,
   allow_origins=FRONTEND_ORIGINS,
   allow_credentials=True,
   allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
   allow_headers=["*"],
)

app.add_middleware(TimeoutMiddleware)

class CompanyRequest(BaseModel):
   company: str


@app.post("/search")
@limiter.limit("20/minute")
@limiter.limit("300/day")
def handle_search(payload: CompanyRequest, request: Request):
   """POST /search - OpenAI sentiment analysis (2 hour cache)"""
   try:
       company = validate_search_request(request, payload.company)
      
       cache_key = f"search_{company.lower()}"
       cached_result = get_cached_data(cache_key, "search")
       if cached_result:
           return cached_result

       search_url = f"https://www.google.com/search?q={company}+news&tbm=nws"
       resp = requests.get(search_url, timeout=10.0)
       soup = BeautifulSoup(resp.content, "html.parser")
       headlines = [h3.get_text(strip=True) for h3 in soup.find_all("h3")][:10]

       if not headlines:
           raise HTTPException(status_code=404, detail="No news found")

       prompt = (
           f"Analyze the sentiment of the following news headlines about {company}'s stock. "
           "Provide a short summary of the sentiment and calculate the average sentiment score on a scale "
           "of 0 (negative) to 10 (positive). Return only the summary in bullet points with specific yet short & concise "
           "news examples and the average sentiment score as a number. Output should be in the exact format of: "
           "Average Sentiment Score: _/10. Then the summary."
       )

       try:
           ai_response = client.chat.completions.create(
               model="gpt-4o",
               messages=[
                   {"role": "system", "content": prompt},
                   {"role": "user", "content": ", ".join(headlines)},
               ],
           )
           sentiment_analysis = ai_response.choices[0].message.content.strip()
       except Exception as ai_error:
           print(f"OpenAI API error: {ai_error}")
           sentiment_analysis = "Unable to analyze sentiment due to API error."

       result = {
           "sentiment": sentiment_analysis,
           "headlines": headlines,
           "company": company,
       }

       set_cached_data(cache_key, result, "search")
       return result

   except HTTPException:
       raise
   except Exception as e:
       print(f"Error in /search: {type(e).__name__}: {str(e)}")
       raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/stocks/{symbol}")
@limiter.limit("150/minute")
async def get_stock_data(
   symbol: str,
   start: str,
   end: str,
   timeframe: str,
   request: Request
):
   """Alpaca stock data (6 minute cache)"""
   try:
       symbol, start, end, timeframe = validate_stocks_request(
           request, symbol, start, end, timeframe
       )
      
       cache_key = f"stocks_{symbol}_{start}_{end}_{timeframe}"
       cached_result = get_cached_data(cache_key, "stocks")
       if cached_result:
           return cached_result

       ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
       ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
      
       if not ALPACA_API_KEY or not ALPACA_API_SECRET:
           raise HTTPException(status_code=500, detail="Alpaca API credentials not configured")
      
       headers = {
           'APCA-API-KEY-ID': ALPACA_API_KEY,
           'APCA-API-SECRET-KEY': ALPACA_API_SECRET,
       }
      
       url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
       params = {
           'start': start,
           'end': end,
           'timeframe': timeframe,
           'feed': 'iex',
           'adjustment': 'all',
       }
      
       async with httpx.AsyncClient(timeout=10.0) as client_http:
           response = await client_http.get(url, headers=headers, params=params)
           response.raise_for_status()
           result = response.json()
           set_cached_data(cache_key, result, "stocks")
           return result
  
   except HTTPException:
       raise
   except httpx.TimeoutException:
       raise HTTPException(status_code=504, detail="External API timeout")
   except Exception as e:
       print(f"Error in /stocks: {type(e).__name__}")
       raise HTTPException(status_code=500, detail="Internal server error")


MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")


@app.get("/news/{symbol}")
@limiter.limit("120/minute")
async def get_news(
   request: Request,
   symbol: str,
   company_name: str = Query(..., alias="companyName"),
):
   """MarketAux news (24 hour cache)"""
   try:
       symbol, canonical_name = validate_news_request(request, symbol, company_name)
      
       cache_key = f"news_{symbol.lower()}_{canonical_name.lower()}"
       cached_result = get_cached_data(cache_key, "news")
       if cached_result:
           return cached_result

       thirty_days_ago = datetime.now() - timedelta(days=90)
       date_string = thirty_days_ago.strftime('%Y-%m-%d')
      
       params = {
           'api_token': MARKETAUX_API_KEY,
           'search': canonical_name,
           'limit': '50',
           'published_after': date_string,
           'sort': 'relevance',
           'sort_order': 'desc',
           'language': 'en',
           'domains': 'bloomberg.com,reuters.com,wsj.com,cnbc.com,marketwatch.com,finance.yahoo.com,forbes.com,businessinsider.com'
       }
      
       query_params = urllib.parse.urlencode(params)
       url = f"https://api.marketaux.com/v1/news/all?{query_params}"
      
       async with httpx.AsyncClient(timeout=20.0) as client_http:
           response = await client_http.get(url)
           response.raise_for_status()
           result = response.json()
           set_cached_data(cache_key, result, "news")
           return result
  
   except HTTPException:
       raise
   except httpx.TimeoutException:
       raise HTTPException(status_code=504, detail="External API timeout")
   except Exception as e:
       print(f"Error in /news: {type(e).__name__}")
       raise HTTPException(status_code=500, detail="Internal server error")


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

async def fetch_with_timeout(url: str, timeout: int = 10) -> Dict[Any, Any]:
   async with httpx.AsyncClient(timeout=timeout) as client_http:
       try:
           response = await client_http.get(url)
           response.raise_for_status()
           return response.json()
       except httpx.TimeoutException:
           raise HTTPException(status_code=408, detail="Request timeout")
       except httpx.HTTPStatusError as e:
           if e.response.status_code == 401:
               raise HTTPException(status_code=401, detail="Invalid API key")
           elif e.response.status_code == 403:
               raise HTTPException(status_code=403, detail="API access forbidden")
           elif e.response.status_code == 429:
               raise HTTPException(status_code=429, detail="Rate limit exceeded")
           else:
               raise HTTPException(status_code=e.response.status_code, detail=f"API error: {e.response.status_code}")
       except Exception as e:
           raise HTTPException(status_code=500, detail="External API error")


def safe_number(value: Any, fallback: float = 0.0) -> float:
   try:
       if value is None:
           return fallback
       num = float(value)
       return num if not (num != num) else fallback
   except (ValueError, TypeError):
       return fallback


def process_earnings_data(earnings_array: list) -> list:
   if not earnings_array:
       return []
  
   processed_earnings = []
   for i, earning in enumerate(earnings_array[:4]):
       try:
           period = earning.get('period')
           if period:
               period_date = datetime.fromisoformat(period.replace('Z', '+00:00'))
               quarter = f"Q{earning.get('quarter', i+1)} {earning.get('year', period_date.year)}"
           else:
               quarter = f"Q{i+1}"
          
           expected = safe_number(earning.get('estimate'), 0)
           actual = safe_number(earning.get('actual'), 0)
           surprise = safe_number(earning.get('surprisePercent'), 0)
          
           processed_earnings.append({
               "quarter": quarter,
               "expected": expected,
               "actual": actual,
               "surprise": round(surprise, 2),
               "period": period
           })
       except Exception as e:
           print(f"Error processing earning {i}: {e}")
           continue
  
   return list(reversed(processed_earnings))


def process_company_metrics(profile: dict, quote: dict, metrics: dict) -> dict:
   metrics_data = metrics.get('metric', {}) if metrics else {}
  
   return {
       "marketCap": safe_number(profile.get('marketCapitalization', 0) * 1000000, 0),
       "grossMargin": safe_number(metrics_data.get('grossMarginTTM'), 0),
       "logo": profile.get('logo', ''),
       "industry": profile.get('finnhubIndustry', ''),
       "peRatio": safe_number(metrics_data.get('peTTM'), 0),
       "volume10Day": safe_number(metrics_data.get('10DayAverageTradingVolume'), 0),
       "weekHigh52": safe_number(metrics_data.get('52WeekHigh'), 0),
       "weekLow52": safe_number(metrics_data.get('52WeekLow'), 0),
   }


@app.get("/finnhub/{symbol}")
async def get_finnhub_data(
   request: Request,
   symbol: str,
   company_name: Optional[str] = Query(None),
):
   """Finnhub earnings/metrics (24 hour cache)"""
   try:
       symbol, canonical_name = validate_finnhub_request(request, symbol, company_name)
      
       if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY":
           raise HTTPException(status_code=500, detail="Finnhub API key not configured")
      
       cache_key = f"finnhub_{symbol.lower()}_{canonical_name.lower()}"
       cached_result = get_cached_data(cache_key, "finnhub")
       if cached_result:
           return cached_result
      
       endpoints = {
           "earnings": f"{FINNHUB_BASE_URL}/stock/earnings?symbol={symbol}&token={FINNHUB_API_KEY}",
           "profile": f"{FINNHUB_BASE_URL}/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}",
           "metrics": f"{FINNHUB_BASE_URL}/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_API_KEY}"
       }
      
       tasks = [
           fetch_with_timeout(endpoints["earnings"]),
           fetch_with_timeout(endpoints["profile"]),
           fetch_with_timeout(endpoints["metrics"]),
       ]
      
       results = await asyncio.gather(*tasks, return_exceptions=True)
       earnings_data, profile_data, metrics_data = results
      
       raw_data = {}
       for key, result in zip(["earnings", "profile", "metrics"], results):
           if isinstance(result, Exception):
               raw_data[key] = {"error": str(result)}
               print(f"Error fetching {key}: {result}")
           else:
               raw_data[key] = result
      
       raw_data["quote"] = {"test": "disabled"}
      
       processed_earnings = []
       if not isinstance(earnings_data, Exception) and isinstance(earnings_data, list):
           if len(earnings_data) == 0:
               raise HTTPException(status_code=404, detail=f"No earnings data for {symbol}")
           processed_earnings = process_earnings_data(earnings_data)
       elif isinstance(earnings_data, Exception):
           raise HTTPException(status_code=500, detail="Failed to fetch earnings data")
      
       profile = profile_data if not isinstance(profile_data, Exception) else {}
       quote = {}
       metrics = metrics_data if not isinstance(metrics_data, Exception) else {}
      
       company_metrics = process_company_metrics(profile, quote, metrics)
      
       validation_warnings = []
       if not profile.get('marketCapitalization'):
           validation_warnings.append("No market cap data")
       if not metrics.get('metric'):
           validation_warnings.append("No metrics data")
      
       response_data = {
           "symbol": symbol,
           "company_name": canonical_name,
           "timestamp": datetime.utcnow().isoformat(),
           "earnings_data": processed_earnings,
           "company_metrics": company_metrics,
           "raw_data": raw_data,
           "validation_warnings": validation_warnings
       }

       arrayChecker = ["marketCap", "grossMargin", "peRatio", "volume10Day", "weekHigh52", "weekLow52"]
       count = sum(1 for s in arrayChecker if response_data["company_metrics"][s] == 0)
      
       earnings_fields = ["expected", "actual", "surprise"]
       missing_earnings_count = sum(
           1 for quarter in response_data["earnings_data"]
           if sum(1 for field in earnings_fields if quarter[field] == 0) == 3
       )

       if (len(response_data["company_metrics"]["logo"]) > 0 and
           len(response_data["company_metrics"]["industry"]) > 0 and
           count <= 3 and missing_earnings_count <= 2):
           set_cached_data(cache_key, response_data, "finnhub")
      
       return response_data
  
   except HTTPException:
       raise
   except Exception as e:
       print(f"Error in /finnhub: {type(e).__name__}")
       raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/cache/stats")
def get_cache_stats():
   try:
       info = redis_client.info()
       keys = redis_client.keys("*")
       return {
           "cache_times": CACHE_TIMES,
           "total_keys": len(keys),
           "redis_used_memory": info.get("used_memory_human"),
           "redis_connected_clients": info.get("connected_clients"),
           "keys": keys
       }
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Redis error: {e}")


@app.get("/cache/clear")
def clear_all_caches():
   try:
       redis_client.flushdb()
       return {"message": "All caches cleared"}
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Redis error: {e}")


@app.get("/test")
def test():
   return {"message": "FastAPI server is running!", "status": "OK"}