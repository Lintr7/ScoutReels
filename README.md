# Welcome to Scout! 📈📊

## What is Scout? 🤔

Scout is an AI-powered stock platform that allows you to view stocks in an Instagram reels format, providing the latest news, price movement, financial metrics, and LLM-powered sentiment analysis on popular stocks.

## Try It Out Now! 🤩

Experience Scout live in action (best viewed on a laptop):  
[https://scout-reels.vercel.app](https://scout-reels.vercel.app)

## Tech Stack 🛠️

**Frontend:**
- React.js
- AceternityUI

**Backend:**
- FastAPI
- Supabase (For database and authentication)
- Railway (Deployment), Used AWS Elastic Beanstalk before
- Python (for web scraping and OpenAI sentiment analysis)
- Marketaux, Finnhub, and Alpaca API calls

## Challenges Faced 😰

During development, I encountered several challenges:

- **Responsive Design:** Ensuring the React components scaled and displayed correctly across different screen sizes required many adjustments and testing.
- **AWS Elastic Beanstalk:** Configuring and restructuring files to meet AWS deployment requirements was definitely complicated.
- **API Communication:** Initially, our API endpoint calls failed due to mixed content issues (HTTPS on the frontend vs. HTTP on the backend). I resolved this by implementing a secure proxy to align both endpoints under HTTPS. Currently, Scout is deployed Railway due to its ease of use.
- **API Optimation:** Looking for the best API providers with generous free tiers that gave all the information we wanted wasn't easy. I also implemented an in-memory cache to drastically reduce API calls.
