#!/usr/bin/env python3
"""
Test script to verify Gemini API integration
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_gemini_api():
    """Test the Gemini API call directly"""
    try:
        import httpx
        
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        if not GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY not found in environment")
            return False
            
        print(f"✅ API Key loaded (length: {len(GEMINI_API_KEY)})")
        
        # Test API endpoint
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        # Simple test prompt
        payload = {
            "contents": [{
                "parts": [{
                    "text": "Generate a brief test response saying 'API connection successful'"
                }]
            }]
        }
        
        print("🔄 Testing Gemini API connection...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(gemini_api_url, json=payload)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                print(f"✅ API Response: {text}")
                return True
            else:
                print(f"❌ API Error: {response.text}")
                return False
                
    except ImportError:
        print("❌ httpx not installed. Run: pip install httpx")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_gemini_api())
    if result:
        print("\n🎉 Gemini API integration is working correctly!")
    else:
        print("\n💥 Gemini API integration has issues.")
