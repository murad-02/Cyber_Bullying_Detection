"""
Simple test script to verify the Flask web application functionality.
"""

import requests
import json

def test_prediction():
    """Test the prediction endpoint"""
    
    # Test cases
    test_cases = [
        {
            "text": "হালের নতুন সেনসেশন বা স্টাইল হলো নাস্তিকতা। আপনি আপনার নাস্তিকবাদ যত বেশি জাহির করবেন আপনি তত জ্ঞানী মানুষ।",
            "description": "Bangla religious content"
        },
        {
            "text": "You are such a nice person! I really appreciate your help.",
            "description": "Positive English content"
        },
        {
            "text": "I hate you so much, you should just die already!",
            "description": "Threatening English content"
        },
        {
            "text": "This is a religious discussion that should be respectful.",
            "description": "Neutral religious discussion"
        }
    ]
    
    print("Testing Cyberbullying Detection Web Application")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Input: {test_case['text'][:50]}...")
        
        try:
            # Make prediction request
            response = requests.post(
                'http://localhost:5000/predict',
                json={'text': test_case['text']},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Model Used: {result['model_used']}")
                
                # Show top 2 probabilities
                probs = result['all_probabilities']
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                print(f"   Top probabilities:")
                for label, prob in sorted_probs[:2]:
                    print(f"     {label}: {prob:.3f}")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the Flask app is running on http://localhost:5000")
            break
        except requests.exceptions.Timeout:
            print("❌ Timeout: Request took too long")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

def test_home_page():
    """Test the home page"""
    try:
        response = requests.get('http://localhost:5000/', timeout=10)
        if response.status_code == 200:
            print("✅ Home page is accessible")
        else:
            print(f"❌ Home page error: {response.status_code}")
    except Exception as e:
        print(f"❌ Home page test failed: {e}")

def test_about_page():
    """Test the about page"""
    try:
        response = requests.get('http://localhost:5000/about', timeout=10)
        if response.status_code == 200:
            print("✅ About page is accessible")
        else:
            print(f"❌ About page error: {response.status_code}")
    except Exception as e:
        print(f"❌ About page test failed: {e}")

if __name__ == "__main__":
    print("Starting tests...")
    print("Make sure the Flask app is running with: python app.py")
    print()
    
    # Test pages
    test_home_page()
    test_about_page()
    
    # Test predictions
    test_prediction()
