import requests
import json

print("\n" + "="*60)
print("🎯 Testing Predict Next Day Endpoint")
print("="*60 + "\n")

url = "http://localhost:8000/predict_next_day"
data = {
    "symbol": "KOTAKBANK",
    "start_date": "2025-04-27",
    "end_date": "2025-10-24"
}

print(f"Request: {json.dumps(data, indent=2)}\n")
print("Sending request...")

try:
    response = requests.post(url, json=data, timeout=30)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n✅ SUCCESS!\n")
        print("="*60)
        print(f"Symbol:           {result['symbol']}")
        print(f"Predicted Close:  ₹{result['predicted_close']}")
        print(f"Last Close:       ₹{result['last_close']}")
        
        change = result['predicted_close'] - result['last_close']
        pct = (change / result['last_close']) * 100
        arrow = "↑" if change > 0 else "↓"
        print(f"Change:           {arrow} ₹{abs(change):.2f} ({pct:+.2f}%)")
        print(f"Date Predicted:   {result['date_predicted_for']}")
        
        print(f"\nMetrics:")
        print(f"  RMSE:              {result['rmse']}")
        print(f"  MAPE:              {result['mape']}%")
        print(f"  R²:                {result['r2']}")
        print(f"  Dir. Accuracy:     {result['directional_accuracy']}%")
        print(f"  Recent Data Pts:   {len(result['recent_data'])}")
        print("="*60)
        
        print("\n🎉 The endpoint is working!")
        print("✨ Refresh your browser and try the prediction module again.\n")
    else:
        print(f"\n❌ Error Response:")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("\n❌ Connection Error: Backend server is not running")
    print("Please start the backend server first.")
except Exception as e:
    print(f"\n❌ Error: {e}")

