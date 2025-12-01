# Fix: "Not Found" Error on Train & Predict

## 🔍 Problem
The `/train_and_predict` endpoint returns 404 "Not Found" because the backend server needs to be restarted to load the new endpoint.

## ✅ Solution

### **Step 1: Restart Backend Server**

1. **Stop the current backend server:**
   - Find the command window running `python api_server.py`
   - Press `Ctrl+C` to stop it

2. **Restart the backend:**
   ```bash
   python api_server.py
   ```
   
   Or use the batch file:
   ```bash
   start_backend.bat
   ```

3. **Verify the endpoint is available:**
   - Open: http://localhost:8000/docs
   - Look for `/train_and_predict` in the list of endpoints
   - It should show as `POST /train_and_predict`

### **Step 2: Test the Endpoint**

After restarting, the endpoint should work. You can test it:
- From the frontend: Click "Train & Predict" with "Fetch new data & train" checked
- Or directly: http://localhost:8000/docs → Try it out on `/train_and_predict`

---

## 🚀 Quick Restart (Both Servers)

If you want to restart both servers:

1. **Stop both servers** (Ctrl+C in each window)

2. **Restart using batch file:**
   ```bash
   start_servers.bat
   ```

This will start both backend and frontend in separate windows.

---

## 📝 Why This Happened

The `/train_and_predict` endpoint was added to `api_server.py` after the server was already running. FastAPI only loads routes when the server starts, so a restart is needed.

---

## ✅ After Restart

Once restarted, the "Train & Predict" button with "Fetch new data & train" checked should:
1. ✅ Connect to `/train_and_predict` endpoint
2. ✅ Show progress indicator
3. ✅ Fetch new data
4. ✅ Train model
5. ✅ Display prediction results

---

## 🔧 Alternative: Use Quick Prediction

If you don't want to restart right now, you can:
- **Uncheck** "Fetch new data & train" checkbox
- Use the existing `/predict_next_day` endpoint (works with current server)
- This uses pre-trained model and existing data (faster, ~5-10 seconds)

