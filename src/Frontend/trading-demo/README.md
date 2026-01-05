## Frontend Demo Dashboard

### Run
npm install
npm start

### Backend URL
Create a .env file in the project root:
REACT_APP_API_BASE=http://localhost:8000

### Expected endpoints
GET /models
GET /prices?symbol=BTCUSD&tf=1D
GET /predict?symbol=BTCUSD&tf=1D&model=tft

### Data formats expected
prices:
{ candles: [{ time, open, high, low, close, volume }] }

prediction:
{ prediction: [{ time, value }] }
