# tito

## 1. Configure the ff. values

### a. filedrag.js --> call to flask
- HOST (default: http://127.0.0.1)
- PORT (default: 5000)

### b. assets/api/flask_clarifai.py --> flask config
- HOST (default: '127.0.0.1')
- PORT (default: '5000')

## 2. run flask_tito.py (dependencies installed in app1 server)
```
python -m http.server
python assets/api/flask_tito.py
```

## 3. open html file (index.html)