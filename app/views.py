from app import app
#@app.route('/index')

@app.route('/')
def index():
    
    return '''
<html>
  <head>
    <title>Intel Mug</title>
  </head>
  <body>
    <h1>Кружки</h1>
  </body>
</html>
'''