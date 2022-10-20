from flask import Flask, render_template, request, url_for
import pickle
import os

clf = pickle.load(open('model.pkl', 'rb'))

PORT = os.getenv('CDSW_APP_PORT', '8090')

app = Flask(__name__)

@app.route("/", methods=["POST","GET"])
def main():
  if request.method == "POST":
    form_data = request.form
    x = form_data["xValue"]
    y = form_data["yValue"]
    z = form_data["zValue"]
    pred = clf.predict([[x,y,z]])
    return render_template('result.html', prediction=pred[0])
  
  return render_template('main.html')
    
if __name__ == '__main__':
  app.run(host='127.0.0.1', port=PORT)
