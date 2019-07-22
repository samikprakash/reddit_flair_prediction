from flask import Flask,render_template,request
# import requests
import predictor
import json

app = Flask(__name__)

@app.route("/")

def home():
    return render_template("home.html")

@app.route("/about/")
def about():
    return "about content goes there"

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
        # result = list(request.form)
    # print((list(request.form))
    #   result = request.form
        result = (request.form['Flair'])
        result = str(result)
        print(result)
        # for i in request.form:
        #     print(request.form[i])
        # print(   result)
        final_result = predictor.predict(result)
    #   print("test1")
        return render_template("result.html",result = final_result)

if __name__ == "__main__":
    app.run(debug=True)
