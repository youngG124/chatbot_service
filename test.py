from flask import Flask, jsonify
from flask_cors import CORS

from answer_function import predict

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

def makeAnswer(question) :
    return 'hello'

@app.route('/')
def default() :
    return "답변 생성 서버입니다."

# 질문에 대한 답변 return 하는 api
@app.route('/<question>')
def exportAnswer(question):
    print(question)
    return jsonify({"answer" : makeAnswer(question)})

if __name__ == '__main__' :
    app.run(host="127.0.0.1", port=5000)