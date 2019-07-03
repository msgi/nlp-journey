from sanic import Sanic
from sanic.response import json


# 启动web
app = Sanic()
model = None

@app.route("/predict", methods=['POST'])
async def predict(request):
    """
    采用restful接口的形式,获取分类结果
    :param request: {
                        "sentence": "待推测文本"
                    }
    :return:
    """
    nlp = request.json.get('sentence')
    answer = model.predict(nlp)

    ans = answer[0]

    return json({'category':ans})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)
