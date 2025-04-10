from flask import Flask, render_template, request
import random
from predict import bert_predict_zh, bert_predict_en, rnn_predict_en, rnn_predict_zh

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    language = 'zh'
    results = []
    if request.method == 'POST':
        text = request.form['text']
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        # 这里可以添加你的检测逻辑，目前用示例结果代替
        language = request.form.get('language', 'zh')  # 默认中文
        if language == 'zh':
            for i, paragraph in enumerate(paragraphs):
                # 使用BERT模型进行预测
                pred1, confidence1 = bert_predict_zh(paragraph)
                pred2, confidence2 = rnn_predict_zh(paragraph)
                results.append({
                    'index': i + 1,
                    'text': paragraph[:30],
                    'confidence1': round(confidence1, 2),
                    'confidence2': round(confidence2, 2),
                    'result1': '真新闻' if pred1 == 1 else '假新闻',
                    'result2': '真新闻' if pred2 == 1 else '假新闻'
                })
        else:
            for i, paragraph in enumerate(paragraphs):
                # 使用Transformer-RNN模型进行预测
                pred1, confidence1 = bert_predict_en(paragraph)
                pred2, confidence2 = rnn_predict_en(paragraph)
                results.append({
                    'index': i + 1,
                    'text': paragraph[:30],
                    'confidence1': round(confidence1, 2),
                    'confidence2': round(confidence2, 2),
                    'result1': '真新闻' if pred1 == 1 else '假新闻',
                    'result2': '真新闻' if pred2 == 1 else '假新闻'
                })
    return render_template('index.html', results=results, language=language)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
