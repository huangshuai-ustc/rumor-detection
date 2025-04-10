from transformers import BertForSequenceClassification
import torch
from english_news_classify import BertGRUClassifier
from transformers import BertTokenizer
import warnings
warnings.filterwarnings("ignore")


def bert_predict_zh(text, model_path='../bert/chinese_news_classifier',
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 文本编码
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, confidence


def bert_predict_en(text, model_path='../bert/english_news_classifier',
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 文本编码
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, confidence


def load_model(model_weights_path, tokenizer_path, device):
    """
    加载自定义 BERT-GRU 模型和分词器。
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertGRUClassifier(bert_model_path=tokenizer_path)  # 用原始bert模型路径
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer


def rnn_predict_zh(text):
    """
    使用训练好的模型对单条文本进行预测。
    """
    model_weights = "../transformer-rnn/chinese_news_classifier/model.pth"
    pretrained_bert_path = "../pretrained_models/bert-base-chinese"  # ✅ 正确路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(model_weights, pretrained_bert_path, device)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return 1-pred, confidence


def rnn_predict_en(text):
    """
    使用训练好的模型对单条文本进行预测。
    """
    model_weights = "../transformer-rnn/english_news_classifier/model.pth"
    pretrained_bert_path = "../pretrained_models/bert-base-uncased"  # ✅ 正确路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(model_weights, pretrained_bert_path, device)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, confidence


if __name__ == "__main__":
    text = "总统举行了关于经济改革的新闻发布会。"
    label, score = bert_predict_en(text)
    print(f"预测结果: {label}，置信度: {score:.4f}")
    label, score = bert_predict_zh(text)
    print(f"预测结果: {label}，置信度: {score:.4f}")
    label, score = rnn_predict_zh(text)
    print(f"预测结果: {label}，置信度: {score:.4f}")
    label, score = rnn_predict_en(text)
    print(f"预测结果: {label}，置信度: {score:.4f}")
