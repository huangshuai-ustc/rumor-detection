import torch
from transformers import BertTokenizer
from english_news_classify import BertGRUClassifier  # 你定义的模型类
import os

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


def predict(text):
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
    sample_text = "The economy is showing signs of strong recovery."
    pred, conf = predict(sample_text)
    print(f"预测标签：{pred}, 置信度：{conf:.4f}")
