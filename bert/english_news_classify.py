import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型和分词器本地路径配置
LOCAL_MODEL_PATH = "../pretrained_models/bert-base-uncased"  # 本地模型路径
TOKENIZER_PATH = "../pretrained_models/bert-base-uncased"  # 本地分词器路径


def smart_truncate(text, title=None, max_length=500):
    if title:
        combined = f"{title}. {text}"
    else:
        combined = text
    if len(combined.split()) <= max_length:
        return combined
    # 计算截断位置（按单词数）
    words = combined.split()
    head = int(max_length * 0.6)
    tail = max_length - head
    return ' '.join(words[:head] + words[-tail:])


# 1. 数据准备（从 Excel 文件读取）
def load_excel_data(file_path, test_size=0.2):
    df = pd.read_excel(file_path)
    # 只保留我们需要的列
    df = df[['text', 'label']].dropna()
    df['title'] = ""
    df['processed_text'] = df.apply(lambda x: smart_truncate(x.text, x.title), axis=1)

    # 使用 stratify 参数按 label 分层抽样
    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],  # 关键行：按label分层抽样
        random_state=42
    )

    return Dataset.from_pandas(train_df.reset_index(drop=True)), Dataset.from_pandas(valid_df.reset_index(drop=True))


# 加载数据集
train_dataset, valid_dataset = load_excel_data("../english_news.xlsx")

# 2. 从本地加载分词器
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)


# 预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples['processed_text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )


# 应用预处理
encoded_train = train_dataset.map(preprocess_function, batched=True)
encoded_valid = valid_dataset.map(preprocess_function, batched=True)

# 3. 从本地加载模型
model = BertForSequenceClassification.from_pretrained(
        LOCAL_MODEL_PATH,
        num_labels=2
    ).to(device)

# 4. 训练参数配置
batch_size = 16 if str(device) == 'cuda' else 8
training_args = TrainingArguments(
    output_dir='./results_en',
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    gradient_accumulation_steps=2 if str(device) == 'cpu' else 1,
    fp16=torch.cuda.is_available(),
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=3e-5,
    logging_dir='./logs_en',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 5. 评估指标
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    # 日志写入
    with open("english_metrics_log.txt", "a") as f:
        f.write(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# 6. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    compute_metrics=compute_metrics
)

# 7. 开始训练
print("Starting training...")
trainer.train()

# 8. 保存微调后的模型
model.save_pretrained("./english_news_classifier")
tokenizer.save_pretrained("./english_news_classifier")


# # 预测函数（处理长文本）
# def predict_long_text(text, title=None):
#     processed_text = smart_truncate(text, title)

#     inputs = tokenizer(
#         processed_text,
#         padding='max_length',
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     ).to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     return torch.argmax(probs).item()


# # 示例预测
# test_text = "This is a very long news article..."  # 长文本示例
# test_title = "Important News"
# print(f"Prediction result: {predict_long_text(test_text, test_title)}")
