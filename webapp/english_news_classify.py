import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class BertGRUClassifier(nn.Module):
    def __init__(self, bert_model_path, hidden_dim=128, num_labels=2, dropout=0.5):
        """
        减少了GRU的隐藏维度，并增加了dropout，简化模型结构。
        """
        super(BertGRUClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        # 将GRU的hidden_dim从256减少到128，并设置dropout较高的值
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,  # 保留1层GRU，避免过拟合
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)  # 提高dropout值
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)  # GRU是双向的，所以hidden_dim*2

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_outputs.last_hidden_state
        
        # 解决非连续tensor问题
        sequence_output = sequence_output.contiguous()

        gru_output, _ = self.gru(sequence_output)
        pooled_output = gru_output[:, -1, :]  # 取GRU输出序列中的最后一个状态
        logits = self.classifier(self.dropout(pooled_output))  # 通过dropout层

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return logits, loss

# 数据加载和预处理
def load_data(file_path):
    """
    从文件中加载数据并进行处理。
    """
    df = pd.read_excel(file_path)
    df = df[['text', 'label']].dropna()  # 保留 'text' 和 'label' 列并去除空值

    tokenizer = BertTokenizer.from_pretrained('../pretrained_models/bert-base-uncased')

    # 将文本转换为 BERT 输入格式
    encodings = tokenizer(list(df['text']), truncation=True, padding=True, max_length=512)

    # 划分数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        list(df['text']),
        list(df['label']),
        test_size=0.2,
        random_state=42
    )

    # 对文本进行编码
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    # 转换为 Tensor
    train_inputs = torch.tensor(train_encodings['input_ids'])
    train_attention_mask = torch.tensor(train_encodings['attention_mask'])
    train_labels = torch.tensor(train_labels)

    val_inputs = torch.tensor(val_encodings['input_ids'])
    val_attention_mask = torch.tensor(val_encodings['attention_mask'])
    val_labels = torch.tensor(val_labels)

    # 创建 DataLoader
    train_dataset = TensorDataset(train_inputs, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_inputs, val_attention_mask, val_labels)

    return train_dataset, val_dataset

# 训练模型
def train_model(model, train_dataset, val_dataset, batch_size=16, epochs=3, learning_rate=3e-5, device='cuda'):
    """
    训练 BERT + GRU 模型。
    """
    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # 打开文件用于写入评价指标
    with open("english_metrics_log.txt", "a") as metrics_file:
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0

            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                optimizer.zero_grad()

                logits, loss = model(input_ids, attention_mask, labels=labels)
                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            # 验证
            model.eval()
            total_val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}"):
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]

                    logits, loss = model(input_ids, attention_mask, labels=labels)
                    total_val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

            # 打印并保存指标
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Precision: {precision:.4f}")
            print(f"Validation Recall: {recall:.4f}")
            print(f"Validation F1: {f1:.4f}")

            # 将指标写入到文本文件
            metrics_file.write(f"Epoch {epoch + 1}:\n")
            metrics_file.write(f"Validation Loss: {avg_val_loss:.4f}\n")
            metrics_file.write(f"Accuracy: {val_accuracy:.4f}\n")
            metrics_file.write(f"Precision: {precision:.4f}\n")
            metrics_file.write(f"Recall: {recall:.4f}\n")
            metrics_file.write(f"F1: {f1:.4f}\n")
            metrics_file.write("\n")

# 保存模型
def save_model(model, tokenizer, save_directory):
    """
    手动保存模型权重和分词器到本地。
    """
    # 保存模型权重
    torch.save(model.state_dict(), f"{save_directory}/model.pth")
    # 保存分词器
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

# 加载模型
def load_model(model, tokenizer, save_directory, device):
    """
    加载模型权重和分词器。
    """
    model.load_state_dict(torch.load(f"{save_directory}/model.pth", map_location=device))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(save_directory)
    return model, tokenizer

# 主函数
if __name__ == "__main__":
    # 数据文件路径
    file_path = '../english_news.xlsx'  # 替换为你的数据文件路径

    # 加载数据
    train_dataset, val_dataset = load_data(file_path)

    # 初始化模型和分词器
    model = BertGRUClassifier(bert_model_path='../pretrained_models/bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('../pretrained_models/bert-base-uncased')

    # 训练模型
    train_model(model, train_dataset, val_dataset, batch_size=16, 
                epochs=3, learning_rate=3e-5, device='cuda' if torch.cuda.is_available() else 'cpu')

    # 保存训练好的模型
    save_model(model, tokenizer, './english_news_classifier')
