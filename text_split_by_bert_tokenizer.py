from transformers import BertTokenizer

# 加载中文BERT分词器（以 BERT-base Chinese 为例）
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 输入中文文本
text = ("紧急通知：家里有孩子的大人都看好了江苏省来东北1000多外地人专偷小孩抢小孩的苏州已丢20多个已解剖7"
        "个拿走器官！今天学校也给家长开会呢说凡是街上转悠跟到家门口楼下就走了面包车收粮食的车收旧家电的人带黑口罩，穿黑裤子有问路的千万别停下不要理会让更多的人知道，转一次可能就拯救了一个孩子")

# 使用tokenizer进行分词（token级别）
tokens = tokenizer.tokenize(text)

print("分词结果：", tokens)
