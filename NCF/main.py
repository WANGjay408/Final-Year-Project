import torch
import pandas as pd
import torch.nn as nn  # 引入 nn 模块
import numpy as np
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 数据加载和预处理
trainData = pd.read_csv('ml100k.train.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
testData = pd.read_csv('ml100k.test.rating', header=None, names=['user', 'item', 'rate'], sep='\t')

user2idx = {user: idx for idx, user in enumerate(trainData.user.unique())}
item2idx = {item: idx for idx, item in enumerate(trainData.item.unique())}

trainData['user_idx'] = trainData['user'].map(user2idx)
trainData['item_idx'] = trainData['item'].map(item2idx)
testData['user_idx'] = testData['user'].map(user2idx)
testData['item_idx'] = testData['item'].map(item2idx)

num_users = len(user2idx)
num_items = len(item2idx)
K = 20
lambd = 0.00001
learning_rate = 1e-3

# 数据转换为PyTorch张量
userIdx = torch.tensor(trainData.user_idx.values, dtype=torch.long)
itemIdx = torch.tensor(trainData.item_idx.values, dtype=torch.long)
ratings = torch.tensor(trainData.rate.values, dtype=torch.float32)

# 创建训练数据集和数据加载器
train_dataset = TensorDataset(userIdx, itemIdx, ratings)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# NCF模型定义
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super(NCF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_idx, item_idx):
        user_embeds = self.user_embedding(user_idx)
        item_embeds = self.item_embedding(item_idx)

        concat_embeds = torch.cat((user_embeds, item_embeds), dim=1)

        output = self.fc_layers(concat_embeds)

        return output


# 创建NCF模型和优化器
ncf_model = NCF(num_users, num_items, K, 64)
optimizer = torch.optim.SGD(ncf_model.parameters(), lr=learning_rate, weight_decay=lambd)
criterion = nn.MSELoss()

# 训练模型
print('Training')
for epoch in range(10):
    total_loss = 0.0
    for user_batch, item_batch, rating_batch in train_loader:
        optimizer.zero_grad()
        outputs = ncf_model(user_batch, item_batch)
        loss = criterion(outputs, rating_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('Epoch:', epoch + 1, 'Loss:', total_loss)


# 测试模型
def getRanking():
    testData_cleaned = testData.dropna()
    ranking = []
    for i in range(len(testData_cleaned)):
        user_idx = int(testData_cleaned.iloc[i]['user_idx'])
        item_idx = int(testData_cleaned.iloc[i]['item_idx'])

        assert user_idx < num_users, "Invalid user index"
        assert item_idx < num_items, "Invalid item index"

        user_idx_tensor = torch.tensor([user_idx], dtype=torch.long)
        item_idx_tensor = torch.tensor([item_idx], dtype=torch.long)
        pred = ncf_model(user_idx_tensor, item_idx_tensor)
        ranking.append((user_idx, item_idx, pred.item()))

    return pd.DataFrame(ranking, columns=['user_idx', 'item_idx', 'pred'])


print('Testing')
ranking = getRanking()
print(ranking.head())
