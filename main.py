import torch
import pandas as pd
from torch.autograd import Variable

trainData = pd.read_csv('ml100k.train.rating', header=None, names=['user', 'item', 'rate'], sep='\t')
testData = pd.read_csv('ml100k.test.rating', header=None, names=['user', 'item', 'rate'], sep='\t')

# 创建userid和itemid的映射表
user2idx = {user: idx for idx, user in enumerate(trainData.user.unique())}
item2idx = {item: idx for idx, item in enumerate(trainData.item.unique())}

# 将训练集和测试集中的userid和itemid替换为连续索引
trainData['user_idx'] = trainData['user'].map(user2idx)
trainData['item_idx'] = trainData['item'].map(item2idx)
testData['user_idx'] = testData['user'].map(user2idx)
testData['item_idx'] = testData['item'].map(item2idx)

userIdx = trainData.user_idx.values
itemIdx = trainData.item_idx.values

num_users = len(user2idx)
num_items = len(item2idx)
K = 20
lambd = 0.00001
learning_rate = 1e-3

U = torch.randn([num_users, K], requires_grad=True)
P = torch.randn([num_items, K], requires_grad=True)

optimizer = torch.optim.SGD([U, P], lr=learning_rate)

print('Training')
for epoch in range(10):
    total_loss = 0.0
    for i in range(len(trainData)):
        user_idx = trainData.iloc[i]['user_idx']
        item_idx = trainData.iloc[i]['item_idx']

        pos_pred = torch.dot(U[user_idx], P[item_idx])  # 正样本预测评分

        neg_item_idx = torch.randint(num_items, size=(1,)).item()  # 随机选择一个负样本
        while neg_item_idx in trainData[trainData['user_idx'] == user_idx]['item_idx'].values:
            neg_item_idx = torch.randint(num_items, size=(1,)).item()  # 确保负样本不在用户的已有物品中

        neg_pred = torch.dot(U[user_idx], P[neg_item_idx])  # 负样本预测评分

        loss = -torch.log(torch.sigmoid(pos_pred - neg_pred))  # BPR损失函数
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch:', epoch + 1, 'Loss:', total_loss)


def getRanking():
    testData_cleaned = testData.dropna()  # 清除包含NaN值的行
    ranking = []
    for i in range(len(testData_cleaned)):
        user_idx = int(testData_cleaned.iloc[i]['user_idx'])
        item_idx = int(testData_cleaned.iloc[i]['item_idx'])

        assert user_idx < num_users, "Invalid user index"
        assert item_idx < num_items, "Invalid item index"

        pred = torch.dot(U[user_idx], P[item_idx])
        ranking.append((user_idx, item_idx, pred.item()))

    return pd.DataFrame(ranking, columns=['user_idx', 'item_idx', 'pred'])
# def getRanking():
#     ranking = []
#     for i in range(len(testData)):
#         user_idx = int(testData.iloc[i]['user_idx'])
#         item_idx = int(testData.iloc[i]['item_idx'])
#
#         assert user_idx < num_users, "Invalid user index"
#         assert item_idx < num_items, "Invalid item index"
#
#         pred = torch.dot(U[user_idx], P[item_idx])
#         ranking.append((user_idx, item_idx, pred.item()))
#
#     return pd.DataFrame(ranking, columns=['user_idx', 'item_idx', 'pred'])


print('Testing')
ranking = getRanking()
print(ranking.head())

