# F027_TransE_kg_music  TransE知识图谱音乐推荐系统基于路径推荐算法+vue+flask+知识图谱可视化+协同过滤推荐算法

> 完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从git来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775
关注B站，有好处！
> 

**编号: F027 TransE版本 （知识图谱路径推荐） **

架构: vue+flask+neo4j+mysql
亮点：TransE 基于路径的知识图谱推荐 + 协同过滤推荐算法+知识图谱可视化
支持爬取音乐数据，数据超过3万条，知识图谱节点几万个
## 模块图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8bf4bc9e148d4128918d86c5c6e58250.png)
## 架构说明
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28246030da9e4dc3a649e64d1125dce5.png)

系统架构主要分为以下几个部分：**用户前端**、**后端服务**、**数据库**、**数据爬取与处理**。各部分通过协调工作，实现数据的采集、存储、处理以及展示。具体如下：
### 1. 用户前端
**用户**通过浏览器访问系统，前端采用了基于 Vue.js 的技术栈来构建。
- **浏览器**：作为用户与系统交互的媒介，用户通过浏览器进行各种操作，如浏览图书、获取推荐等。
- **Vue 前端**：使用 Vue.js 框架搭建前端界面，包含 HTML、CSS、JavaScript，以及 Vuex（用于状态管理），vue-router（用于路由管理），和 Echarts（用于数据可视化）等组件。前端向后端发送请求并接收响应，展示处理后的数据。
### 2. 后端服务
后端服务采用 Flask 框架，负责处理前端请求，执行业务逻辑，并与数据库进行交互。
- **Flask 后端**：使用 Python 编写，借助 Flask 框架处理 HTTP 请求。通过 SQLAlchemy 与 MySQL 进行交互，通过 py2neo 与 Neo4j 进行交互。后端主要负责业务逻辑处理、 数据查询、数据分析以及推荐算法的实现。
### 3. 数据库
系统使用了两种数据库：关系型数据库 MySQL 和图数据库 Neo4j。
- **MySQL**：存储从网络爬取的基本数据。数据爬取程序从外部数据源获取数据，并将其存储在 MySQL 中。MySQL 主要用于存储和管理结构化数据。
- **Neo4j**：存储图谱数据，特别是歌手、歌曲及其关系。通过利用 py2neo 库将 MySQL 中的数据结构化为图节点和关系，再通过图谱生成程序（可能是一个 Python 脚本）将其导入到 Neo4j 中。
### 4. 数据爬取与处理

数据通过爬虫从外部数据源获取，并存储在 MySQL 数据库中，然后将数据转换为图结构并存储在 Neo4j 中。
- **爬虫**：实现数据采集，从网络数据源抓取相关信息。爬取的数据首先存储在 MySQL 数据库中。
- **图谱生成程序**：利用 py2neo 将爬取到的结构化数据（如歌曲、歌手，以及它们之间的关系）从 MySQL 导入到 Neo4j 中。通过构建图谱数据，使得后端能够进行复杂的图查询和推荐计算。
### **工作流程**

1. **数据爬取**：爬虫程序从外部数据源抓取数据并存储到 MySQL 数据库中。
2. **数据处理与导入**：图谱生成程序将 MySQL 中的数据转换为图结构并导入到 Neo4j 中，利用 py2neo 与 Neo4j 交互。
3. **前后端交互**：
    - 用户通过浏览器访问系统，前端用 Vue.js 构建，提供友好的用户界面和交互。
    - 前端向 Flask 后端发送请求，获取歌曲信息或推荐歌曲。
4. **推荐算法**：后端在接收请求后，利用 Neo4j 图数据库中的数据和关系进行处理（如推荐计算），并使用 py2neo 库与 Neo4j 交互获取数据结果。
5. **数据返回与展示**：后端将计算结果返回给前端进行展示，通过 Vue.js 的图表库（如 Echarts）进行数据可视化，让用户得到直观的推荐结果和分析信息。
### **小结**

这套系统通过整合爬虫、关系型数据库、图数据库，以及前后端的协调配合，实现了数据的高效采集、存储、处理、推荐和展示。从用户体验的角度，系统能够提供高度个性化的推荐，并通过图形化的方式呈现数据分析结果。
## 功能介绍
### 0 图谱构建
利用python读取数据并且构建图谱到neo4j中：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d149c2186bb24f66ba97387f1b3f6b4a.png)
### 1 系统主页，统计页面
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bad3f552dcd3486399ce26673fcc8f13.png)
动态统计卡片
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/54adc702908e4efa92f4ccaff79ea744.png)
### 2 知识图谱
支持可视化
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dfe2aa51aaa44123804890fe766f80e0.png)
支持模糊搜索，比如搜索特定关键词【林俊杰】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9d4dd93bca604c34b76d91f0d764d556.png)
### 3 推荐算法
没有登录无法推荐，一共三种推荐算法，第一种是transe的知识图谱路径推荐算法
第一种是基于知识图谱的transe推荐算法，训练过程：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2de2f0359eac4de3abfc0feaffe38e45.png)
界面效果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2de8be5c069e4b38bda06303663dd83a.png)

**两种协同过滤推荐算法推荐**
第二种推荐算法
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cac54356a8ac4a72a0dcb53918235411.png)
第三种推荐算法
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f083dd35e9be41e98cd981eac9ca9f6e.png)
点击可以播放可以播放歌曲
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5eaf155ab3444f51bcd6f53805008d8c.png)
### 4 可视化分析
分为4个页面
歌手分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1bd049b908694b9cb2f7deed38a7f5ba.png)
专辑、热评分析等
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cbb06dfae4b847aea1479b6df8f40f9f.png)
歌词词云分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4ebe966e13db4f50993716303907599e.png)
评论词云分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1a84c0467a874a5880f6136bea2669fe.png)
### 5 数据查询
数据关键词可以进行歌曲的查询：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43065ff24a9544cb88187c92289a61a0.png)
## 算法部分
### 代码介绍
该功能利用TransE模型从知识图谱中学习音乐实体的向量表示，计算用户与音乐、艺术家、风格的相似度，以此生成个性化推荐列表。

实现细节
数据预处理：读取音乐知识图谱数据，构建实体和关系映射。
TransE模型训练：通过优化模型参数，学习实体和关系的向量表示。
推荐生成：基于用户历史行为，计算相似度并生成推荐列表。
评估：通过精确率和召回率评估推荐效果。。
### 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7484dcb519ca4c408d8225e88e87439c.png)

### 核心代码
```python
import numpy as np
import pandas as pd

# 数据加载与预处理
def load_data():
    data = pd.read_csv("music_data.csv")
    entities = data["entities"].unique()
    relations = data["relations"].unique()
    return data, entities, relations

# TransE模型定义
class TransE:
    def __init__(self, entities, relations, dim=100):
        self.dim = dim
        self.entity_vec = {e: np.random.randn(dim) for e in entities}
        self.relation_vec = {r: np.random.randn(dim) for r in relations}

    def train(self, data, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            for _, row in data.iterrows():
                h, r, t = row["head"], row["relation"], row["tail"]
                h_vec = self.entity_vec[h]
                r_vec = self.relation_vec[r]
                t_vec = self.entity_vec[t]
                loss = np.linalg.norm(h_vec + r_vec - t_vec)
                # 梯度下降
                h_vec -= learning_rate * (h_vec + r_vec - t_vec)
                r_vec -= learning_rate * (h_vec + r_vec - t_vec)
                t_vec -= learning_rate * (h_vec + r_vec - t_vec)

    def get_recommendations(self, user_history, top_k=10):
        user_vec = np.mean([self.entity_vec[item] for item in user_history], axis=0)
        scores = []
        for song in self.entity_vec:
            score = np.dot(user_vec, self.entity_vec[song])
            scores.append((song, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


