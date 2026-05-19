# 问题

1. openai api进行条目向量化时候出现限流：最多一次64条
    解决：用一个循环批次处理

2. 加入where条件搜索location字段，搜索不到
    原因：location字段一个字符串，里面逗号分隔地点，无法精确匹配，而where不支持contains
    解决：一起放在where_document里检索，$and连接

3. 对话幻觉，把快手的岗位直接给到rag里进行查询
    解决：目前使用提示词prompt告诉agent只有快手岗位，其他的没有要求用户提供url进行爬取

## 修改项目代码

4. 点击元素，最开始没给元素
    解决：关键词过滤可交互元素，限制10个

5. xhr爆炸问题
    解决：预评分，前8个给到llm

# 面试问题

## 第一条
1. 关键词评分验证机制说一下，得分≥3 这个阈值是怎么来的？
    答：关键词匹配（location、title...）score += 1
        存在 list[dict] 结构 score += 2
        score >= 3 则验证通过
        
2. playwright 相关问题

## 第二条
3. 与单 agent 相比优势？
    答：节省token、任务不混叠效果好
        可维护性，出错可以定位
        可以复用、后两部之后可以重复使用在别的任务
        并发扩展
4. ChromaDB 相关问题

5. 幂等去重？
    答：岗位id做md5、相同md5一个岗位内容hash有变动则更新；脏数据用id集合差集，超过n次没有爬到则从数据库删除

6. 提取内容

## 第三条
7. query understanding？
    答：提取location、keywords、精简后问句 + few shot提示词

8. RAG 修改
    答：最初方案：meta关键词过滤 + 向量检索
        修改方案：hybrid retrieve（bm25 + 向量检索） + rerank（向量相似度bi-encoder，rerank用cross-encoder，精度高（每个词都cross））

