# MCP
三层：host、client（通讯模块）、server
能力：tool、resource、prompt
传输协议：json-rpc 2.0 stdio + streamable-http

# function calling
如何训练：STF(训练如何输出json调用工具)；RLHF(人工标注哪里调用哪里不调用)

# Skills
md文件 + 文件夹内容（脚本/资源）
使用：渐进式  阅读所有skills的name + description -> 判断使用到哪个，阅读全文md -> 如果需要使用其中的脚本/资源，再去调用

# RAG
1. chunk分割：固定长度、语义分割（段落->句子->分割标点；，切）、特殊内容（代码段、整个表格）、父子分割（分割小块、检索大段）、late chunk（先全局编码再按chunk分割）

2. embedding：word2vec(处理不了多义词)->bert(必须两个向量拼接后算相似度)->BGE(可以离线算向量，加快速度)

3. 向量数据库索引：HNSW(地图索引，消耗内存大；召回率高速度快；chromadb默认)/IVF(先桶筛选再找，占用内存小)

