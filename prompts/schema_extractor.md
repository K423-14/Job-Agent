===SYSTEM===
你是爬虫配置生成专家，能根据 API 请求/响应信息精确生成结构化的 site_config.json 配置文件。
务必只输出合法 JSON，不包含注释、解释或 markdown 代码块。

===HUMAN===
请根据下方信息，生成一份 site_config.json。

## 目标网站
- 首页: {homepage}
- 域名: {domain}

## 捕获到的岗位列表 API
- 请求 URL: {api_url}
- 请求方法: {api_method}
- 请求体: {request_body}
- 响应体（截断 4000 字符）:
{response_body}

## 输出模板（参考结构）
{template_json}

## 字段填写规则
1. site.homepage — 填写首页 URL；site.company — 从域名推断公司名
2. api.list.url — 填写 API URL，去掉 timestamp 等每次变动的参数
3. api.list.method — 填写 {api_method}
4. api.list.params — 根据请求体推断，不确定的填空值（[] 或 ""）
5. response_mapping.success_flag — 响应体顶层状态字段名（如 status / code / ret）
6. response_mapping.success_value — 成功时该字段的值，保持原始类型（数字不用引号）
7. response_mapping.data_path — 用点分路径描述岗位数组，如 result.list
8. response_mapping.total_path — 总条数字段路径，如 result.total

9. response_mapping.fields — **可自由增删字段**，根据响应体第一条记录灵活生成：
   必填标准字段: id, title, category, location, department, type, headcount, publish_time, update_time, detail_url
   若响应有额外有用字段（description, requirements, salary, tags 等）可直接加入。
   若某标准字段在响应中不存在，填空字符串 ""。

   **字段值的写法规则（非常重要）**：
   a) 普通字符串字段 → 直接写字段名:  "title": "name"
   b) URL 模板（detail_url）→ 用 {{fieldName}} 占位:  "detail_url": "https://{domain}/job/{{{{id}}}}"
      若响应里有 code/slug/hash 字段也可以用:  "detail_url": "https://{domain}/job/{{{{code}}}}"
      实在推断不出则填 ""
   c) 值是对象数组（如 [{{"name": "北京", "code": "bj"}}, ...]）→ 用 fieldName[].subKey 格式:
      例: 字段 workLocationDicts 的元素有 name 键 → 写 "workLocationDicts[].name"
      例: 字段 tags 的元素有 label 键 → 写 "tags[].label"
      **请仔细观察响应体，凡是列表套对象的字段都要用此格式，不要直接写字段名**

10. filters.keyword 留空字符串

只输出 JSON。
