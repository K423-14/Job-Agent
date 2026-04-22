# 腾讯校招岗位爬虫 MVP

目标网站：https://join.qq.com/post.html

---

## 快速开始

```bash
pip install requests playwright rich
playwright install chromium
python scraper.py
```

---

## 文件说明

```
join_qq_scraper/
├── site_config.json   # 网站爬取配置（核心）
├── scraper.py         # 爬虫主程序
├── jobs.json          # 输出：完整数据
└── jobs.csv           # 输出：Excel 可打开
```

---

## site_config.json 核心字段说明

| 字段路径 | 说明 |
|---|---|
| `api.list.url` | 岗位列表的 API 地址 |
| `api.list.params` | 默认请求参数 |
| `api.list.pagination.page_param` | 翻页参数名 |
| `response_mapping.data_path` | 响应 JSON 中岗位数组的路径 |
| `response_mapping.total_path` | 响应 JSON 中总数的路径 |
| `response_mapping.fields` | 字段名映射（API字段 → 统一字段名） |
| `filters.keyword` | 关键词过滤（如 "java"） |

---

## 如何找到真实 API 地址

1. 打开 Chrome，进入 `https://join.qq.com/post.html`
2. F12 → Network → 勾选 XHR/Fetch
3. 搜索关键词 `list` 或 `position`
4. 找到返回岗位数据的请求，复制 URL
5. 将 URL 更新到 `site_config.json` 的 `api.list.url`
6. 同时更新 `response_mapping` 中的字段路径

> 常见腾讯校招 API 路径参考：
> - `https://join.qq.com/api/campus/apply/position/list`
> - `https://join.qq.com/api/post/list`

---

## 运行模式

```bash
# 模式1：自动（优先 API，失败切 Playwright）
python scraper.py

# 模式2：强制 Playwright（适合 API 有签名保护）
python scraper.py --playwright
```

### Playwright 模式说明
- 启动真实 Chromium 浏览器（无头模式）
- 自动拦截页面发出的 XHR 请求
- 输出真实 API URL 供你更新配置

---

## response_mapping 示例

**实际 API 响应（示例）：**
```json
{
  "code": 0,
  "data": {
    "total": 120,
    "list": [
      {
        "postId": "12345",
        "postName": "后端开发实习生（Java）",
        "categoryName": "技术",
        "locationName": "深圳",
        "bgName": "IEG 互动娱乐",
        "workNature": "实习",
        "createTime": "2026-03-01"
      }
    ]
  }
}
```

**对应配置：**
```json
"response_mapping": {
  "success_flag": "code",
  "success_value": 0,
  "data_path": "data.list",
  "total_path": "data.total",
  "fields": {
    "id":           "postId",
    "title":        "postName",
    "category":     "categoryName",
    "location":     "locationName",
    "department":   "bgName",
    "type":         "workNature",
    "publish_time": "createTime"
  }
}
```

---

## 扩展到其他网站

只需复制 `site_config.json`，修改以下字段即可适配任意招聘官网：

1. `site.homepage` → 目标网址
2. `api.list.url` → 岗位列表 API
3. `api.list.pagination.*` → 翻页方式
4. `response_mapping.*` → 字段映射


## 接口启动

D:\miniconda3\envs\scraper_env\python.exe -m uvicorn rag.api:app --reload

查看Swagger页面 http://127.0.0.1:8000/docs