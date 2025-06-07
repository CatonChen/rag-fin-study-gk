# 金融文本处理工具箱

一个基于 React 和 FastAPI 构建的现代化金融文本处理工具箱，提供全面的金融文本分析服务。
<a href="https://u.geekbang.org/subject/airag/1009927"> 极客时间RAG进阶训练营</a>

学习链接： https://u.geekbang.org/subject/airag/1009927 

## 功能特点

### 1. 金融实体识别
- 支持识别多种金融实体类型：
  - 公司（COMPANY）
  - 股票（STOCK）
  - 基金（FUND）
  - 债券（BOND）
  - 货币（CURRENCY）
  - 指数（INDEX）
  - 行业（SECTOR）
  - 金融术语（FINANCIAL_TERM）
- 提供实体关系分析
- 支持实体验证和分类
- 支持批量文本处理

### 2. 金融术语标准化
- 支持多种金融术语类型
- 基于FAISS向量数据库的相似度匹配
- 提供标准术语定义
- 支持相似度阈值过滤
- 提供术语元数据查询

### 3. 金融术语缩写展开
- 支持多种展开方法
- 提供标准定义查询
- 支持上下文理解
- 支持批量处理

### 4. 金融文本纠错
- 支持多种错误类型：
  - 金融术语拼写纠正
  - 金融术语验证
  - 金融缩写处理
  - 金融数字格式处理
- 提供智能纠错建议
- 支持批量文本处理

### 5. 金融文本生成
- 支持多种生成类型：
  - 财务报告生成
  - 财务分析报告
  - 投资计划生成
  - 市场分析报告
  - 风险评估报告
  - 投资组合回顾
- 提供参数化控制
- 支持结构化输出
- 提供数据验证和格式化

## 技术栈

### 前端
- **框架**: React 18
- **路由**: React Router 6
- **UI组件**: Chakra UI 2
- **样式**: Tailwind CSS 3
- **图标**: Lucide React
- **动画**: Framer Motion
- **状态管理**: React Hooks

### 后端
- **框架**: FastAPI
- **AI模型**: 
  - 智谱GLM-4（文本生成）
  - 智谱Embedding-3（向量嵌入）
- **数据库**: 
  - SQLite（存储术语元数据）
  - FAISS（向量索引）
- **工具库**: 
  - Pydantic（数据验证）
  - SQLAlchemy（ORM）
  - ZhipuAI SDK

## 数据库结构

### 主要表结构
1. **terms（术语表）**
   - term_id: 术语ID
   - term_name: 术语名称
   - category: 术语类别
   - created_at: 创建时间

### 向量索引
- 使用智谱AI的embedding-3模型生成向量
- 基于FAISS向量数据库
- 支持相似度搜索

## 开始使用

1. 克隆仓库
2. 安装后端依赖：
   ```bash
   cd backend
   pip install -r requirements_win.txt  # Windows
   # 或
   pip install -r requirements_mac.txt  # Mac
   # 或
   pip install -r requirements_ubun.txt  # Ubuntu
   ```

3. 安装前端依赖：
   ```bash
   cd frontend
   npm install
   ```

4. 启动后端服务：
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. 启动前端开发服务器：
   ```bash
   cd frontend
   npm start
   ```

6. 在浏览器中打开 [http://localhost:3000](http://localhost:3000)

## 项目结构

```
rag-fin-study-gk/
├── backend/
│   ├── services/          # 后端服务实现
│   │   ├── ner_service.py    # 实体识别服务
│   │   ├── std_service.py    # 术语标准化服务
│   │   ├── abbr_service.py   # 缩写展开服务
│   │   ├── corr_service.py   # 文本纠错服务
│   │   └── gen_service.py    # 文本生成服务
│   ├── utils/            # 工具函数
│   ├── db/              # 数据库文件
│   ├── data/            # 数据文件
│   └── main.py          # 主程序入口
├── frontend/
│   ├── src/
│   │   ├── components/      # 可复用组件
│   │   ├── pages/          # 页面组件
│   │   │   ├── NERPage.js     # 实体识别页面
│   │   │   ├── StdPage.js     # 术语标准化页面
│   │   │   ├── AbbrPage.js    # 缩写展开页面
│   │   │   ├── CorrPage.js    # 文本纠错页面
│   │   │   └── GenPage.js     # 文本生成页面
│   │   ├── App.js           # 应用入口
│   │   └── index.js         # 渲染入口
│   └── public/           # 静态资源
└── README.md
```

## 数据查询方式

### 1. 精确查询
- 通过术语ID查询
- 通过术语名称查询
- 通过术语类别查询

### 2. 模糊查询
- 通过语义相似度查询
- 通过关键词查询
- 通过上下文查询

## 数据更新机制

### 1. 数据库更新
- 支持实时添加新术语
- 支持实时修改术语
- 支持实时删除术语

### 2. 向量索引更新
- 自动更新向量索引
- 支持增量更新
- 支持全量更新

## 参与贡献

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m '添加一些特性'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情