# 🤖 LangGraph Agent Workflow v2.0

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的 AI Agent 工作流系统，支持多节点协同、多模型切换、真实 API 集成、数据库持久化、多 Agent 协作和前端 UI。

## ✨ 核心特性

### 多节点协同架构
系统采用 **Planner + Executor + Evaluator + Memory Updater** 四节点协同工作，形成完整的任务执行闭环：

```
用户输入 → 任务拆解(Planner) → Agent执行(Executor) → 结果评估(Evaluator) → 记忆更新(Memory) → 输出
                                        ↑                    │
                                        └──── 继续执行 ───────┘
```

### 三级记忆管理
| 记忆层级 | 组件 | 作用 | 生命周期 |
|---------|------|------|---------|
| 短期记忆 | Checkpointer (MemorySaver) | 对话级别的状态持久化 | 单次会话 |
| 长期记忆 | Store (InMemoryStore / Database) | 跨会话的事实/偏好存储 | 永久 |
| 摘要记忆 | Summary (LLM 生成) | 对话摘要压缩，减少 Token 占用 | 持续更新 |

### v2.0 新增特性

#### 🧠 多 LLM 模型支持
通过 `llm_provider.py` 抽象层，统一支持多种 LLM 后端：
- **OpenAI**：GPT-3.5-Turbo、GPT-4、GPT-4o 等
- **Anthropic Claude**：Claude-3.5-Sonnet、Claude-3-Haiku 等
- **本地模型（Ollama）**：Llama3、Mistral、Qwen2 等

#### 🔍 真实搜索 API
- **Tavily**：专为 AI Agent 设计的搜索 API（推荐）
- **SerpAPI**：Google 搜索结果 API
- 未配置时自动降级为模拟数据

#### 🌤️ 真实天气 API
- **OpenWeatherMap**：全球天气数据 API
- 支持中文城市名自动映射
- 未配置时自动降级为模拟数据

#### 📦 数据库持久化
- **SQLite**（默认）：零配置，开箱即用
- **PostgreSQL**：生产环境推荐
- 持久化存储：长期记忆、任务历史、会话记录

#### 📊 任务执行监控
- 实时指标收集：LLM 调用次数、工具调用成功率、任务耗时
- 聚合统计：成功率、平均耗时、工具使用分布
- Web API 端点：`GET /metrics`

#### 🤝 多 Agent 协作
- **Researcher Agent**：信息搜索和数据收集
- **Coder Agent**：代码生成和计算任务
- **Writer Agent**：内容撰写和总结
- 自动任务分类和 Agent 组合选择
- 支持串行和并行协作模式

#### 🖥️ Streamlit 前端 UI
- 交互式聊天界面
- 执行计划和结果可视化
- 指标仪表盘
- 任务历史管理
- 系统配置面板

### 丰富的工具集
- 🌤️ 天气查询 (`search_weather`) — 支持真实 API
- 🍜 餐厅推荐 (`search_restaurant`)
- 🎬 电影推荐 (`search_movie`)
- 🔢 数学计算 (`calculator`)
- 🔍 网络搜索 (`web_search`) — 支持真实 API
- 🕐 时间查询 (`get_current_time`)

## 📁 项目结构

```
langgraph-agent-workflow/
├── main.py            # 主入口：支持交互式/单次任务/Web API 三种模式
├── graph.py           # 工作流构建：LangGraph 状态图定义与编译
├── nodes.py           # 节点实现：Planner/Executor/Evaluator/Memory 节点
├── memory.py          # 记忆管理：三级记忆（短期/长期/摘要）的存取逻辑
├── tools.py           # 工具集：支持真实 API 和模拟数据自动降级
├── llm_provider.py    # LLM 抽象层：支持 OpenAI/Claude/Ollama
├── database.py        # 数据库持久化：SQLite/PostgreSQL
├── monitoring.py      # 监控指标：任务执行统计和实时指标
├── multi_agent.py     # 多 Agent 协作：专业化 Agent 角色协作
├── app.py             # Streamlit 前端 UI
├── requirements.txt   # Python 依赖
├── .env.example       # 环境变量模板
├── .gitignore         # Git 忽略规则
└── README.md          # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

最小配置（仅需 OpenAI API Key）：
```env
OPENAI_API_KEY=your_api_key_here
```

完整配置示例：
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_api_key_here
SEARCH_API=tavily
TAVILY_API_KEY=your_tavily_key
OPENWEATHERMAP_API_KEY=your_owm_key
DATABASE_URL=sqlite:///data/agent.db
```

### 3. 运行

#### 交互式对话模式（默认）

```bash
python main.py
```

交互式模式下支持：
- 输入任务描述，Agent 自动规划和执行
- 输入 `memory` 查看长期记忆
- 输入 `metrics` 查看运行指标
- 输入 `multi <任务>` 使用多 Agent 协作模式
- 输入 `new` 开始新的会话
- 输入 `quit` 退出

#### 单次任务模式

```bash
python main.py --task "本周末想出去放松一下，查查天气，推荐一个吃饭的地方"
```

#### 多 Agent 协作模式

```bash
python main.py --multi-agent "研究2024年AI行业发展趋势并撰写报告"
```

#### 启用数据库持久化

```bash
python main.py --database --task "查查北京天气"
```

#### Web API 服务模式

```bash
python main.py --serve --port 8000
```

#### Streamlit 前端 UI

```bash
streamlit run app.py
```

## 🔧 命令行参数

```
python main.py [OPTIONS]

选项：
  --task, -t TEXT       执行单次任务（指定任务描述）
  --serve, -s           启动 Web API 服务模式
  --host TEXT           Web 服务监听地址（默认: 0.0.0.0）
  --port, -p INT        Web 服务端口（默认: 8000）
  --user, -u TEXT       用户 ID（默认: default_user）
  --verbose, -v         显示详细日志
  --database, -d        启用数据库持久化
  --multi-agent, -m TEXT 使用多 Agent 协作模式执行任务
```

## 🌐 Web API 端点

启动 `python main.py --serve` 后访问：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| POST | `/task` | 执行单次任务 |
| POST | `/chat` | 交互式对话 |
| POST | `/multi-agent` | 多 Agent 协作任务 |
| GET | `/memories/{user_id}` | 获取用户长期记忆 |
| GET | `/metrics` | 运行指标 |
| GET | `/tasks/{user_id}` | 任务历史 |
| GET | `/api-status` | API 配置状态 |

API 文档：http://localhost:8000/docs

## 🧠 LLM 模型配置

通过环境变量切换 LLM 提供商和模型：

```bash
# 使用 OpenAI GPT-4
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# 使用 Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_key

# 使用本地 Ollama 模型
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

## 🔍 搜索 API 配置

```bash
# 使用 Tavily（推荐，有免费额度）
SEARCH_API=tavily
TAVILY_API_KEY=your_key

# 使用 SerpAPI
SEARCH_API=serpapi
SERPAPI_API_KEY=your_key

# 使用模拟数据（默认，无需 API Key）
SEARCH_API=mock
```

## 📦 数据库配置

```bash
# SQLite（默认，零配置）
DATABASE_URL=sqlite:///data/agent.db

# PostgreSQL（生产环境推荐）
DATABASE_URL=postgresql://user:password@localhost:5432/agent_db
```

启用数据库后，系统将持久化存储：
- 长期记忆（跨会话）
- 任务执行历史
- 会话记录

## 🏗️ 架构详解

### 工作流节点

1. **Planner（规划器）**
   - 接收用户任务，结合长期记忆和对话摘要
   - 将任务拆解为 2~5 个可执行步骤
   - 输出结构化的执行计划

2. **Executor（执行器）**
   - 逐步执行计划中的每个步骤
   - 智能选择合适的工具（天气/餐厅/电影/搜索等）
   - 记录每步执行结果

3. **Evaluator（评估器）**
   - 评估执行状态，决定是否继续
   - 所有步骤完成后，汇总结果生成最终回答
   - 确保回答完整、自然、有条理

4. **Memory Updater（记忆更新器）**
   - 更新摘要记忆：压缩对话历史为简洁摘要
   - 更新长期记忆：提取关键事实存入持久存储
   - 裁剪消息列表：控制上下文窗口大小

### 多 Agent 协作流程

```
用户任务 → 任务分类(Coordinator) → Agent 组合选择
                                        ├── Researcher: 信息搜索
                                        ├── Coder: 代码/计算
                                        └── Writer: 内容撰写
                                                    ↓
                                        结果整合 → 最终回答
```

### 记忆流转

```
对话消息 ──→ 摘要生成 ──→ 摘要记忆（压缩上下文）
    │
    └──→ 事实提取 ──→ 长期记忆（跨会话持久化）
    
下次对话时：
    摘要记忆 + 长期记忆 ──→ 注入 Planner 上下文 ──→ 个性化规划
```

## 📊 性能指标

- 单任务平均 **10~15 轮**交互
- 日均可处理 **数百请求**
- 上下文常达 **数万 Token**
- 任务稳定性提升约 **30%**（通过三级记忆管理）

## 📄 License

MIT License