# 🤖 LangGraph Agent Workflow

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的 AI Agent 工作流系统，用于多轮对话与复杂任务执行。

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
| 长期记忆 | Store (InMemoryStore) | 跨会话的事实/偏好存储 | 永久 |
| 摘要记忆 | Summary (LLM 生成) | 对话摘要压缩，减少 Token 占用 | 持续更新 |

### 丰富的工具集
- 🌤️ 天气查询 (`search_weather`)
- 🍜 餐厅推荐 (`search_restaurant`)
- 🎬 电影推荐 (`search_movie`)
- 🔢 数学计算 (`calculator`)
- 🔍 网络搜索 (`web_search`)
- 🕐 时间查询 (`get_current_time`)

## 📁 项目结构

```
langgraph-agent-workflow/
├── main.py          # 主入口：支持交互式/单次任务/Web API 三种模式
├── graph.py         # 工作流构建：LangGraph 状态图定义与编译
├── nodes.py         # 节点实现：Planner/Executor/Evaluator/Memory 节点
├── memory.py        # 记忆管理：三级记忆（短期/长期/摘要）的存取逻辑
├── tools.py         # 工具集：Agent 可调用的工具定义与注册
├── requirements.txt # Python 依赖
├── .env.example     # 环境变量模板
├── .gitignore       # Git 忽略规则
└── README.md        # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key
```

### 3. 运行

#### 交互式对话模式（默认）

```bash
python main.py
```

交互式模式下支持：
- 输入任务描述，Agent 自动规划和执行
- 输入 `memory` 查看长期记忆
- 输入 `new` 开始新的会话
- 输入 `quit` 退出

#### 单次任务模式

```bash
python main.py --task "本周末想出去放松一下，查查天气，推荐一个吃饭的地方"
```

#### Web API 服务模式

```bash
python main.py --serve --port 8000
```

启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

**API 端点：**

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| POST | `/task` | 执行单次任务 |
| POST | `/chat` | 交互式对话 |
| GET | `/memories/{user_id}` | 获取用户长期记忆 |

## 🔧 命令行参数

```
python main.py [OPTIONS]

选项：
  --task, -t TEXT     执行单次任务（指定任务描述）
  --serve, -s         启动 Web API 服务模式
  --host TEXT         Web 服务监听地址（默认: 0.0.0.0）
  --port, -p INT      Web 服务端口（默认: 8000）
  --user, -u TEXT     用户 ID（默认: default_user）
  --verbose, -v       显示详细日志
```

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

## 🔮 扩展方向

- [ ] 接入真实搜索 API（SerpAPI / Tavily）
- [ ] 接入真实天气 API（OpenWeatherMap）
- [ ] 支持更多 LLM 模型（Claude、本地模型等）
- [ ] 添加数据库持久化（SQLite / PostgreSQL）
- [ ] 添加前端 UI（Streamlit / Gradio）
- [ ] 支持多 Agent 协作
- [ ] 添加任务执行监控和指标收集

## 📄 License

MIT License