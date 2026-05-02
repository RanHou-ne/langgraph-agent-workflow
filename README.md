# langgraph-agent-workflow
用 LangGraph 搭了个多智能体工作流，专门啃多轮对话和长链路任务。没整花活，就 Planner、Executor、Memory 三个节点转起来，走「用户输入 → 任务拆解 → Agent 执行 → 结果评估 → 记忆更新」的闭环，评估完决定继续拆还是直接收
