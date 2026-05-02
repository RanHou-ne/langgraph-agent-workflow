"""
工具模块：为 Agent 提供可调用的工具集。
支持模拟数据和真实 API 调用两种模式。
"""
import os
import json
import httpx
from langchain_core.tools import tool


# ─────────────────────────── 模拟数据工具 ───────────────────────────

@tool
def search_weather(city: str) -> str:
    """搜索指定城市的天气预报信息。输入城市名称，返回天气详情。"""
    # 模拟天气数据（可替换为真实 API）
    weather_data = {
        "北京": "北京：晴转多云，气温 18~28℃，空气质量良，适合外出。",
        "上海": "上海：多云，气温 20~26℃，东南风3级，偶有阵雨。",
        "广州": "广州：雷阵雨，气温 24~32℃，湿度较高，注意防雨。",
        "深圳": "深圳：多云转晴，气温 25~31℃，适合户外活动。",
        "杭州": "杭州：晴，气温 19~27℃，西湖风景宜人。",
        "成都": "成都：阴转多云，气温 16~24℃，适合吃火锅。",
        "武汉": "武汉：晴，气温 20~30℃，紫外线较强。",
        "南京": "南京：多云，气温 18~26℃，适合出游。",
    }
    for key, value in weather_data.items():
        if key in city:
            return value
    return f"{city}：多云转晴，气温 20~28℃，适合外出活动。"


@tool
def search_restaurant(location: str, cuisine: str = "") -> str:
    """搜索指定位置附近的餐厅推荐。输入位置和可选的菜系偏好。"""
    restaurants = {
        "北京": [
            "🍜 大董烤鸭店（人均 280 元，正宗北京烤鸭）",
            "🍲 东来顺饭庄（人均 150 元，老字号涮羊肉）",
            "🥟 庆丰包子铺（人均 25 元，经济实惠）",
        ],
        "上海": [
            "🦀 王宝和酒家（人均 200 元，蟹粉小笼）",
            "🍜 南翔馒头店（人均 60 元，小笼包名店）",
            "🥘 望湘园（人均 90 元，正宗湘菜）",
        ],
        "广州": [
            "🦐 炳胜品味（人均 180 元，粤菜精品）",
            "🍜 银记肠粉（人均 30 元，地道肠粉）",
            "🍲 广州酒家（人均 120 元，老字号粤菜）",
        ],
    }
    for key, values in restaurants.items():
        if key in location:
            result = f"📍 {location}附近推荐餐厅：\n"
            for v in values:
                result += f"  {v}\n"
            if cuisine:
                result += f"\n💡 偏好菜系：{cuisine}，建议优先选择相关餐厅。"
            return result
    return f"📍 {location}附近推荐：巷子里的私房菜（人均 80 元，口碑好）、老码头火锅（人均 100 元）。"


@tool
def search_movie(genre: str = "") -> str:
    """搜索当前热映或推荐的电影。可选输入电影类型偏好。"""
    movies = [
        "🎬 《星际穿越》重映 - 科幻经典，诺兰执导，豆瓣 9.4",
        "🎬 《长安三万里》- 动画/历史，诗意盎然，适合全家观看",
        "🎬 《奥本海默》- 传记/历史，诺兰执导，奥斯卡最佳影片",
        "🎬 《流浪地球2》- 科幻/冒险，中国科幻里程碑",
        "🎬 《灌篮高手》- 动画/运动，青春回忆，热血沸腾",
        "🎬 《消失的她》- 悬疑/犯罪，剧情反转不断",
    ]
    if genre:
        filtered = [m for m in movies if genre in m]
        if filtered:
            return "🎥 根据您的偏好推荐：\n" + "\n".join(filtered)
    return "🎥 当前热映推荐：\n" + "\n".join(movies)


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入数学表达式字符串，返回计算结果。支持基本四则运算。"""
    try:
        # 安全限制：只允许数字和基本运算符
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"❌ 表达式包含不允许的字符：{expression}"
        result = eval(expression)
        return f"🔢 {expression} = {result}"
    except Exception as e:
        return f"❌ 计算错误：{str(e)}"


@tool
def web_search(query: str) -> str:
    """模拟网络搜索。输入搜索关键词，返回相关搜索结果。"""
    # 模拟搜索结果（可替换为真实搜索 API 如 SerpAPI、Tavily 等）
    results = {
        "default": [
            f"🔍 搜索结果：关于「{query}」",
            f"  1. {query} - 百科介绍：这是一个热门话题...",
            f"  2. {query} - 最新资讯：相关报道显示...",
            f"  3. {query} - 用户评价：大多数人给出了正面反馈...",
        ]
    }
    for key, values in results.items():
        if key in query.lower():
            return "\n".join(values)
    return "\n".join(results["default"])


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    from datetime import datetime
    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return f"🕐 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[now.weekday()]}"


# ─────────────────────────── 工具注册表 ───────────────────────────

# 所有可用工具列表
ALL_TOOLS = [
    search_weather,
    search_restaurant,
    search_movie,
    calculator,
    web_search,
    get_current_time,
]

# 工具名称到工具对象的映射
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def get_tool_by_name(name: str):
    """根据工具名称获取工具对象。"""
    return TOOL_MAP.get(name)


def execute_tool(name: str, **kwargs) -> str:
    """根据工具名称和参数执行工具。"""
    tool_fn = get_tool_by_name(name)
    if tool_fn is None:
        return f"❌ 未找到工具：{name}"
    try:
        return tool_fn.invoke(kwargs)
    except Exception as e:
        return f"❌ 工具执行错误：{str(e)}"