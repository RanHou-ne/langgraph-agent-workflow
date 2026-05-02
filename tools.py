"""
工具模块：为 Agent 提供可调用的工具集。
支持真实 API 和模拟数据两种模式，当 API Key 未配置时自动降级为模拟数据。

真实 API 支持：
  - 搜索：Tavily / SerpAPI（通过 SEARCH_API 环境变量切换）
  - 天气：OpenWeatherMap
"""
import os
import json
import httpx
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ─────────────────────────── 配置常量 ───────────────────────────

# 搜索 API 配置
SEARCH_API = os.getenv("SEARCH_API", "mock").lower().strip()  # tavily | serpapi | mock
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# 天气 API 配置
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# 中文城市名到英文的映射（用于 OpenWeatherMap）
CITY_NAME_MAP = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "成都": "Chengdu",
    "武汉": "Wuhan",
    "南京": "Nanjing",
    "西安": "Xi'an",
    "重庆": "Chongqing",
    "天津": "Tianjin",
    "苏州": "Suzhou",
    "长沙": "Changsha",
    "郑州": "Zhengzhou",
    "青岛": "Qingdao",
    "大连": "Dalian",
    "厦门": "Xiamen",
    "昆明": "Kunming",
    "哈尔滨": "Harbin",
    "济南": "Jinan",
}


# ─────────────────────────── 天气描述映射 ───────────────────────────

def _weather_code_to_chinese(description: str) -> str:
    """将 OpenWeatherMap 英文天气描述转换为中文。"""
    mapping = {
        "clear sky": "晴天 ☀️",
        "few clouds": "少云 🌤️",
        "scattered clouds": "多云 ⛅",
        "broken clouds": "阴天 ☁️",
        "overcast clouds": "阴天 ☁️",
        "shower rain": "阵雨 🌦️",
        "rain": "雨 🌧️",
        "light rain": "小雨 🌦️",
        "moderate rain": "中雨 🌧️",
        "heavy rain": "大雨 🌧️",
        "thunderstorm": "雷阵雨 ⛈️",
        "snow": "雪 🌨️",
        "light snow": "小雪 🌨️",
        "mist": "薄雾 🌫️",
        "fog": "雾 🌫️",
        "haze": "霾 🌫️",
        "drizzle": "毛毛雨 🌦️",
    }
    lower_desc = description.lower()
    for eng, chn in mapping.items():
        if eng in lower_desc:
            return chn
    return description


def _wind_direction(deg: float) -> str:
    """将风向角度转换为中文方向。"""
    directions = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    idx = round(deg / 45) % 8
    return directions[idx]


# ─────────────────────────── 天气工具 ───────────────────────────

@tool
def search_weather(city: str) -> str:
    """搜索指定城市的天气预报信息。输入城市名称，返回天气详情。"""
    # 尝试使用真实 API
    if OPENWEATHERMAP_API_KEY:
        try:
            return _fetch_weather_real(city)
        except Exception as e:
            logger.warning(f"OpenWeatherMap API 调用失败，降级为模拟数据: {e}")

    # 降级为模拟数据
    return _fetch_weather_mock(city)


def _fetch_weather_real(city: str) -> str:
    """调用 OpenWeatherMap API 获取真实天气数据。"""
    # 中文城市名转英文
    city_en = CITY_NAME_MAP.get(city, city)

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_en,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric",
        "lang": "zh_cn",
    }

    with httpx.Client(timeout=10) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    # 解析数据
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    temp_min = data["main"]["temp_min"]
    temp_max = data["main"]["temp_max"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    wind_speed = data.get("wind", {}).get("speed", 0)
    wind_deg = data.get("wind", {}).get("deg", 0)
    visibility = data.get("visibility", 10000) / 1000  # 转为 km

    weather_cn = _weather_code_to_chinese(description)
    wind_dir = _wind_direction(wind_deg)

    result = (
        f"🌤️ {city}实时天气：{weather_cn}\n"
        f"🌡️ 当前温度：{temp}°C（体感 {feels_like}°C）\n"
        f"📊 温度范围：{temp_min}°C ~ {temp_max}°C\n"
        f"💧 湿度：{humidity}%\n"
        f"💨 风向：{wind_dir}风，风速 {wind_speed} m/s\n"
        f"👁️ 能见度：{visibility:.1f} km"
    )

    # 添加出行建议
    if temp > 30:
        result += "\n💡 建议：天气炎热，注意防晒补水。"
    elif temp < 5:
        result += "\n💡 建议：天气寒冷，注意保暖。"
    elif "雨" in weather_cn:
        result += "\n💡 建议：有降水，出门记得带伞。"
    else:
        result += "\n💡 建议：天气适宜，适合外出活动。"

    return result


def _fetch_weather_mock(city: str) -> str:
    """模拟天气数据（降级方案）。"""
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


# ─────────────────────────── 搜索工具 ───────────────────────────

@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。输入搜索关键词，返回相关搜索结果。"""
    # 根据配置选择搜索 API
    if SEARCH_API == "tavily" and TAVILY_API_KEY:
        try:
            return _search_tavily(query)
        except Exception as e:
            logger.warning(f"Tavily 搜索失败，降级为模拟数据: {e}")
    elif SEARCH_API == "serpapi" and SERPAPI_API_KEY:
        try:
            return _search_serpapi(query)
        except Exception as e:
            logger.warning(f"SerpAPI 搜索失败，降级为模拟数据: {e}")

    # 降级为模拟数据
    return _search_mock(query)


def _search_tavily(query: str) -> str:
    """使用 Tavily Search API 进行搜索。"""
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "使用 Tavily 搜索需要安装 tavily-python：\n"
            "  pip install tavily-python"
        )

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query=query,
        max_results=5,
        search_depth="basic",
    )

    results = response.get("results", [])
    if not results:
        return f"🔍 关于「{query}」未找到相关结果。"

    output = f"🔍 搜索结果：关于「{query}」\n\n"
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        url = r.get("url", "")
        snippet = r.get("content", "")[:200]
        output += f"  {i}. {title}\n"
        output += f"     {snippet}\n"
        output += f"     🔗 {url}\n\n"

    return output.strip()


def _search_serpapi(query: str) -> str:
    """使用 SerpAPI (Google) 进行搜索。"""
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num": 5,
        "hl": "zh-cn",
        "gl": "cn",
    }

    with httpx.Client(timeout=15) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    organic = data.get("organic_results", [])
    if not organic:
        return f"🔍 关于「{query}」未找到相关结果。"

    output = f"🔍 搜索结果：关于「{query}」\n\n"
    for i, r in enumerate(organic[:5], 1):
        title = r.get("title", "无标题")
        link = r.get("link", "")
        snippet = r.get("snippet", "")[:200]
        output += f"  {i}. {title}\n"
        output += f"     {snippet}\n"
        output += f"     🔗 {link}\n\n"

    return output.strip()


def _search_mock(query: str) -> str:
    """模拟搜索结果（降级方案）。"""
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


# ─────────────────────────── 餐厅推荐工具 ───────────────────────────

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


# ─────────────────────────── 电影推荐工具 ───────────────────────────

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


# ─────────────────────────── 计算器工具 ───────────────────────────

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


# ─────────────────────────── 时间查询工具 ───────────────────────────

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


def get_api_status() -> dict:
    """
    获取各 API 的配置状态（用于诊断）。

    Returns:
        包含各 API 状态信息的字典
    """
    return {
        "search": {
            "provider": SEARCH_API,
            "tavily_configured": bool(TAVILY_API_KEY),
            "serpapi_configured": bool(SERPAPI_API_KEY),
        },
        "weather": {
            "provider": "openweathermap" if OPENWEATHERMAP_API_KEY else "mock",
            "openweathermap_configured": bool(OPENWEATHERMAP_API_KEY),
        },
    }