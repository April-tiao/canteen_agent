from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any


TODAY = date.today()


MEAL_PERIODS = {
    "breakfast": ("早餐", "早饭", "早上", "上午", "早晨"),
    "lunch": ("午餐", "午饭", "中午", "午间"),
    "dinner": ("晚餐", "晚饭", "晚上", "傍晚", "今晚"),
    "night_snack": ("夜宵", "宵夜", "夜餐"),
}


CORRECTION_TABLE = {
    "宫爆鸡丁": "宫保鸡丁",
    "冲值": "充值",
    "充直": "充值",
    "饭咔": "饭卡",
    "二食唐": "二食堂",
    "一食唐": "一食堂",
    "三食唐": "三食堂",
    "扣废": "扣费",
    "扣宽": "扣款",
    "预定": "预订",
    "退欵": "退款",
    "麻辣堂": "麻辣烫",
    "沙县小尺": "沙县小吃",
    "营页": "营业",
    "档囗": "档口",
    "发飘": "发票",
    "荤素达配": "荤素搭配",
}


CANTEEN_KEYWORDS = {
    "食堂",
    "餐厅",
    "档口",
    "窗口",
    "饭卡",
    "餐卡",
    "余额",
    "充值",
    "退款",
    "扣款",
    "扣费",
    "订单",
    "支付",
    "预订",
    "订餐",
    "取餐",
    "菜",
    "菜品",
    "菜单",
    "饭",
    "餐",
    "早餐",
    "午餐",
    "晚餐",
    "夜宵",
    "价格",
    "多少钱",
    "营业",
    "开门",
    "关门",
    "投诉",
    "评价",
    "发票",
    "打包",
    "少辣",
    "不辣",
    "清真",
    "素食",
}


DISH_KEYWORDS = {
    "宫保鸡丁",
    "鱼香肉丝",
    "番茄炒蛋",
    "鸡腿饭",
    "牛肉面",
    "麻辣烫",
    "黄焖鸡",
    "沙县小吃",
    "拉面",
    "米饭",
    "包子",
    "豆浆",
    "粥",
    "水饺",
    "麻婆豆腐",
    "红烧肉",
    "青椒肉丝",
    "鸡蛋",
    "鸡腿",
    "土豆牛肉",
}


OUT_DOMAIN_PATTERNS = (
    "写一首诗",
    "写篇论文",
    "生成代码",
    "数学题",
    "股票",
    "股市",
    "基金",
    "天气",
    "手机",
    "电脑",
    "旅游",
    "酒店",
    "电影",
    "游戏",
    "王者荣耀",
    "海报",
    "行业趋势",
    "经营模式",
    "英语作文",
    "翻译",
    "会议",
    "机票",
)


TIME_RELATIVE_DAYS = {
    "今天": 0,
    "今晚": 0,
    "明天": 1,
    "后天": 2,
    "昨天": -1,
    "前天": -2,
}


TIME_WEEK_RANGES = {
    "上周": -1,
    "上星期": -1,
    "上礼拜": -1,
    "本周": 0,
    "这周": 0,
    "本星期": 0,
    "这个星期": 0,
    "下周": 1,
    "下星期": 1,
    "下礼拜": 1,
}


TIME_MONTH_RANGES = {
    "上个月": -1,
    "上月": -1,
    "这个月": 0,
    "本月": 0,
    "下个月": 1,
    "下月": 1,
}


TIME_QUARTER_RANGES = {
    "上个季度": -1,
    "上季度": -1,
    "这个季度": 0,
    "本季度": 0,
    "下个季度": 1,
    "下季度": 1,
}


TIME_YEAR_RANGES = {
    "去年": -1,
    "上一年": -1,
    "今年": 0,
    "本年": 0,
    "明年": 1,
    "下一年": 1,
}


WEEKDAY_MAP = {
    "周一": 0,
    "星期一": 0,
    "礼拜一": 0,
    "周二": 1,
    "星期二": 1,
    "礼拜二": 1,
    "周三": 2,
    "星期三": 2,
    "礼拜三": 2,
    "周四": 3,
    "星期四": 3,
    "礼拜四": 3,
    "周五": 4,
    "星期五": 4,
    "礼拜五": 4,
    "周六": 5,
    "星期六": 5,
    "礼拜六": 5,
    "周日": 6,
    "星期日": 6,
    "礼拜日": 6,
    "周天": 6,
    "星期天": 6,
    "礼拜天": 6,
}


CHINESE_HOURS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


def normalize_text(query: str) -> str:
    text = query.strip()
    text = re.sub(r"\s+", "", text)
    return text.replace("？", "?").replace("！", "!")


def correct_query(query: str) -> tuple[str, list[dict[str, Any]]]:
    corrected = query
    corrections: list[dict[str, Any]] = []
    for wrong, right in CORRECTION_TABLE.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, right)
            corrections.append(
                {
                    "from": wrong,
                    "to": right,
                    "type": "domain_dictionary",
                    "confidence": 0.96,
                }
            )
    return corrected, corrections


def contains_any(query: str, words: set[str] | tuple[str, ...]) -> bool:
    return any(word in query for word in words)


def add_months(base: date, offset_months: int) -> date:
    month_index = base.year * 12 + base.month - 1 + offset_months
    year = month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


def week_range(base: date, offset_weeks: int = 0) -> tuple[date, date]:
    monday = base - timedelta(days=base.weekday()) + timedelta(days=offset_weeks * 7)
    return monday, monday + timedelta(days=6)


def month_range(base: date, offset_months: int = 0) -> tuple[date, date]:
    first = add_months(base, offset_months)
    return first, add_months(first, 1) - timedelta(days=1)


def quarter_range(base: date, offset_quarters: int = 0) -> tuple[date, date]:
    first_month = ((base.month - 1) // 3) * 3 + 1
    current_start = date(base.year, first_month, 1)
    start = add_months(current_start, offset_quarters * 3)
    return start, add_months(start, 3) - timedelta(days=1)


def year_range(base: date, offset_years: int = 0) -> tuple[date, date]:
    year = base.year + offset_years
    return date(year, 1, 1), date(year, 12, 31)


def next_weekday(base: date, target_weekday: int) -> date:
    delta = (target_weekday - base.weekday()) % 7
    return base + timedelta(days=delta or 7)


def build_time_result(
    raw_parts: list[str],
    start_date: date,
    end_date: date,
    *,
    meal_period: str | None = None,
    clock_text: str | None = None,
    granularity: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "raw": "".join(raw_parts),
        "granularity": granularity,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
    }
    if meal_period:
        result["meal_period"] = meal_period
    if clock_text:
        result.update(
            {
                "granularity": "time_point",
                "time_point": start_date.isoformat(),
                "start": start_date.isoformat(),
                "end": start_date.isoformat(),
            }
        )
    return result


def extract_time(query: str, today: date | None = None) -> dict[str, Any] | None:
    base_date = today or date.today()
    raw_parts: list[str] = []
    start_date: date | None = None
    end_date: date | None = None
    meal_period: str | None = None
    granularity: str | None = None

    for word, year_offset in TIME_YEAR_RANGES.items():
        if word in query:
            raw_parts.append(word)
            start_date, end_date = year_range(base_date, year_offset)
            granularity = "year"
            break

    if start_date is None:
        for word, quarter_offset in TIME_QUARTER_RANGES.items():
            if word in query:
                raw_parts.append(word)
                start_date, end_date = quarter_range(base_date, quarter_offset)
                granularity = "quarter"
                break

    if start_date is None:
        for word, month_offset in TIME_MONTH_RANGES.items():
            if word in query:
                raw_parts.append(word)
                start_date, end_date = month_range(base_date, month_offset)
                granularity = "month"
                break

    if start_date is None:
        for word, week_offset in TIME_WEEK_RANGES.items():
            if word in query:
                raw_parts.append(word)
                start_date, end_date = week_range(base_date, week_offset)
                granularity = "week"
                break

    if start_date is None:
        for word, offset in TIME_RELATIVE_DAYS.items():
            if word in query:
                raw_parts.append(word)
                start_date = base_date + timedelta(days=offset)
                end_date = start_date
                granularity = "day"
                break

    if start_date is None:
        for word, weekday in WEEKDAY_MAP.items():
            if word in query:
                raw_parts.append(word)
                start_date = next_weekday(base_date, weekday)
                end_date = start_date
                granularity = "day"
                break

    for period, aliases in MEAL_PERIODS.items():
        for alias in aliases:
            if alias in query:
                raw_parts.append(alias)
                meal_period = period
                break
        if meal_period:
            break

    clock_text = None
    clock_match = re.search(r"(?P<hour>[01]?\d|2[0-3])[:点](?P<minute>[0-5]\d|半)?", query)
    if clock_match:
        clock_text = clock_match.group(0)
        raw_parts.append(clock_text)

    chinese_clock_match = re.search(
        r"(上午|中午|下午|晚上|今晚)?(?P<hour>十一|十二|十|[一二两三四五六七八九])点(?P<half>半)?",
        query,
    )
    if clock_text is None and chinese_clock_match:
        clock_text = chinese_clock_match.group(0)
        raw_parts.append(clock_text)

    if start_date is None and (meal_period or clock_text):
        start_date = base_date
        end_date = base_date
        granularity = "day"

    if start_date is None or end_date is None:
        return None

    if meal_period:
        granularity = "meal_period"

    return build_time_result(
        raw_parts,
        start_date,
        end_date,
        meal_period=meal_period,
        clock_text=clock_text,
        granularity=granularity or "day",
    )


def rule_domain_score(query: str, time_info: dict[str, Any] | None) -> float:
    score = 0.08
    keyword_hits = sum(1 for word in CANTEEN_KEYWORDS if word in query)
    dish_hits = sum(1 for word in DISH_KEYWORDS if word in query)

    score += min(keyword_hits * 0.16, 0.56)
    score += min(dish_hits * 0.22, 0.32)

    if re.search(r"[一二三四五六七八九十]食堂|[1-9]食堂", query):
        score += 0.26
    if contains_any(query, ("多少钱", "价格", "贵不贵", "有吗", "还有吗")):
        score += 0.16
    if contains_any(query, ("退款", "扣款", "扣费", "充值", "余额", "饭卡", "餐卡")):
        score += 0.24
    if contains_any(query, ("支付", "发票", "取餐", "打包", "订", "预订", "订餐")):
        score += 0.20
    if contains_any(query, ("开门", "关门", "营业", "几点", "开吗")):
        score += 0.16
    if contains_any(query, ("投诉", "评价")) and contains_any(query, ("窗口", "档口", "排队")):
        score += 0.22
    if contains_any(query, ("有什么", "有吗", "还有吗")) and (
        dish_hits > 0 or contains_any(query, ("素食", "清真", "午餐", "晚餐", "早餐", "夜宵"))
    ):
        score += 0.20
    if "发票" in query:
        score += 0.24
    if time_info:
        score += 0.08

    if contains_any(query, OUT_DOMAIN_PATTERNS):
        score -= 0.55
    if "食堂" in query and contains_any(query, ("论文", "海报", "行业趋势", "经营模式")):
        score -= 0.35

    return max(0.0, min(1.0, score))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p)))
    return sorted_values[index]
