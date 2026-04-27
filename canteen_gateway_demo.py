from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


TODAY = date(2026, 4, 27)
DATA_FILE = Path(__file__).with_name("canteen_test_data_300.json")


MEAL_PERIODS = {
    "breakfast": {
        "aliases": ("早餐", "早饭", "早上", "上午", "早晨"),
        "start": "07:00:00",
        "end": "10:00:00",
    },
    "lunch": {
        "aliases": ("午餐", "午饭", "中午", "午间"),
        "start": "11:00:00",
        "end": "14:00:00",
    },
    "dinner": {
        "aliases": ("晚餐", "晚饭", "晚上", "傍晚", "今晚"),
        "start": "17:00:00",
        "end": "20:00:00",
    },
    "night_snack": {
        "aliases": ("夜宵", "宵夜", "夜餐"),
        "start": "20:30:00",
        "end": "23:30:00",
    },
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


@dataclass(frozen=True)
class GatewayResult:
    domain: str
    domain_score: float
    original_query: str
    corrected_query: str
    correction_applied: bool
    corrections: list[dict[str, Any]]
    time: dict[str, Any] | None
    latency_ms: float


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


def next_weekday(base: date, target_weekday: int) -> date:
    delta = (target_weekday - base.weekday()) % 7
    return base + timedelta(days=delta or 7)


def extract_time(query: str, today: date = TODAY) -> dict[str, Any] | None:
    raw_parts: list[str] = []
    target_date: date | None = None
    meal_period: str | None = None

    for word, offset in TIME_RELATIVE_DAYS.items():
        if word in query:
            raw_parts.append(word)
            target_date = today + timedelta(days=offset)
            break

    if target_date is None:
        if "下周" in query:
            raw_parts.append("下周")
            target_date = today + timedelta(days=7)
        elif "本周" in query or "这周" in query:
            raw_parts.append("本周")
            target_date = today

    if target_date is None:
        for word, weekday in WEEKDAY_MAP.items():
            if word in query:
                raw_parts.append(word)
                target_date = next_weekday(today, weekday)
                break

    for period, config in MEAL_PERIODS.items():
        for alias in config["aliases"]:
            if alias in query:
                raw_parts.append(alias)
                meal_period = period
                break
        if meal_period:
            break

    clock_text = None
    clock_match = re.search(r"(?P<hour>[01]?\d|2[0-3])[:点](?P<minute>[0-5]\d|半)?", query)
    if clock_match:
        hour = int(clock_match.group("hour"))
        minute_raw = clock_match.group("minute")
        minute = 30 if minute_raw == "半" else int(minute_raw or 0)
        clock_text = f"{hour:02d}:{minute:02d}:00"
        raw_parts.append(clock_match.group(0))

    chinese_clock_match = re.search(
        r"(上午|中午|下午|晚上|今晚)?(?P<hour>十一|十二|十|[一二两三四五六七八九])点(?P<half>半)?",
        query,
    )
    if clock_text is None and chinese_clock_match:
        prefix = chinese_clock_match.group(1) or ""
        hour = CHINESE_HOURS[chinese_clock_match.group("hour")]
        if prefix in {"下午", "晚上", "今晚"} and hour < 12:
            hour += 12
        minute = 30 if chinese_clock_match.group("half") else 0
        clock_text = f"{hour:02d}:{minute:02d}:00"
        raw_parts.append(chinese_clock_match.group(0))

    if target_date is None and (meal_period or clock_text):
        target_date = today

    if target_date is None:
        return None

    result: dict[str, Any] = {
        "raw": "".join(raw_parts),
        "date": target_date.isoformat(),
    }

    if meal_period:
        period_config = MEAL_PERIODS[meal_period]
        result.update(
            {
                "meal_period": meal_period,
                "start": f"{target_date.isoformat()} {period_config['start']}",
                "end": f"{target_date.isoformat()} {period_config['end']}",
            }
        )

    if clock_text:
        result["time_point"] = f"{target_date.isoformat()} {clock_text}"

    return result


def contains_any(query: str, words: set[str] | tuple[str, ...]) -> bool:
    return any(word in query for word in words)


def domain_score(query: str, time_info: dict[str, Any] | None) -> float:
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


def classify_domain(score: float) -> str:
    return "in_domain" if score >= 0.45 else "out_domain"


def gateway(query: str, today: date = TODAY) -> GatewayResult:
    start = time.perf_counter_ns()

    normalized = normalize_text(query)
    corrected_query, corrections = correct_query(normalized)
    time_info = extract_time(corrected_query, today=today)
    score = domain_score(corrected_query, time_info)
    domain = classify_domain(score)

    latency_ms = (time.perf_counter_ns() - start) / 1_000_000
    return GatewayResult(
        domain=domain,
        domain_score=round(score, 4),
        original_query=query,
        corrected_query=corrected_query,
        correction_applied=bool(corrections),
        corrections=corrections,
        time=time_info,
        latency_ms=latency_ms,
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p)))
    return sorted_values[index]


def load_test_data(path: Path = DATA_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing test data file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate() -> None:
    test_data = load_test_data()
    rows: list[dict[str, Any]] = []
    domain_correct = 0
    time_correct = 0
    correction_correct = 0
    latencies: list[float] = []

    for _ in range(300):
        gateway("今天中午二食堂有宫爆鸡丁吗")

    for i, item in enumerate(test_data, start=1):
        result = gateway(item["q"])
        latencies.append(result.latency_ms)

        is_domain_correct = result.domain == item["domain"]
        is_time_correct = bool(result.time) == item["time"]
        is_correction_correct = result.corrected_query == item["corrected"]

        domain_correct += int(is_domain_correct)
        time_correct += int(is_time_correct)
        correction_correct += int(is_correction_correct)

        rows.append(
            {
                "id": i,
                "query": item["q"],
                "pred_domain": result.domain,
                "score": result.domain_score,
                "expected_domain": item["domain"],
                "time_found": bool(result.time),
                "expected_time": item["time"],
                "corrected": result.corrected_query,
                "expected_corrected": item["corrected"],
                "latency_ms": result.latency_ms,
                "ok": is_domain_correct and is_time_correct and is_correction_correct,
            }
        )

    total = len(test_data)
    end_to_end_correct = sum(1 for row in rows if row["ok"])

    print("=== Smart Canteen Gateway Demo ===")
    print(f"base_date: {TODAY.isoformat()}")
    print(f"data_file: {DATA_FILE.name}")
    print(f"test_size: {total}")
    print()

    print("=== Accuracy ===")
    print(f"domain_accuracy:     {domain_correct / total:.2%} ({domain_correct}/{total})")
    print(f"time_accuracy:       {time_correct / total:.2%} ({time_correct}/{total})")
    print(f"correction_accuracy: {correction_correct / total:.2%} ({correction_correct}/{total})")
    print(f"end_to_end_accuracy: {end_to_end_correct / total:.2%} ({end_to_end_correct}/{total})")
    print()

    print("=== Latency ===")
    print(f"avg_ms: {statistics.mean(latencies):.4f}")
    print(f"p50_ms: {statistics.median(latencies):.4f}")
    print(f"p95_ms: {percentile(latencies, 0.95):.4f}")
    print(f"p99_ms: {percentile(latencies, 0.99):.4f}")
    print(f"max_ms: {max(latencies):.4f}")
    print(f"under_50ms: {sum(1 for latency in latencies if latency <= 50) / total:.2%}")
    print()

    print("=== Failed Cases ===")
    failed = [row for row in rows if not row["ok"]]
    if not failed:
        print("none")
    else:
        for row in failed[:30]:
            print(
                f"#{row['id']} q={row['query']} pred={row['pred_domain']} "
                f"expected={row['expected_domain']} score={row['score']} "
                f"time={row['time_found']}/{row['expected_time']} corrected={row['corrected']}"
            )
        if len(failed) > 30:
            print(f"... {len(failed) - 30} more")

    print()
    print("=== Sample Outputs ===")
    for row in rows[:10]:
        print(
            f"#{row['id']:03d} domain={row['pred_domain']:<10} "
            f"score={row['score']:<4} time={row['time_found']} "
            f"latency_ms={row['latency_ms']:.4f} corrected={row['corrected']}"
        )


if __name__ == "__main__":
    evaluate()
