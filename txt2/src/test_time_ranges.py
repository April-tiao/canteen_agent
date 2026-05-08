from __future__ import annotations

from datetime import date

from text_processing import extract_time


BASE_DATE = date(2026, 4, 27)


CASES = [
    {
        "query": "上周二食堂退款记录",
        "start": "2026-04-20",
        "end": "2026-04-26",
        "granularity": "week",
    },
    {
        "query": "本周午餐菜单",
        "start": "2026-04-27",
        "end": "2026-05-03",
        "granularity": "meal_period",
    },
    {
        "query": "下周早餐有什么",
        "start": "2026-05-04",
        "end": "2026-05-10",
        "granularity": "meal_period",
    },
    {
        "query": "昨天晚餐扣款异常",
        "start": "2026-04-26",
        "end": "2026-04-26",
        "granularity": "meal_period",
    },
    {
        "query": "今天12点半二食堂还有牛肉面吗",
        "start": "2026-04-27",
        "end": "2026-04-27",
        "granularity": "time_point",
    },
    {
        "query": "上个月饭卡消费明细",
        "start": "2026-03-01",
        "end": "2026-03-31",
        "granularity": "month",
    },
    {
        "query": "本月充值记录",
        "start": "2026-04-01",
        "end": "2026-04-30",
        "granularity": "month",
    },
    {
        "query": "上个季度食堂投诉统计",
        "start": "2026-01-01",
        "end": "2026-03-31",
        "granularity": "quarter",
    },
    {
        "query": "本季度订单数量",
        "start": "2026-04-01",
        "end": "2026-06-30",
        "granularity": "quarter",
    },
    {
        "query": "去年食堂消费总额",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "granularity": "year",
    },
    {
        "query": "今年退款记录",
        "start": "2026-01-01",
        "end": "2026-12-31",
        "granularity": "year",
    },
]


def main() -> None:
    failed = []
    for case in CASES:
        result = extract_time(case["query"], today=BASE_DATE)
        if result is None:
            failed.append((case["query"], "None"))
            continue
        for key in ("start", "end", "granularity"):
            if result.get(key) != case[key]:
                failed.append((case["query"], key, case[key], result.get(key), result))

    if failed:
        print("FAILED")
        for item in failed:
            print(item)
        raise SystemExit(1)

    print("time range tests passed")
    for case in CASES:
        result = extract_time(case["query"], today=BASE_DATE)
        print(f"{case['query']} -> {result['start']} - {result['end']}")


if __name__ == "__main__":
    main()
