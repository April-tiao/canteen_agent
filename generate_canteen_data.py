from __future__ import annotations

import json
from pathlib import Path

from canteen_gateway_demo import correct_query


OUTPUT_FILE = Path(__file__).with_name("canteen_test_data_300.json")


TIME_PREFIXES = [
    ("今天中午", True),
    ("明天早餐", True),
    ("后天午饭", True),
    ("昨天晚餐", True),
    ("周三中午", True),
    ("下周", True),
    ("下午一点", True),
    ("12点半", True),
    ("今晚", True),
    ("", False),
]


IN_DOMAIN_TEMPLATES = [
    "{time}二食堂有{dish}吗",
    "{time}一食堂{dish}多少钱",
    "{time}三食堂{dish}还有吗",
    "{time}午餐有什么素食",
    "{time}窗口排队太久怎么评价",
    "{time}饭卡余额怎么查",
    "{time}帮我冲值二十块",
    "{time}这笔订单可以退欵吗",
    "{time}二食唐开门吗",
    "{time}三食唐麻辣堂档囗还营页吗",
    "{time}能预定{dish}吗",
    "{time}帮我开发飘",
    "{time}餐卡充值不到账",
    "{time}这张支付截图扣款失败",
    "{time}{dish}能少辣吗",
    "{time}清真窗口在哪里",
    "{time}牛肉面档口几点关门",
    "{time}取餐码在哪里看",
    "{time}能不能打包",
    "{time}饭咔支付失败怎么办",
]


DISHES = [
    "宫爆鸡丁",
    "鱼香肉丝",
    "番茄炒蛋",
    "鸡腿饭",
    "牛肉面",
    "麻辣堂",
    "黄焖鸡",
    "沙县小尺",
    "拉面",
    "水饺",
    "麻婆豆腐",
    "红烧肉",
]


OUT_DOMAIN_TEMPLATES = [
    "{time}股市怎么样",
    "{time}天气如何",
    "{time}帮我写一首诗",
    "{time}推荐一款手机",
    "{time}生成Python代码",
    "{time}这道数学题怎么做",
    "{time}写篇论文介绍智慧食堂",
    "{time}设计一个食堂海报",
    "{time}智能餐饮行业趋势",
    "{time}食堂经营模式有哪些",
    "{time}帮我订酒店",
    "{time}有什么电影",
    "{time}王者荣耀鸡腿皮肤多少钱",
    "{time}翻译这句话",
    "{time}去上海旅游攻略",
    "{time}电脑蓝屏怎么修",
    "{time}基金适合长期持有吗",
    "{time}帮我写英语作文",
    "{time}会议议程怎么安排",
    "{time}手机怎么截图",
]


def compact(text: str) -> str:
    return text.replace(" ", "")


def make_item(query: str, domain: str, has_time: bool) -> dict[str, object]:
    normalized = compact(query)
    corrected, _ = correct_query(normalized)
    return {
        "q": normalized,
        "domain": domain,
        "time": has_time,
        "corrected": corrected,
    }


def generate() -> list[dict[str, object]]:
    data: list[dict[str, object]] = []

    for i in range(180):
        time_text, has_time = TIME_PREFIXES[i % len(TIME_PREFIXES)]
        template = IN_DOMAIN_TEMPLATES[i % len(IN_DOMAIN_TEMPLATES)]
        dish = DISHES[i % len(DISHES)]
        query = template.format(time=time_text, dish=dish)
        data.append(make_item(query, "in_domain", has_time))

    for i in range(120):
        time_text, has_time = TIME_PREFIXES[i % len(TIME_PREFIXES)]
        template = OUT_DOMAIN_TEMPLATES[i % len(OUT_DOMAIN_TEMPLATES)]
        query = template.format(time=time_text)
        data.append(make_item(query, "out_domain", has_time))

    return data


def main() -> None:
    data = generate()
    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(data)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
