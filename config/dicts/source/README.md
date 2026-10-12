# config/dicts/source — 词典源数据归档

> 2026-10-12 建。目的：让「DUT 转换规则误伤 ~4,500 词」结论（《12》§9）及其数字**可独立复核**——
> 源数据 + 转换脚本 + 挡词清单三件套齐备后，该发现不再依赖任何临时文件。

## dutir_emotional.csv
- **内容**：大连理工大学情感词汇本体（DUTIR），**27,466 词** × 13 列
  （词语 / 词性种类 / 词义数 / 词义序号 / 情感分类 / 强度 / 极性 / 辅助情感分类 / 强度 / 极性 / …）
- **来源**：GitHub `yizhanmiao/DLUT-Emotionontology` → `情感词汇/情感词汇.csv`
  （经由 ghproxy 镜像拉取，2026-10-12；原链 raw.githubusercontent.com）
- **校验**：md5 `10d16e0ea013ff124eba973150b9c855` ｜ 1,495,588 字节
- **学术溯源**：徐琳宏, 林鸿飞, 潘宇, 等. 情感词汇本体的构造[J]. 情报学报, 2008, 27(2): 180-185.
- **用途**：`tools/build_dict_dut.py`（v2）的输入
- **编码**：UTF-8（无 BOM）；表头中文

## dut_skipped.tsv
- v2 规则下被挡的 **1,048** 条（原因列：`dual(3)` 兼有 / `polarity0_cat_unknown` 极性 0 且类别不明 / parse_error）
- 说明：v1 规则（只收极性 1/2）挡掉 ~5,500 条，其中 **4,355** 条已由 v2「极性 0 按类别补方向」补回；
  本清单为 v2 视角的**最终排除项**。复现 v1 全量挡词 = 原始 CSV + git 历史版脚本（`build_dict_dut.py@fb7c5a8`）

## 复现指引
```bash
cd ~/projects/danmaku-v2
# v2 转换（当前生效）
python3 tools/build_dict_dut.py            # → config/dicts/dut_sentiment.tsv
# v1 对照（挡词复现）
git show fb7c5a8:tools/build_dict_dut.py > /tmp/build_v1.py && python3 /tmp/build_v1.py   # 需 SRC 指向本目录
```
