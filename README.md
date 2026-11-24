# pairwise-ui-ninepatch-batch

## 领域

游戏、图形与多媒体

## 主题

界面九宫格与合批

## 说明

DPI 切分、图集稳定与双缓冲顶点。

初始状态下 `graphics/ui_ninepatch_batch.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
