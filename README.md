# pairwise-wet-surface-rain

## 领域

游戏、图形与多媒体

## 主题

雨天湿地面

## 说明

湿度分层、按遮罩反射与蒸发过渡。

初始状态下 `render/wet_surface_rain.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
