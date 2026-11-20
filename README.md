# pairwise-uvwrap

## 题面（可直接粘贴）

纹理 UV 环绕与各向异性 mip 选择。约束：repeat 取小数部分、clamp 限幅、mirror 镜像，负坐标与超大坐标正确；必须实现 `wrap_uv_derivatives(u, v, du, dv, mode)`：返回环绕后的 uv 与导数，mirror 下导数符号翻转、repeat 下保持，并正确处理跨整数边界的接缝（接缝处导数不得出现跳变）；必须实现 `mip_level(du, dv, tex_w, tex_h, levels)`：按最大拉伸比取 log2，钳制到 [0, levels) 并返回三线性插值权重，各向异性拉伸时不得退回各向同性；10 万次调用不超过 100 ms；非法模式、非有限输入、tex 尺寸为 0 要给出可区分的错误；全部函数无状态且并发安全。验收：python -m pytest tests -q -k wrap。改动限 2 到 3 个文件，增量构建加测试不超过 30 秒。不得改动既有测试与既有行为。交付给命令、输出、耗时三行自测证据。

## 边界要求（评审侧隐藏用例判分）

- 非有限输入
- 并发
- 超大坐标
- 模式枚举
- 镜像边界

## 运行方式

```bash
python -m pytest tests -q
```

初始 15 failed。可见测试只覆盖主功能（3 分线）。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
