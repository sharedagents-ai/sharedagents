# Agent logo sources

`pi.svg` is the unmodified square badge from [Pi's official press kit](https://pi.dev/press-kit), downloaded from <https://pi.dev/favicon.svg> on 2026-09-24.

`zcode.png` is the square product mark from the official ZCode site, cropped from <https://zcode.z.ai/images/hero-visual/zcode-logo@2x.png> on 2026-10-01. The wordmark at <https://z-cdn.chatglm.cn/z-ai/static/logo.svg> is the same Z, but it is a wide lockup and does not fit the picker tile.

`devin.png` is the square app icon from the official Devin site, downloaded from <https://devin.ai/icon.png> on 2026-10-26. The brand-kit symbol at <https://devin.ai/assets/brand/logos/devin/symbol-light.svg> is the same three-hexagon mark in black ink for light backgrounds; on the dark picker that ink disappears, so the tile uses the dark app icon instead.

`qwen.svg` is the unmodified official Qwen Code mark shipped **inside the product itself**: `web-shell/assets/icon.svg` of `@qwen-code/qwen-code` (copied from the locally installed 0.27.0 global package on 2026-11-08). Using the package's own asset keeps the tile byte-identical to what the CLI and its web-shell render, and needs no external download. The mark is the blue (#082DFF) Qwen glyph on transparency with a square viewBox, so it renders correctly on the dark picker with `fit: contain`.
