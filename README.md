# Creative Agent Skills

OpenCodeから、1枚のNVIDIA GPUをllama.cpp/Qwen 3.8とComfyUI/MiniMax H3で安全に切り替えて使うためのAgent Skillです。

主な機能は次のとおりです。

- LLMと動画生成の排他的なGPU/VRAM管理
- Qwen 3.8によるMiniMax H3プロンプト作成
- MiniMax H3のT2Vおよび1枚画像R2Vワークフロー生成・実行
- 動画生成後のComfyUI解放とQwen復帰

## 必要な環境

- Linux、NVIDIA GPU 1枚、`nvidia-smi`
- Python 3.10以降、PyYAML
- 任意: `nvidia-ml-py`（未導入時は`nvidia-smi`を使用）
- llama.cpp / llama-server
- ComfyUIとMiniMax H3の必要なモデル・カスタムノード
- OpenCode

## 設定

環境に合わせて次のファイルを編集してください。

- `model.ini`: GGUFのパスとllama.cppのコンテキスト長
- `opencode.json`: llama-serverのURLとOpenCodeモデル設定
- `skills/gpu-runtime-manager/config.yaml`: VRAM要件、ComfyUI/llama-serverのURL、タイムアウト

このリポジトリの`opencode.json`は、`./skills`からSkillを検出し、Qwen 3.8のreasoning effortを`medium`に設定します。

## 使い方

リポジトリのルートからOpenCodeを起動してください。Agentは通常、`skills/gpu-runtime-manager/SKILL.md`を読み、以下の高水準コマンドを使用します。

状態確認:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py status
```

LLMを利用可能にする:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py acquire llm
```

動画生成を安全に実行してLLMへ戻す:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py run-video \
  --prompt-file /tmp/minimax-h3-prompt.txt \
  --width 864 --height 480 --duration 3 \
  --output-prefix video/creative_agent
```

動画側を解放する:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py release video
```

変更を伴わず予定だけ確認する場合は`--dry-run`、機械可読な結果が必要な場合は`--json`を各サブコマンドの末尾へ追加します。

R2V、OpenCode添付画像、エラー時の安全規則などの詳細は[`skills/gpu-runtime-manager/SKILL.md`](skills/gpu-runtime-manager/SKILL.md)を参照してください。

## テスト

```bash
python -m unittest discover -s skills/gpu-runtime-manager/tests -p 'test_*.py'
```

## License

[MIT](LICENSE)
