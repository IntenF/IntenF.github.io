# AI Agent Guidelines (AGENTS.md)

このリポジトリ（IntenF.github.io）で作業を行うAIエージェントに向けた特別なガイドラインです。

## プロジェクトのアーキテクチャ
- **目的**: リードラグ投資戦略（PCA SUB λ=0.9）の毎日のシグナル生成とダッシュボード表示。
- **Pythonバックエンド**: `.github/scripts/update_signals.py` がYFinanceを用いて日次で動作し、最新の投資結果と翌日のシグナルを `signals.json` に出力します。
- **GitHub Actions**: `.github/workflows/daily_update.yml` により、毎日米国市場終了後（日本時間午前6時等）に自動で `.github/scripts/update_signals.py` が実行され、`signals.json` がリポジトリへ自動コミット・プッシュされます。
- **フロントエンドダッシュボード**: `investment.html` は静的HTMLおよびCDN上のtailwindcss, Chart.jsを用いており、ローカル実行（file:///）ではCORSポリシーにより `signals.json` をロードできないため、動作確認には `python -m http.server 8000`等を推奨します。

## ⚠️ 重要: Git操作時のコンフリクト回避ルール
AIエージェントが**新しいスクリプトやUIを変更してプッシュしようとする際**、高確率でリモートリポジトリ（GitHub Actionsによって自動更新された `signals.json` のコミット）と競合（Reject）が発生します。

通常の `git pull` や `git pull --rebase` を行うと `signals.json` 同士でマージコンフリクトが起きやすいため、本リポジトリで作業内容を Push する際のアプローチは以下の手順を推奨します：

```bash
# 1. リモートの最新情報を取得（リベースはしない）
git fetch

# 2. 作業中のファイル（PythonスクリプトやHTML）の中身はそのままに、Gitの履歴だけをリモートに追従させる
git reset origin/master

# 3. エージェントが手元で生成・更新した最新のコードおよびsignals.jsonをすべてaddする
git add .

# 4. コミットして強制プッシュを避けてそのままPush
git commit -m "Update logic and signals"
git push
```
この手順により、GitHub Actionsが裏で作成した `signals.json` の差分を吸収しつつ、エージェントが作成した機能追加や新しいJSONスキーマを安全に本番反映することができます。

## 留意事項
米国市場の開場時間中（例えば日本時間の夜〜深夜）に手動でシグナル生成スクリプトを走らせる場合、YFinanceの最新日に「不完全な日中データ（Incomplete day）」が含まれるため、Python側で開場時間を判定し（現在16:00 EST前等）、不完全データを「Drop」して前日確定値で算出するロジックが組み込まれています。運用時はこれに抵触しないよう留意してください。
