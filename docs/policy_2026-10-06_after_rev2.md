# rev2 結果を受けた新方針：N128 の速度競争を終え、「保証付き圧縮壁帯」へ主張を移す

作成日：2026-10-06
前提資料：moment-dense 変種 I rev2 の同一入力検証（2026-10-06）、SUMMARY.json、RESULT.json、VARIANT_RECORD.json、VARIANT_DETAILS.json、`docs/decision_2026-10-06.md`
付属：`scripts/toy_cross_oversampling.py`、`docs/toy_cross_oversampling_*.json`

---

## 0. 決定

1. **N128 での計算速度の競争は終える。** 同じ dense 配列の native 全更新が 0.077 秒で測られた。TT 経路の NumPy 下限（面評価 4 回、source 準備、構築、丸め）は 0.3 秒を下回らず、r≈25・N128 では TT 代数の flop 数自体が dense と同程度である。どの構築法・どの検査契約でも native dense には届かない。これは finding として会計つきで記録する。
2. **構築は moment-dense を維持し、per-step 費用を決定的にする二つの修正を入れる。** (a) Eq の fixed cross の pivot は前回受理 F1 ではなく前回 Eq TT から継承する。合成場では F1 由来 pivot が 1e-5〜1e-7 で失敗し過剰標本化でも回復しないのに対し、Eq 由来 pivot は 1.7e-9 で現在 Eq 自身の pivot と同等だった（2 節）。(b) two-site warm TCI を fallback から外す。9.38 秒のうち oracle は 0.14 秒で、残りは 1 要求あたり 0.82 μs の機構費用である。fallback は構造付き randomized TT-SVD（合成場 0.33 秒）とする。
3. **研究の主張を「保証付き圧縮壁帯」へ移す。** 速度ではなく、SVD 台帳で全域 1e-8 を保証した TT 壁帯が 32 更新以上の軌道を payload 1.47 MB（dense 113 MB の 1/77）で保つこと、および N を上げたときの rank・payload・費用の振る舞いを主張の中心にする。検査契約は変種 II（f_eq 面検証＋丸め台帳＋H 毎更新＋面端到端 k 更新ごと）を明文化し、変種 I と並べて測る。
4. **停止条件を 3.4 節に固定する。** Eq pivot 継承でも fixed cross が面 y2 の velocity guard を落とすなら cross を捨てる。32 更新で fixed 成功率が 9 割を切るなら軌道主張を取り下げる。N256 で rank が 1.5 倍以上増えるなら N 非依存仮説を棄却する。

---

## 1. rev2 結果の読み

### 1.1 費用会計（保存 B step2、各 1 更新）

| 区間 | 秒 |
|---|---:|
| moment 前計算（4 moment、1,963,008 値、予約穴 768） | 0.037 |
| source 準備（packet 埋込 0.110、sum/round 2 回） | 0.167 |
| Eq fixed cross（F1 pivot、fiber scalars 165,521、条件数 6.9e11） | 0.074 |
| Eq warm two-site（11.46M 要求、oracle 計 0.14、機構 9.24） | 9.382 |
| F 組立＋丸め（fixed / warm） | 0.037 / 0.030 |
| F fresh 検証（2 面＋S_f、1 試行あたり） | 0.398 / 0.387 |
| 完全 attempt（両 fit、F/P/H、stream、history） | 10.877 |
| 従来法 fresh 共有 B の完全 attempt | 0.869 |
| 同じ dense 配列の NumPy 全更新（中央値） | 0.642 |
| 同じ dense 配列の native 全更新（中央値） | **0.077** |

### 1.2 読み取り

- **構築は問題でなくなった。** moment 0.037 ＋ Eq fixed 0.074 ＋ 組立 0.037 = 0.15 秒で、完成 F は全域 F 相対 L2 3.80e-9、H 3.51e-9、stream 全域 3.61e-9 を満たした。oracle の安さは確認された。
- **fixed cross の失敗は pivot の出自による。** F1 pivot での Eq は面 y2 で population 相対 3.06e-8、velocity 相対 3.33e-5 で guard 2e-5 を超えた。面 y2 は velocity の参照 norm が population の約 1/460 で、velocity guard が population 誤差を約 1,090 倍に増幅する。Eq の面 y2 population 誤差は 1.8e-8 以下が必要で、normalizer の Eq 相対許容 1.63e-8 と同程度である。S_f の 3.7e-8 や面 y31 の通過は、この面には効かない。
- **two-site TCI は fallback に不適。** 11.46M 要求、固有空間点 310,300（帯の 63%）、cache 70〜120 MB。oracle を O(1) にしても機構費用が 0.82 μs/要求で残る。
- **fixed が成功しても更新は約 0.85 秒で、従来法 0.87 秒と同じ。** 残る費用は検査 0.40（47%）、source 準備 0.17（20%）、stream・history・P/H。検査の内訳は source 面評価 0.064〜0.068、candidate 面評価 0.065〜0.079、metric 0.031〜0.035 が 2 面分で、BGK 参照は 0.011。面評価は演算量（rank 25/18 で約 25 Mflop、数 ms）の 10〜20 倍で、NumPy の経路費用が主である。
- **native dense 0.077 秒が基準なら、TT 経路は構造的に届かない。** 面評価だけで 4 回 × 数十 ms、source 準備、構築、丸めを NumPy で最善に実装しても 0.3 秒程度。r=25、N=128 では、丸め O(n R² r) と面評価 O(N² r) の flop 数が dense の O(27 N² N_y) と同桁である。TT の優位は flop ではなく、保存容量と、N を上げたときの定数にしか現れない。

---

## 2. 合成場での追加確認：pivot の出自

`scripts/toy_cross_oversampling.py`。擾乱 81% の 4 モード場で、現在状態と前回状態を位相の進み（ステップ数）で作り、現在の Eq を fixed cross で構築した。Eq 相対許容は研究側 normalizer の 1.63e-8。両状態がせん断で傾いた場合（現在 10 ステップ、前回 9 または 7 ステップ）の結果：

| pivot の出自 | pivot rank | Eq 相対誤差（前回 9） | Eq 相対誤差（前回 7） | 面 y2 population / velocity（前回 9） | 判定 |
|---|---|---:|---:|---|---|
| 前回 F（完全 population、rank 上限 25/16） | 25/16 | 1.1e-5 | 8.3e-6 | 9.5e-6 / 7.3e-4 | 不合格 |
| 前回 F、過剰標本化 +4〜+12、pinv | 29〜37 / 18〜21 | 1.8e-5〜2.4e-6 | 3.6e-6〜9.1e-7 | 2.3e-6 / 1.6e-4 | 不合格、改善せず |
| **前回 Eq TT** | 43/13 | **1.8e-9** | **1.7e-9** | 1.1e-9 / 1.3e-7 | 合格 |
| 現在 Eq 自身（上限性能） | 43/13 | 1.7e-9 | 1.7e-9 | 1.0e-9 / 1.2e-7 | 合格 |

現在が phase0 の場で前回を 1 ステップ進めた場でも同じ傾向で、前回 F pivot は 9.3e-8、前回 Eq pivot は 2.3e-14 だった。前回 F の pivot は非平衡と前時刻の構造に最適化されており、現在 Eq の部分空間を 1e-8 で張らない。pivot 行列の条件数は 1e18〜1e26 で、pinv の打切りでは救えない。rev2 の 3.06e-8 は合成場の 1e-5 より良いが、同じ機構で失敗している。

限界：合成場であり、実際の前回 Eq TT（rev2 では warm で得た rank 23/13）の pivot が次更新で同じ性能を出すかは研究側の保存入力で確認する。

---

## 3. 新方針の詳細

### 3.1 N128 を finding として閉じる

記録する内容：1.1 節の会計、native と NumPy の dense、従来法と moment-dense の構築費用、検査費用の内訳、two-site TCI の機構費用、fixed cross の失敗機構と pivot 出自の効果。結論は「N128・r≈25 では TT 壁帯は計算を加速しない。構築費用は dense の 1/4 以下まで下がったが、検査と TT 代数の flop が dense と同桁であり、native dense には届かない」。これは実装の不足ではなく規模と rank の関係による。

### 3.2 構築の修正（費用の決定性のため、速度主張のためではない）

1. **Eq pivot の継承。** 各更新で受理した Eq TT の pivot（左右の入れ子集合）を保存し、次更新の fixed cross に使う。初回だけ warm または randomized TT-SVD で Eq を作る。rank は Eq の rank（23/13 程度）で、F1 の 25/16 より少ないか同程度。期待：fixed 成功、Eq 0.07 秒。
2. **fallback の置換。** two-site TCI を外し、dense moment からの構造付き randomized TT-SVD（10 本の (N_x × N_zN_y) 行列との batched matmul、k=r+15、power 1、合成場 0.33 秒）を fallback にする。誤差は厳密 f_eq との面比較で直接測る。
3. **source 準備の縮約。** 組立を f* = [(1−ω) bulk + ω Eq] + (1−ω)(packet + ghost) と分け、境界項を split-q の明示平面のまま次段へ渡す。sum/round を 3 項 2 回から 2 項 1 回に減らす。期待：0.167 → 0.05 秒。split-q が (1−ω)(packet+ghost) を明示平面として streaming へ運べない場合は見送る。
4. **面評価の行列積化。** source・candidate の面評価を core3 の面スライス縮約と core1×core2 の積の 2 段にまとめる。契約・数式・対象点は不変。期待：1 面 0.065 → 0.01 秒。

4 点を入れた B 更新の見込みは、構築 0.15 ＋ source 0.05 ＋ 検査 0.15 ＋ stream/history/P/H 0.1 ≒ 0.45 秒。NumPy dense 比 0.7、native dense 比 6。この数値は仮説で、目的は 9 秒級の外れ値を消して軌道費用を予測可能にすることにある。

### 3.3 主張の再設定：保証付き圧縮壁帯

**主張 A（精度と容量）。** B 方式で 32 更新、合格後 128 更新。各更新で f_eq の面検証（厳密 f_eq 対 TT、全 F 面）、丸め δ の台帳、H 256 点の fresh 端到端比較を行い、面の端到端比較は k=8 更新ごと。全更新の累積 velocity・壁 shear・ledger・kernel は現行通り。報告量：全域 F 相対誤差の台帳上界と実測、payload、rank 推移、fixed 成功率、更新費用の分布。変種 I（現契約）を同じ入力で並走させ、保証の差と費用の差を表にする。

**主張 B（N 依存性）。** N256（面内）、可能なら N512、帯厚 32、同じ 4 モード場。測るもの：Eq と F の rank、payload、更新費用の内訳、dense native の時間とメモリ。仮説：rank は N に依存しない（合成場で確認可能）、payload ∝ N、更新費用は O(N²) だが定数が dense の 1/5〜1/10。moment-dense は Πn を触るので O(N) ではないことを明記する。O(N) 経路（moment を spatial TT のまま pivot fiber 上だけで評価する）は N512 以降の課題として分離する。RSS 1280 MiB では N512 の dense は 1 配列 1.8 GB で in-core 不可であり、out-of-core の dense か「不可」を対照として報告する。資源契約（suffix cap 8 MiB、pair/local 上限、workspace）の改定は別登録で行う。

**失うもの（変種 II）。** 和・丸め・面組立・streaming 経路の実装不良を毎更新の面で検出する能力。H 点（毎更新）と k 更新ごとの面で部分補償する。独立第二経路の頻度低下として記録する。閾値・P/H・ledger・kernel・finite・保存量拒否・rollback は変えない。

### 3.4 停止条件

- 3.2-1 の Eq pivot 継承でも、保存入力で fixed cross が面 y2 の velocity guard を落とす → cross を捨て、randomized TT-SVD を既定構築にする。それでも 0.5 秒を超えるなら構築法の改良を止める。
- 32 更新で fixed 成功率 < 90%、または更新費用の最大/中央値 > 3 → 費用の決定性なし。軌道主張を取り下げ、1.1 節とともに負の結果として記録する。
- N256 で Eq rank が N128 の 1.5 倍以上 → N 非依存仮説を棄却。主張 B を取り下げ、主張 A のみ残す。
- 変種 II で全域 F の実測が台帳上界を超える → 台帳の定義不備。変種 I へ戻し原因を特定するまで変種 II を使わない。

### 3.5 やらないこと

two-site TCI の改良、F1 pivot の過剰標本化、凍結基底と基底拡張、N128 のさらなる速度最適化、R 省略、許容値緩和、finite 範囲縮小、観測者台帳の書換え、H2、quantics。

---

## 4. 実行順序

1. **保存 B step2 で Eq pivot 継承の確認。** 1 更新前の Eq TT を一度 warm または randomized TT-SVD で作って保存し、その pivot で現在の Eq を fixed cross する。費用、Eq 相対誤差、面 y2 の population と velocity、pivot 条件数を記録する。
2. **fallback の置換と小 fixture。** randomized TT-SVD の誤差を厳密 f_eq との面比較で確認し、two-site を経路から外す。
3. **source 準備の縮約と面評価の行列積化。** 契約不変。受理値・finite・例外・offset・最終候補の所属が同じ入力で一致することを確認する。
4. **変種 II 契約の明文化。** 保証の増減を表にし、k を固定する。
5. **32 更新 B 方式、変種 I と II。** dense 対照は NumPy と native を同条件で再計測し、両方を報告する。
6. **N256 の資源契約改定と 8 更新試験。** rank・payload・費用の内訳。合成場で N256 の Eq rank を先に測り、仮説を事前に書く。

各段階で停止条件に触れたら次へ進まない。速度比だけで結論しない。初期構築込み総時間、受理更新時間、失敗候補の費用、監査の外側時間を分けて記録する。
