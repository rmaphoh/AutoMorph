# AutoMorph_fork

[日本語](#日本語) | [English](#english)

## 日本語

[AutoMorph](https://github.com/rmaphoh/AutoMorph)（コミット [`9a953e5`](https://github.com/rmaphoh/AutoMorph/tree/9a953e5edfa419b454e9fb7b06235eb413828d05)）の M3（形態指標の計算）を、原著の定義に合うように書き直した非公式の版です。AutoMorph の開発者による版ではありません。

- 元の AutoMorph のコードはこのフォルダの外にそのまま残してあり、変更していません。
- M0〜M2（前処理、画質判定、血管の分割）は変更していません。元の AutoMorph の出力をそのまま入力に使います。
- 変更前の計算の出どころは、多くが元にしたライブラリ [retipy](https://github.com/alevalv/retipy-python) です。AutoMorph が加えた変更が含まれる箇所は下の表に記しました。

### 改正点

変更はすべて 2026-11-02 に行いました。ID の A は原著の定義や数値計算との違い、B は値に偏りが出る性質、C は動かない箇所または主な値に影響しない箇所です。「元のコード」のリンク先は、変更前のコードの該当行です。Zone 用のコードにも同じ内容があります。

| ID | 対象 | 元のコード | この版 | 根拠 |
|:---|:---|:---|:---|:---|
| A1 | 点の順序（蛇行度3種） | 幅優先探索の訪問順をそのまま曲線とする。始点が血管の途中だと、両側を交互にたどる。x 順の並べ替えは AutoMorph で外され、コメントが残る（[L583][a1c]、[L603-617][a1d]）（[retina.py L584-625][a1]） | 血管の端から順に並べる。単純な経路でない区間は除く | 蛇行度は血管に沿った曲線の量である [1,3]。真の弧長÷弦長が 1.053 の合成曲線で、最大 158 になった |
| A2 | 曲率の蛇行度 | 符号つきの曲率を積分し、最後に絶対値をとる（[tortuosity_measures.py L332-333][a2]） | κ² を弧長で積分する | Hart 1999 の total squared curvature は ∫κ²dt、total curvature は ∫\|κ\|dt である（p.244）[1]。\|∫κdt\| はどちらでもない。弧長での積分は Grisan 2003 の式(15)と同じ形である [2]。元の式では S 字の血管が打ち消し合う |
| A3 | 2階差分 | 間隔1の2階差分を4で割る（[math.py L56][a3]） | 4で割らない | 数値計算の違いで、原著には現れない。間隔1の中心差分の分母は1である。曲率が4分の1になる |
| A4 | 蛇行密度の式 | `(n-1)/n + (1/L)Σ(…)`（[tortuosity_measures.py L314][a4]） | `(n-1)/n × (1/L)Σ(…)` | Grisan 2008 の式(19) [3]。2003 年版の式(14)も掛け算である [2]。元の式では値がほぼ (n-1)/n で決まる |
| A5 | 蛇行密度の n | 変曲点の数。最後の変曲点より後の部分を計算に入れない（[tortuosity_measures.py L299-312][a5]） | 曲率の符号が一定の部分の数。血管全体を計算に入れる | Grisan 2008 の式(19) [3] |
| A6 | 変曲点 | y の増減の変化を数える（[tortuosity_measures.py L138-143][a6]） | スプラインの曲率の符号の変化。しきい値 0.01 を超えた区間の間の中点で分ける | 変曲点は曲率の符号が変わる点であり、増減の変化は極値である。しきい値と中点は Grisan 2003（p.868）、2008（p.314）[2,3]。元の計算は、真の変曲点が 0〜3 個の合成曲線で最大 372 個を検出した |
| B1 | 血管幅（M3） | 面積÷骨格の画素数（[tortuosity_measures.py L73][b1]） | 面積÷中心線の長さ。元の値も `width_pixelcount` として出力する | 斜めの1画素も長さ1と数えるので、幅がおよそ W/cosθ になる。45度の合成の帯で +30〜68%、この版では −8〜+19% だった |
| B2 | 曲線の表現 | 画素の点の列をそのまま用いる | 共通の平滑化スプラインを当ててから計算する | Grisan 2003, 2008 も平滑化スプラインを用いる [2,3]。画素の階段がそのまま長さと曲率になるのを防ぐ |
| B3 | CRAE/CRVE の幅 | 円環で切ったマスクで測る（[tortuosity_measures.py L370-414][b3]） | 切る前のマスクで測る | 円環の縁で血管が切れ、幅が小さく出る（合成画像で真の値の約 0.95〜0.97 倍） |
| B4 | CRAE/CRVE の幅 | 各区間の最後の点を除く（[L375][b4]） | 全ての点を用いる | 除く理由がなく、点の順序で結果が変わる |
| B9 | CRAE/CRVE の単位 | 912 画素上の値に `Scale_resolution` を掛ける。この値には1000が掛かっている（[EyeQ_process_main.py L39][b9]） | 912 画素上の値を出力する | 実寸の情報がないときは、この倍率に意味がない |
| C1, C2, C5 | 使われない関数 | `fractal_tortuosity` は AttributeError、`_curve_to_image` は TypeError で止まる。`distance_inflection_count_tortuosity` は出力されない（[L21][c1a]、[L29][c1b]、[L180][c2]、[L271][c5]） | 削除 | 出力に関わらない |
| C3 | 範囲の確認 | `len(y) - 1 <= target <= 0` は有効な入力で常に偽になる（[math.py L41][c3a]、[L54][c3b]） | 修正 | 端の点で例外を出すようにした |
| C4 | 欠測 | 裸の `except:` で −1 にする | 6本に満たないときだけ NaN。他の例外は止める | 例外が −1 という値に化けるのを防ぐ。NumPy 2 で `np.trapz` が消えたときに起きる |

A2〜A5 は retipy のコミット [`89cccd3`](https://github.com/alevalv/retipy-python/commit/89cccd3)（2018-03-07）から、A6 は [`313df3d`](https://github.com/alevalv/retipy-python/commit/313df3d)（2017-11-13）からあります。

このほか `retina.py` では、Windows のパス区切りを置換し、交点を探す二重ループを `np.where` に置き換えました。後者は走査の順序を変えず、計算を速くするためのものです。

### Pull request #19 との関係

A1 の点の順序の問題は、[rmaphoh/AutoMorph の Pull request #19](https://github.com/rmaphoh/AutoMorph/pull/19)（staskh、2026-08-11 に作成、2026-11-03 時点で未マージ）が先に報告しています。この版は #19 のコードを含みません。違いは次のとおりです。

| 項目 | Pull request #19 | この版 |
|:---|:---|:---|
| 点の順序 | 端点から順にたどる（`order_as_path`） | 端点から順にたどる（`order_vessel_pixels`） |
| 分岐のある区間 | 最長の経路を残す | 除く |
| 閉じた輪 | 最小の座標の画素から順にたどる | 除く |
| 分割のしきい値 | 0.5 → 0.2 に変更 | M2 を変更しないので 0.5 のまま |
| 区間の下限の比較 | `>=` に変更 | `>` のまま |
| A2〜A6、B1〜B4、B9 | 扱っていない | 上の表のとおり |

M2 の骨格では、分岐点を先に消しています。手元の M2 の骨格で数えた範囲では、この版が除いた区間はありませんでした。したがって、分岐と輪の扱いの違いは、この範囲では結果に影響していません。

### 変更していない点

- **M2 の血管幅:** M2 の CSV の `Vessel_Width_*` は骨格の画素数のままです。この版の血管幅は `run_m3_fork.py` の出力を使ってください。
- **距離法の −1（B10）:** Hart 1999 の τ1 は `s/chord − 1` で直線が 0 です [1]。この版も元の AutoMorph と同じく −1 なしで出力するので、直線は 1 になります。順位は変わりません。
- **FD（B6, B7）:** 全部が血管で埋まった箱を数えない定義と、912 の格子はそのままです。
- **血管密度:** 向きで変わらず、真の値に一致したので変更していません。
- **円を広げて幅を測る方法（B5）:** 手法の変更になるのでそのままです。向きによって階段状にずれます。合成画像で CRAE/CRVE は真の値の約 1.05 倍に出ます。
- **Knudtson の式:** 定数 0.88, 0.95 と手順は Knudtson 2003 の式(2)(3)と一致し、論文の計算例（134.4 µm）を再現しました [4]。変更していません。
- **CRAE/CRVE の6本の選び方（B8）:** 血管でなく区間から選ぶままです。

### 設定値

| 名前 | 値 | 決め方 |
|:---|---:|:---|
| `SMOOTHING` | 0.35 | 合成の曲線で候補10個を比べ、直線に偽の曲率が出なくなる最小の値。臨床のデータは用いていない |
| `HYSTERESIS` | 0.01 | Grisan 2003 の静脈の値（動脈は 0.03）[2] |
| `MIN_PIXELS` | 15 | 元の AutoMorph の `retipy.config` の値（retipy は 8） |

### 検証と限界

合成の曲線（直線、円弧、正弦波。太さと縁のギザつきを変えたもの）で、真の値と比べました。

| 指標 | この版の結果 |
|:---|:---|
| 距離法（弧長÷弦長） | 誤差の中央値 0.0007、最大 0.014 |
| total squared curvature | 真の値の 0.91〜1.22 倍（中央値 1.006） |
| 変曲点の数 | 93.8% で一致 |

- `tests/test_fork.py` の単体テストは8件あります。合成の曲線の検証スクリプトはこのリポジトリに含めていません。
- 実際の血管の蛇行度には正解がないので、実画像での正確さは確かめていません。
- 平滑化の強さは原著と同じではなく、しきい値は動脈にも静脈の値を用います。原著の値を厳密に再現する版ではありません。
- 論文の査読を受けた版ではありません。

### 使い方

Python 3.10 以上、NumPy 2.0 以上、SciPy、OpenCV、scikit-image、pandas、joblib、matplotlib、Pillow が必要です。リポジトリの直下で実行します。

```bash
python AutoMorph_fork/tests/test_fork.py
```

```bash
python AutoMorph_fork/run_m3_fork.py --results Results --out automorph_fork_m3.csv
```

`--results` には元の AutoMorph の `Results` フォルダを指定します。読み込むだけで書き込みません。出力の列は次のとおりです。

| 列 | 内容 |
|:---|:---|
| `{vessel,artery,vein}_fd`, `_vd` | FD、血管密度 |
| `{vessel,artery,vein}_width`, `_width_pixelcount` | 面積÷中心線の長さ、面積÷骨格の画素数 |
| `{vessel,artery,vein}_dist`, `_tsc`, `_dens` | 距離法、total squared curvature、蛇行密度 |
| `{vessel,artery,vein}_mean_abs_curv`, `_n_seg` | 平均絶対曲率、区間の数 |
| `zone{B,C}_{artery,vein}_cre` | CRAE/CRVE（912 画素上の画素単位） |

### ライセンス

AutoMorph 本体は Apache License 2.0（[LICENSE](LICENSE)）、retipy 由来の部分は GPL-3（[LICENSE_retipy_GPL3.md](LICENSE_retipy_GPL3.md)）です。`M3_feature/retipy` と `M3_feature/function_` は retipy を改変したもので、GPL-3 に従います。この表が変更点の記載を兼ねます。

このコードを用いたときは、元の AutoMorph の論文 [5] を引用してください。

## English

An unofficial version of M3 (morphology measurement) of [AutoMorph](https://github.com/rmaphoh/AutoMorph) (commit [`9a953e5`](https://github.com/rmaphoh/AutoMorph/tree/9a953e5edfa419b454e9fb7b06235eb413828d05)), rewritten to follow the definitions in the original papers. It is not released by the AutoMorph developers.

- The original AutoMorph code remains unchanged outside this folder.
- M0–M2 (preprocessing, image quality grading, and vessel segmentation) are unchanged. The output of the original AutoMorph is used as the input.
- Most of the replaced computations come from the underlying library, [retipy](https://github.com/alevalv/retipy-python). The table notes where AutoMorph's own changes are involved.

### Changes

All changes were made on 2026-11-02. In the IDs, A marks differences from the original definitions or from standard numerical computation, B marks properties that bias the values, and C marks code that does not run or does not affect the main values. The links under "Original code" point to the lines before the change. The zone code has the same content.

| ID | Item | Original code | This version | Evidence |
|:---|:---|:---|:---|:---|
| A1 | Point order (3 tortuosity measures) | The breadth-first visiting order is used as the curve. If the start is in the middle of a vessel, the order alternates between the two sides. Sorting by x was removed in AutoMorph, and the comments remain ([L583][a1c], [L603-617][a1d]) ([retina.py L584-625][a1]) | Points are ordered from one vessel end to the other. Segments that are not simple paths are excluded | Tortuosity is a quantity of the curve along the vessel [1,3]. A synthetic curve with a true arc/chord of 1.053 reached up to 158 |
| A2 | Curvature tortuosity | The signed curvature is integrated, and the absolute value is taken at the end ([tortuosity_measures.py L332-333][a2]) | κ² is integrated over arc length | Hart 1999 defines total squared curvature as ∫κ²dt and total curvature as ∫\|κ\|dt (p.244) [1]. \|∫κdt\| is neither. Integration over arc length follows eq. (15) of Grisan 2003 [2]. With the original formula, the two halves of an S-shaped vessel cancel out |
| A3 | Second difference | The second difference with unit spacing is divided by 4 ([math.py L56][a3]) | No division by 4 | A numerical difference that does not come from the papers. The denominator of a central difference with unit spacing is 1. Curvature becomes one quarter |
| A4 | Tortuosity density formula | `(n-1)/n + (1/L)Σ(…)` ([tortuosity_measures.py L314][a4]) | `(n-1)/n × (1/L)Σ(…)` | Eq. (19) of Grisan 2008 [3]. Eq. (14) of the 2003 paper is also a product [2]. With the original formula, the value is determined almost entirely by (n-1)/n |
| A5 | n in tortuosity density | The number of inflection points. The part after the last inflection point is not counted ([tortuosity_measures.py L299-312][a5]) | The number of parts with constant curvature sign. The whole vessel is counted | Eq. (19) of Grisan 2008 [3] |
| A6 | Inflection points | Changes in the direction of y are counted ([tortuosity_measures.py L138-143][a6]) | Sign changes of the spline curvature, split at the midpoint between parts exceeding a threshold of 0.01 | An inflection point is where the curvature changes sign, whereas a change in the direction of y is an extremum. The threshold and midpoint follow Grisan 2003 (p.868) and 2008 (p.314) [2,3]. The original computation detected up to 372 points on synthetic curves with 0–3 true inflection points |
| B1 | Vessel width (M3) | Area ÷ skeleton pixel count ([tortuosity_measures.py L73][b1]) | Area ÷ centerline length. The original value is also output as `width_pixelcount` | A diagonal pixel counts as length 1, so the width becomes about W/cosθ. On synthetic bands at 45 degrees, the error was +30–68%, compared with −8 to +19% in this version |
| B2 | Curve representation | The pixel chain is used directly | A common smoothing spline is fitted first | Grisan 2003 and 2008 also use smoothing splines [2,3]. This prevents pixel staircases from entering length and curvature directly |
| B3 | CRAE/CRVE width | Measured on the mask cut by the annulus ([tortuosity_measures.py L370-414][b3]) | Measured on the uncut mask | Vessels are cut at the annulus edge, which lowers the width (about 0.95–0.97 of the true value on synthetic images) |
| B4 | CRAE/CRVE width | The last point of each segment is dropped ([L375][b4]) | All points are used | There is no reason to drop it, and the result depends on point order |
| B9 | CRAE/CRVE unit | The value at 912 pixels is multiplied by `Scale_resolution`, which itself includes a factor of 1000 ([EyeQ_process_main.py L39][b9]) | The value at 912 pixels is output | This factor has no meaning without physical scale information |
| C1, C2, C5 | Unused functions | `fractal_tortuosity` raises AttributeError and `_curve_to_image` raises TypeError. `distance_inflection_count_tortuosity` is not output ([L21][c1a], [L29][c1b], [L180][c2], [L271][c5]) | Removed | They do not affect the output |
| C3 | Range check | `len(y) - 1 <= target <= 0` is always false for valid input ([math.py L41][c3a], [L54][c3b]) | Fixed | End points now raise an exception |
| C4 | Missing values | A bare `except:` returns −1 | NaN only with fewer than 6 vessels. Other exceptions stop the run | This prevents exceptions from turning into a value of −1, as happens when `np.trapz` is missing in NumPy 2 |

A2–A5 date from retipy commit [`89cccd3`](https://github.com/alevalv/retipy-python/commit/89cccd3) (2018-03-07), and A6 from [`313df3d`](https://github.com/alevalv/retipy-python/commit/313df3d) (2017-11-13).

In addition, `retina.py` now normalizes Windows path separators, and the nested loop that searches for intersections was replaced with `np.where`. The latter keeps the scan order and only speeds up the computation.

### Relation to pull request #19

The point order issue (A1) was reported earlier in [pull request #19 of rmaphoh/AutoMorph](https://github.com/rmaphoh/AutoMorph/pull/19) (staskh, opened on 2026-08-11 and not merged as of 2026-11-03). This version does not contain the code of #19. The differences are:

| Item | Pull request #19 | This version |
|:---|:---|:---|
| Point order | Traced from an endpoint (`order_as_path`) | Traced from an endpoint (`order_vessel_pixels`) |
| Segments with branches | The longest path is kept | Excluded |
| Closed loops | Traced from the pixel with the smallest coordinates | Excluded |
| Segmentation threshold | Changed from 0.5 to 0.2 | Remains 0.5, because M2 is unchanged |
| Comparison for the minimum segment length | Changed to `>=` | Remains `>` |
| A2–A6, B1–B4, and B9 | Not addressed | As in the table above |

Branch points are removed from M2 skeletons beforehand. In the M2 skeletons we counted, this version excluded no segments. The different handling of branches and loops therefore did not affect the results in that range.

### What was not changed

- **M2 vessel width:** `Vessel_Width_*` in the M2 CSV still uses the skeleton pixel count. Use the output of `run_m3_fork.py` for the width of this version.
- **The −1 of the distance measure (B10):** τ1 of Hart 1999 is `s/chord − 1`, which is 0 for a straight line [1]. Like the original AutoMorph, this version outputs the value without −1, so a straight line gives 1. Ranks are unaffected.
- **FD (B6, B7):** The definition that does not count boxes filled entirely with vessel, and the 912 grid, are unchanged.
- **Vessel density:** It did not vary with orientation and matched the true value, so it is unchanged.
- **The expanding-circle width method (B5):** Unchanged, because replacing it would change the method. It shifts in steps with orientation. On synthetic images, CRAE/CRVE are about 1.05 times the true value.
- **The Knudtson formula:** The constants 0.88 and 0.95 and the procedure match eqs. (2) and (3) of Knudtson 2003, and the worked example in the paper (134.4 µm) is reproduced [4]. Unchanged.
- **Selection of the 6 vessels for CRAE/CRVE (B8):** They are still selected from segments, not whole vessels.

### Settings

| Name | Value | How it was chosen |
|:---|---:|:---|
| `SMOOTHING` | 0.35 | The smallest of 10 candidates that removed false curvature on straight lines, compared on synthetic curves. No clinical data were used |
| `HYSTERESIS` | 0.01 | The value for veins in Grisan 2003 (0.03 for arteries) [2] |
| `MIN_PIXELS` | 15 | The value in `retipy.config` of the original AutoMorph (8 in retipy) |

### Validation and limitations

Values were compared with the truth on synthetic curves (lines, arcs, and sine waves, with varying thickness and rough edges).

| Measure | Result of this version |
|:---|:---|
| Distance measure (arc/chord) | Median error 0.0007, maximum 0.014 |
| Total squared curvature | 0.91–1.22 times the true value (median 1.006) |
| Number of inflection points | Matched in 93.8% |

- `tests/test_fork.py` contains 8 unit tests. The validation script for the synthetic curves is not included in this repository.
- Real vessels have no ground truth for tortuosity, so accuracy on real images has not been established.
- The smoothing strength differs from that in the original papers, and the threshold for veins is also used for arteries, so this version does not strictly reproduce their values.
- This version has not been peer reviewed.

### Usage

Python 3.10 or later, NumPy 2.0 or later, SciPy, OpenCV, scikit-image, pandas, joblib, matplotlib, and Pillow are required. Run from the repository root.

```bash
python AutoMorph_fork/tests/test_fork.py
```

```bash
python AutoMorph_fork/run_m3_fork.py --results Results --out automorph_fork_m3.csv
```

`--results` is the `Results` folder of the original AutoMorph. It is only read, never written. The output columns are:

| Column | Content |
|:---|:---|
| `{vessel,artery,vein}_fd`, `_vd` | FD, vessel density |
| `{vessel,artery,vein}_width`, `_width_pixelcount` | Area ÷ centerline length, area ÷ skeleton pixel count |
| `{vessel,artery,vein}_dist`, `_tsc`, `_dens` | Distance measure, total squared curvature, tortuosity density |
| `{vessel,artery,vein}_mean_abs_curv`, `_n_seg` | Mean absolute curvature, number of segments |
| `zone{B,C}_{artery,vein}_cre` | CRAE/CRVE (in pixels at 912 pixels) |

### License

AutoMorph itself is under the Apache License 2.0 ([LICENSE](LICENSE)), and the parts derived from retipy are under GPL-3 ([LICENSE_retipy_GPL3.md](LICENSE_retipy_GPL3.md)). `M3_feature/retipy` and `M3_feature/function_` are modified from retipy and follow GPL-3. The table above serves as the statement of changes.

Please cite the original AutoMorph paper [5] when using this code.

## References / 文献

1. Hart WE, Goldbaum M, Côté B, Kube P, Nelson MR. Measurement and classification of retinal vascular tortuosity. *Int J Med Inform*. 1999;53(2-3):239-252. [doi:10.1016/s1386-5056(98)00163-4](https://doi.org/10.1016/s1386-5056(98)00163-4)
2. Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic evaluation of retinal vessel tortuosity. *Proc 25th Annu Int Conf IEEE Eng Med Biol Soc*. 2003:866-869. [doi:10.1109/IEMBS.2003.1279902](https://doi.org/10.1109/IEMBS.2003.1279902)
3. Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic grading of retinal vessel tortuosity. *IEEE Trans Med Imaging*. 2008;27(3):310-319. [doi:10.1109/TMI.2007.904657](https://doi.org/10.1109/TMI.2007.904657)
4. Knudtson MD, Lee KE, Hubbard LD, Wong TY, Klein R, Klein BEK. Revised formulas for summarizing retinal vessel diameters. *Curr Eye Res*. 2003;27(3):143-149. [doi:10.1076/ceyr.27.3.143.16049](https://doi.org/10.1076/ceyr.27.3.143.16049)
5. Zhou Y, Wagner SK, Chia MA, et al. AutoMorph: automated retinal vascular morphology quantification via a deep learning pipeline. *Transl Vis Sci Technol*. 2022;11(7):12. [doi:10.1167/tvst.11.7.12](https://doi.org/10.1167/tvst.11.7.12)

[a1]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/retina.py#L584-L625
[a1c]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/retina.py#L583
[a1d]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/retina.py#L603-L617
[a2]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L332-L333
[a3]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/math.py#L56
[a4]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L314
[a5]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L299-L312
[a6]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L138-L143
[b1]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L73
[b3]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_zone/retipy/retipy/tortuosity_measures.py#L370-L414
[b4]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_zone/retipy/retipy/tortuosity_measures.py#L375
[b9]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M0_Preprocess/EyeQ_process_main.py#L39
[c1a]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L21
[c1b]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L29
[c2]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L180
[c5]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/tortuosity_measures.py#L271
[c3a]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/math.py#L41
[c3b]: https://github.com/rmaphoh/AutoMorph/blob/9a953e5edfa419b454e9fb7b06235eb413828d05/M3_feature_whole_pic/retipy/retipy/math.py#L54
