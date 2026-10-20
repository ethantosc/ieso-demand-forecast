# Ontario Hourly Electricity Demand Forecasting — v2 Spec

**Status:** pre-registration · 2026-10-20 · written and committed before any v2 model code
**Owner:** Ethan Chang
**Builds on:** [`spec.md`](spec.md) (v1.1) and the v1 results in [`../reports/findings.md`](../reports/findings.md)

---

## 0. 為什麼要先寫這份

v2 的每一個改動都是看了 v1 的結果才想到的(例如 LightGBM 在夏天輸給 MSTL + ETS)。如果用 v1 看過的評估窗來評 v2,就是在測試集上做選擇——跟碩論的 look-ahead 問題同一族,只是發生在評估層而不是特徵層。

所以這份文件在寫任何 v2 模型程式**之前**定下:在哪裡評分、哪些選擇可以看資料做、看哪一段資料做、以什麼標準判定。git 的 commit 時間就是證據。

---

## 1. 評估窗

| 窗 | 期間 | 用途 | 乾淨嗎 |
|---|---|---|---|
| **開發 dev** | 2023-01-01 – 2023-12-31 | **唯一**可以看結果做選擇的地方:門檻是哪一個、組合權重 | — |
| **保留 holdout** | 2026-01-01 – 2026-06-13 | v2 的正式成績 | ✅ v1 從未對這段評分 |
| **前瞻 prospective** | 凍結日之後的每一個交付日 | 正式成績,含對 IESO 官方預測 | ✅ 設計時資料尚不存在 |
| **已看過 seen** | 2024-01-01 – 2025-12-31;2026-06-14 – 凍結日 | 跟 v1 對照 | ❌ 不作結論依據 |

**holdout 的限制,先寫清楚:** 只有冬季與春季,沒有夏季;也沒有 IESO 官方預測(收割封存從 2026-06-14 才開始)。乾淨的夏季測試要等前瞻窗走到 2027 年夏天。v1 曾用 2026 年 1–5 月的資料**訓練**官方對照期的模型,以及在 EDA 看過 2026 年的年平均與尖峰——這些都不是評分,不影響 holdout 的乾淨度。

**凍結規則(機械化):** 凍結日 = `src/` 底下最後一次會影響分數的 commit 的日期(`git log -1 -- src ':!src/evaluate/plots.py'`),由程式在執行時讀取並印在報告裡。前瞻窗 = 交付日 > 凍結日。**任何改動模型、特徵或評估邏輯的 commit 都會自動把前瞻窗重新歸零。** 只改圖或報告文字不會。

---

## 2. 評估規則(所有項目共用)

**門檻:** 在**開發窗**上比較合法 naive 與 seasonal naive,MAE 較低者成為所有窗的門檻。v1 規格把 seasonal naive 定為門檻,但 v1 結果顯示合法 naive 更強;v2 不在測試窗上挑,而在開發窗上挑。

**階梯順序(依設計固定,不依分數):**
門檻 → MSTL + ETS → MSTL + ARIMA → LightGBM A → 組合 A → LightGBM C → 組合 C

軌道 B(實測氣溫)是參考上界,不在階梯裡。

**顯著性:** Diebold-Mariano(日 MAE、Newey-West 變異數),檢定家族在每個窗內固定為:
1. 每一階對上一階(相鄰)
2. 每一階對門檻
3. 指定比較:C 對 A_2021(預報天氣的價值)、組合 A 對 LightGBM A、組合 A 對 MSTL + ETS、最佳可營運模型對 IESO(前瞻窗)

報原始 p 值與 **Holm 校正後** p 值;結論只依 Holm 校正後 p < 0.05。

**指標:** MAE(主)、RMSE、MAPE、ME(= 預測 − 實際)。common sample:只比所有模型都有預測的小時。

**超參數:** 沿用 v1(`src/models/lgbm.py`),不在任何窗上調。

**v1 的設計沿用:** LightGBM 預測週變化、輸入不含水準;資料截止 D-1 11:00 EST;交付日以 EST 計。v1 的擴張窗勝過 3 年窗,v2 的 LightGBM A 只跑擴張窗——這是依 v1 結果做的選擇,所以只能由 holdout 和前瞻窗來檢驗。

---

## 3. 天氣預報:軌道 C

**資料:** [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api),NCEP GFS global(**不用 seamless**:seamless 會混入 HRRR,而 HRRR 的封存起點不同,歷史會不一致)。三個機場、等權平均,與軌道 B 相同。授權 CC BY 4.0,非商業免費。

**合法性規則(本節最重要):**

`temperature_2m_previous_dayN` 在有效時刻 t 的值,來自約在 t 之前 24·N 小時起跑的那一次模型(GFS 每 6 小時一次:00/06/12/18Z)。在 origin(D-1 11:00 EST = 16:00 UTC),最新可用的一次是 **D-1 06Z**(起跑後約 4–6 小時發布;12Z 那次要到 16–18Z 才出來,不算)。

| D 的 EST 小時 h | 有效時刻 t(UTC) | 用 | 對應的模型起跑 | lead |
|---|---|---|---|---|
| 0–6 | D 05Z – 11Z | `previous_day1` | D-1 00Z / 06Z | 24–29 h |
| 7–23 | D 12Z – D+1 04Z | `previous_day2` | D-2 12Z – D-1 00Z | 48–53 h |

推導:day1 合法 ⇔ 起跑 ≤ D-1 06Z ⇔ t < D 12Z ⇔ h ≤ 6。day2 對所有 h 都合法。代價:h ≥ 7 的預報比真正可用的 06Z 那次舊 6–18 小時,所以軌道 C 是**保守**的(IESO 用得到更新的預報)。

**驗證(過不了就停):** 規則中「day N ≈ 24·N 小時前起跑」是從文件推論的。執行前用 [Single Runs API](https://open-meteo.com/en/docs/single-runs-api)(可指定起跑時刻)抽樣比對:把 previous_day1 / day2 的每個值,去找哪一次起跑的預報與它相同,統計 lead 分布。**通過條件:≥ 95% 的抽樣值落在預期 lead(day1 24–29 h,day2 48–53 h)。** 不通過 → notebook 停止。Single Runs API 拿不到 GFS 時 → 全部小時改用 `previous_day2`(依文件自己的定義必然合法),並在報告標明。

**訓練也用預報氣溫**,不用實測——訓練用完美天氣、預測用有誤差的天氣,模型會過度相信氣溫。訓練起點 = GFS 封存中第一個覆蓋率 ≥ 95% 的月份(文件說 2021 年 3 月)。

**對照組(同一訓練起點,才能隔離效果):**
- `A_2021`:軌道 A,同起點 → **C − A_2021 = 預報天氣的價值**
- `B_2021`:實測氣溫,同起點 → **C − B_2021 = 預報誤差的代價**
- 回收比例 = (A_2021 − C) / (A_2021 − B_2021)

**特徵:** 與軌道 B 相同的天氣特徵,全部由預報序列計算;週變化 = 預報 − 一週前的預報。

---

## 4. 組合:LightGBM + MSTL + ETS

`組合 = w · LightGBM + (1 − w) · MSTL+ETS`,w ∈ {0, 0.1, …, 1},**在開發窗上**取 MAE 最低者(平手取較小的 w)。兩個變體,事先宣告:
- **全域 w**
- **分季 w**:夏季(6–8 月)一個、其餘一個

對 LightGBM A 與 LightGBM C 各做一組。不做其他組合。權重凍結後套用到所有窗。

---

## 5. Ontario Demand 與出口分解

v1 的假設:夜間誤差(尤其 IESO 的 hour-ending 24)來自出口,而非省內用電。

**做法:** 目標改成 Ontario Demand,以及出口 = Market − Ontario,各自用合法 naive、seasonal naive、LightGBM A(擴張窗)預測,在 holdout 與 seen 回測窗上評估。

**判定標準(事先宣告,在 holdout 上用 LightGBM A):**
1. 夜間(hour-ending 24、1–5)出口誤差佔「出口 + Ontario」誤差的比例,高於白天(hour-ending 8–20);**且**
2. 夜間出口的平均 MAE ≥ Ontario Demand 的平均 MAE

兩者成立 → 「夜間誤差主要來自出口」得到支持。**這只能檢驗我們的模型;** IESO 的預測無法拆成兩部分,所以 IESO 的夜間弱點只能說「與此一致」或「不一致」,不能證明。

---

## 6. 機率預測

**模型:** quantile LightGBM,q ∈ {0.05, 0.25, 0.5, 0.75, 0.95},軌道 C、2021 起點(天氣預報不可用時改軌道 A),目標同樣是週變化;分位數交叉時逐列排序修正。

**基準(階梯精神):** 門檻模型 + 該模型過去 365 天殘差的經驗分位數(依 EST 小時分組,每月重算,只用 origin 之前的資料)。

**指標:** 平均 pinball loss(主)、50% 與 90% 區間的實際覆蓋率、90% 區間平均寬度、每個分位數的實際落在其下的比例(可靠度)。

---

## 7. 執行

- 一個 notebook:`notebooks/03_v2.ipynb`,Run all,約 60–80 分鐘
- 預測結果依(commit, 窗, 模型)快取,中途斷線重跑會接續
- 輸出:`reports/v2/findings_auto.md`、`reports/v2/results/`、`reports/v2/figures/`
- v1 的 `src/evaluate/run.py` 保持可重現 v1 結果

---

### 7.1 首次執行前的修正(2026-10-20,尚未有任何真實結果)

用合成資料試跑後估計 Colab 上要約 95 分鐘,超出原訂的 60–80 分鐘。以下兩項不影響本規格的任何問題,刪掉:

- **機率預測只在 holdout 與官方封存窗評估**,不在 seen 回測窗上跑(第 6 節本來就沒有要求 seen 窗)
- **LightGBM B(2003 起)不在 seen 回測窗重跑**——v1 已經報過它在那裡的成績;軌道 C 的對照組 `B_2021` 照常在所有窗執行

## 8. v2 不做

- 負載加權氣溫(維持等權,與 v1 可比)
- 「昨天的預測誤差」當特徵——合法做法需要巢狀的滾動樣本外預測,成本高,留給 v3
- 神經網路、分區需求、電價

---

## 9. 完成的定義

1. holdout 上的完整階梯、DM + Holm、五維分解
2. 軌道 C 的三個量:預報天氣價值、預報誤差代價、回收比例
3. 組合權重(開發窗)與 holdout 成績
4. 出口分解與第 5 節判定標準的結果
5. 機率預測的 pinball、覆蓋率與基準比較
6. 前瞻窗的計數與(累積足夠時的)IESO 對照
7. `findings.md` 的 v2 章節,只引用 `reports/v2/findings_auto.md` 的數字
