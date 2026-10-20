# Ontario Hourly Electricity Demand Forecasting — Spec

**Status:** v1.1 · 2026-10-19(v1 draft 2026-09-06)
**Owner:** Ethan Chang

---

## v1.1 變更紀錄(相對 v1)

| § | 變更 | 原因 |
|---|---|---|
| 1 | 資料截止定為 D-1 11:00 EST 之前開始的小時;預測日 = IESO 交付日(EST,固定 24 小時) | `CreatedAt` 實測 127 天中 125 天為 12:31,全部在交付前一天;時區未標示,取較嚴格解讀 |
| 2.1 | 1994–2002 檔移出 v1 範圍 | Market Demand 在市場開放(2002-05-01)前不存在 |
| 2.2 | 「夏季尖峰省份」改寫;測站定案 | 25 年中 6 年年尖峰在冬季;ECCC 小時資料為 LST = 固定 EST |
| 2.3 | 封存現況;只收 DATotals | Adequacy3 量太大已移除;DATotals 實際保留期遠超 30 天 |
| 3 | lag 24h 只在目標小時 < 11 時合法;新增無水準特徵 | 原定義在 12:30 origin 下洩漏 13 個小時 |
| 5 | naive 改為合法版本;ME 正負號統一;第 3–4 階實作定案;加 DM 檢定 | 同上;§1 與碩論的 ME 慣例相反 |
| 6 | Repo 結構對齊實際 | 多了 `scripts/`、`src/analysis/`、`docs/`;預測封存進 git |
| 7 | 執行環境改 Colab;不做 streamlit/Quarto | MacBook Air 不裝 Python;報告以 GitHub 渲染的 Markdown 呈現 |
| 8 | 里程碑狀態 | |

---

## 0. 這份專案的目的(先寫清楚,避免中途漂移)

這是一個**技能訓練專案**,不是研究貢獻。

- **主要目的**:用非微軟的 Python 工程棧找回建模手感,特別是 baseline 比較、時序回測、殘差診斷這幾塊(上次認真做是 2022 年碩論)
- **次要目的**:補 DuckDB / Polars / statsforecast 這類工作上碰不到的工具
- **明確不是目的**:題目新穎性、發表、portfolio 展示

電力負載預測是被做過無數次的題目。**這是刻意的選擇**——參考資料多,卡住有得查,失敗成本低。

> 時間預算提醒:這個專案的節奏是「每週幾小時」,不取代其他優先事項。

---

## 1. 預測任務定義

### 目標變數

**Market Demand,小時級,單位 MW。**

IESO 的定義:Ontario Demand = Market Demand 減去 scheduled exports。

**為什麼用 Market Demand:** 拿 2026-06-14 的 DATotals 預測對實際值實測:

| 預測欄位 | 對照序列 | MAE | ME(偏誤) |
|---|---|---|---|
| Total Load | **Market Demand** | **475 MW** | **+61 MW** |
| Total Load | Ontario Demand | 2,911 MW | +2,911 MW |
| Total Energy | Market Demand | 631 MW | +455 MW |
| Total Energy | Ontario Demand | 3,305 MW | +3,305 MW |

官方日前預測建立在 Market Demand 上。要跟它比,目標變數就必須是 Market Demand。

> `Total Load = Total Energy − Total Loss`。10S / 10N / 30R 是運轉備轉容量,不是需求。

### 預測地平線與時鐘(v1.1 定案)

| 項目 | 定義 |
|---|---|
| 預測日 D | IESO 交付日:EST 日期,hour-ending 1–24,**永遠 24 小時** |
| 預測製作時點 | D-1 約 12:30(DATotals `CreatedAt` 127 天中 125 天為 12:31,全部在 D-1) |
| **資料截止** | **D-1 中開始時間早於 11:00 EST 的小時為已知,其餘皆未知** |
| 預測區間 | D 的 24 個小時;從最後已知小時算起的 lead time 為 hour-ending + 13 小時 |
| 時區 | 全程固定 EST;行為特徵(本地小時、星期、假日)用多倫多本地時間 |

`CreatedAt` 是 EST 還是 EDT 沒有標示。截止取較嚴格的解讀:嚴一小時只會讓我們的模型看起來稍差,鬆一小時就是洩漏。

> DST:交付日以 EST 計,所以預測單位沒有 23 / 25 小時日。本地時間的 23 / 25 小時日與 11 月重複小時仍存在,寫進 limitations。

---

## 2. 資料來源

### 2.1 IESO 實際需求 ✅

**2002-05-01 – 現在**,年度檔 `PUB_Demand_{year}.csv`

- 目錄:`https://reports-public.ieso.ca/public/Demand/`
- 3 行標頭、Hour 為 hour-ending 1–24、**固定 EST**(三種獨立方式驗證,見 `reports/data_quality.md` §3)

**1994–2002 檔:v1 不用。** 它只有市場開放前由 operational meters 推導的 Ontario demand;Market Demand 在 2002-05-01 前不存在。原「2002 接點落差」問題因此不成立,而非被量測。

### 2.2 天氣(Track B 專用)✅

Environment and Climate Change Canada 歷史小時資料(bulk CSV)。

- 測站:Toronto、Ottawa、London 機場;各城市新舊兩個測站 ID 依時間接續,新站優先;三城等權平均
- 時間基準:ECCC 小時資料為 **LST(本地標準時間,全年不調 DST)= 固定 EST**,與需求檔同基準,不需轉換
- 每個檔都驗證檔內測站名含預期城市名;錯的 ID 只會損失該城市該期間的覆蓋率,不會把別城天氣混進來
- 涵蓋率逐城逐年報在 `findings_auto.md`

**修正:安大略不是單純的夏季尖峰省份。** 25 年中有 6 年年尖峰在冬季(2004、2014、2015、2017、2018、2023)。夏季尖峰在本地 14–17 時,冬季在 17–18 時。

### 2.3 IESO 官方預測 ✅

**只收 `DATotals`**(日前計算引擎的每小時預期負載)。Adequacy3 試過後移除:每天最多約 90 個修訂版、每個約 194 KB,而且回答的是 35 天充裕度,不是日前負載。PredispTotals 未測試。

收割器(`.github/workflows/harvest.yml` + `scripts/harvest_forecasts.py`)每天比對遠端清單、只下載缺的檔,只保留無後綴的定版檔。**截至 2026-10-18:127 個連續交付日,2026-06-14 起,零缺口。** DATotals 的實際保留期明顯長於原先估的 30 天。

### 2.4 DATotals 檔案結構 ✅

```
Document
├── DocHeader
│   └── CreatedAt             ← 預測製作時間(D-1 約 12:31)
└── DocBody
    ├── DeliveryDate          ← 被預測的日子
    └── Energies
        └── HourlyEnergy × 24
            ├── DeliveryHour  ← 1–24,與需求 CSV 同慣例
            └── MQ × 7
                ├── MarketQuantity
                └── EnergyMW
```

兩個解析陷阱:(1)預設命名空間 `http://www.ieso.ca/schema`,不加前綴的 `find()` 回傳 `None` 不報錯——`src/ingest/forecasts.py` 改為缺元素就報錯;(2)檔名日期 = DeliveryDate,不是製作時間。

---

## 3. 資料處理

### 分層

```
data/
  raw/demand/      下載檔,不進 git,可重建
  raw/weather/     下載檔,不進 git,可重建
  raw/forecasts/   官方預測封存,**進 git**(不可重生)
  staging/         parquet
  curated/         預測結果
```

規則不變:原檔不改、全部落 Parquet、缺值不靜默填補(`data_quality.md` 記錄每一處)。2003 停電與節電期共 6 天以旗標排除,不刪資料。

### 特徵(v1.1)

**洩漏規則先行:** 目標小時 h 的 t-24 是 D-1 的 h 點,h ≥ 11 時在 origin 尚未發生。

| 類別 | 特徵 | 合法性 |
|---|---|---|
| 日曆 | EST 小時、本地小時、星期、月、年內日(sin/cos)、週末、法定假日、非法定但普遍放假(Civic Holiday、Easter Monday、Remembrance Day)、假日前一日 / 後一日 | 永遠合法 |
| 滯後 | lag 48 / 168 / 336 | 永遠合法(落在 D-2 以前) |
| 滯後 | lag 24 | **只在 h < 11 合法**,其餘為空 |
| Origin 摘要 | 最後已知值、截止前 24h / 168h 平均、D-1 已知時段對上週同時段比 | 截止當下計算 |
| 無水準版本 | lag 差值(48−168、168−336、24−168)、24h/168h 平均比、最後已知值週比 | 給 LightGBM 用,見 §5 |
| 天氣(B 軌) | 氣溫、HDD / CDD(基準 18°C)、日均 / 最高 / 最低、氣溫週變化 | 實測值 = 完美預知,只能當上界 |

`tests/test_features.py` 把截止後的每筆觀測改成垃圾值,驗證 D 日的任何特徵都不變,並有對照組證明測試不是空轉。

---

## 4. 洩漏規則(本專案最重要的一條)

**用實際氣溫去預測明天的負載,是資訊洩漏。** 營運上你在 origin 時點只有**預報**氣溫。

| 軌道 | 天氣輸入 | 定位 |
|---|---|---|
| **A — 乾淨軌** | 不用天氣 | 真正可營運的預測。主結果 |
| **B — 上界軌** | 實際氣溫(完美預知) | **不是預測,是上界** |

**兩軌的誤差絕對不可以並排比較後宣稱 B 比較好。** v1.1 補充:同一族的錯誤也出現在 v1 自己的 naive 與 lag 24 定義裡(§3、§5),已修正,並保留 `naive_leaky` 作為標明的診斷項,量化那個洩漏值多少 MW。

---

## 5. 評估協定

### Baseline 階梯(由下往上,不准跳)

1. **Naive(合法版)**:h < 11 用 D-1 同小時,否則用 D-2 同小時。*(v1 的 y(t-24) 會讀到 origin 時尚未公布的值)*
2. **Seasonal naive**:`ŷ(t) = y(t-168)` ← **門檻**
3. **MSTL + ETS**:24 與 168 雙季節分解,餘項用 AutoETS;每個 origin 用前 8 週重新擬合
4. **MSTL + ARIMA**:同上,餘項用 AutoARIMA。*(單一週期的 SARIMA 無法同時處理日與週兩個週期,period-168 的 SARIMA 也無法在每個 origin 合理時間內擬合)*
5. **LightGBM**:全域模型,24 個小時一起訓;目標為 y − y(t−168),輸入只用無水準特徵(理由見下)

**規則:第 5 步必須贏過第 2 步才有意義。** 贏不了就照實寫。

**為什麼 LightGBM 預測週變化、輸入不帶水準:** 樹模型無法預測訓練目標範圍外的值,而需求正在創新高(2026-07-02 為 2011 年以來最高)。只改目標還不夠:在內建上升趨勢的合成資料上,吃原始水準特徵的模型每個月都偏 −90 MW 左右——它在訓練期學到「水準高 = 即將回落」,整條序列上移後就一直預測不會發生的回落。改成無水準特徵後偏誤降到 ±5 MW 內。兩個決定都基於結構論證、在任何真實結果出現前定案,不是比分數挑的。

### 回測

- **Rolling origin**,每天一次預測,各自從自己的 12:30 origin 出發
- 古典模型每個 origin 重新擬合;LightGBM 每月重訓,訓練資料到該月第一天的前兩天為止
- 訓練窗:3 年滾動 vs 2003 年起的擴張窗,兩種都跑
- **回測期:2024-01-01 – 2025-12-31**(兩個完整年度)
- **官方對照期**:封存中所有有實際值的交付日(2026-06-14 起)
- 超參數固定在 `src/models/lgbm.py`,從未在評估窗上調整

### 指標

- 主指標 **MAE(MW)**;輔助 MAPE、RMSE、**ME**
- **ME = 預測 − 實際,正值 = 高估。** 與 §1 表格一致。*(注意:v1 提到碩論「ME 全為負值 = 系統性高估」,那是 實際 − 預測 的相反慣例。本 repo 一律用 預測 − 實際)*
- 只在所有模型都有預測的小時上比較(common sample)
- **Diebold-Mariano 檢定**(日 MAE、Newey-West 變異數):差距是不是真的,而不是雜訊

### 誤差分解

- 交付小時(hour-ending 1–24)
- 本地時鐘小時
- 季節(夏 / 冬 / 過渡季)
- 平日 / 週末 / 法定假日
- 極端溫度日(評估窗內最熱、最冷各 5% 的日子;需要天氣資料)

---

## 6. Repo 結構(v1.1 實際)

```
ieso-demand-forecast/
├── README.md
├── requirements.txt
├── .github/workflows/harvest.yml   每日收割(GitHub Actions)
├── scripts/harvest_forecasts.py
├── docs/spec.md                    本文件
├── data/
│   ├── raw/forecasts/              官方預測封存 ← 進 git
│   ├── raw/demand/  raw/weather/   .gitignore
│   ├── staging/                    .gitignore
│   └── curated/                    .gitignore
├── src/
│   ├── ingest/      fetch_demand, demand, check_demand, fetch_weather, forecasts
│   ├── transform/   features.py(洩漏邊界)
│   ├── models/      baselines.py, lgbm.py
│   ├── evaluate/    run.py, metrics.py, plots.py
│   └── analysis/    eda.py
├── notebooks/       00_ingest_colab, 01_eda, 02_results(只驅動,不放邏輯)
├── reports/
│   ├── data_quality.md / data_quality_auto.md
│   ├── findings.md / findings_auto.md
│   ├── results/  figures/
└── tests/
```

**`data/` 不能整個 ignore**:`raw/forecasts/` 是收割器 commit 的封存,忽略它會讓收割器安靜地失效。

---

## 7. 技術棧

| 用途 | 選擇 |
|---|---|
| 執行環境 | **Google Colab**(不在本機裝 Python);本機可用 `pip install -r requirements.txt` |
| 抓取 | 標準庫 `urllib`(收割器在 GitHub Actions 上零依賴) |
| 資料框 | `polars` |
| 查詢 | DuckDB(資料品質檢查,SQL over Parquet) |
| 古典模型 | `statsforecast` |
| ML | `lightgbm` |
| 假日 | `holidays` |
| 輸出 | **GitHub 渲染的 Markdown + PNG**;不做 streamlit / Quarto |

**刻意不用**:任何 Azure / Fabric / Power BI。

---

## 8. 里程碑

| # | 內容 | 狀態 |
|---|---|---|
| 0 | 官方預測收割器 | ✅ 2026-09-13 起每日運作,127 天零缺口 |
| 1 | 資料取得 | ✅ `data_quality.md` |
| 2 | EDA | ✅ `findings.md` §3、`notebooks/01_eda.ipynb` |
| 3 | 回測框架 | ✅ |
| 4 | Baseline | ✅ seasonal naive 1,275 MW;合法 naive 995 MW 反而更強 |
| 5 | 古典模型 | ✅ MSTL + ARIMA 834 MW |
| 6 | LightGBM 軌道 A | ✅ 801 MW,贏過門檻 37%(p < 0.0001) |
| 7 | LightGBM 軌道 B | ✅ 620 MW(上界) |
| 8 | 撰寫 | ✅ `reports/findings.md`;IESO 官方 470 MW |

> v1 結果對規格的一個修正:§5 把 seasonal naive 當「真正的門檻」,但安大略的需求「昨天」比「上週」更像今天,合法 naive 顯著勝過 seasonal naive。v2 的門檻應取兩者中較強者。

---

## 9. v1 明確不做

- ❌ LLM
- ❌ 機率預測 / 預測區間(v2)
- ❌ 分區需求
- ❌ 電價、發電組合、intertie
- ❌ 部署、CI/CD(收割器的每日排程是唯一例外,因為資料會過期)
- ❌ 神經網路——除非 LightGBM 已經贏過 seasonal naive

---

## 10. 完成的定義

1. 任何人 clone 之後能重建全部資料並重跑結果 → `notebooks/00` + `02`,或 README 的三行指令
2. README 講清楚:問題 → 資料 → 方法 → baseline → 結果 → 限制
3. 誤差表按 §5 的五個維度分解呈現 → `findings_auto.md`
4. 軌道 B 明確標示為上界 → 報告中每處都標 UPPER BOUND
5. Limitations 至少三項,含 DST、天氣預報缺口、測站代表性 → `findings.md` §7(八項)

---

## 附:待確認清單

- [x] ~~IESO 歷史小時需求的下載 endpoint~~ → §2.1
- [x] ~~IESO 歷史預測值是否封存~~ → 自己收割,§2.3
- [x] ~~ECCC 取得方式與測站~~ → §2.2
- [x] ~~`PUB_Demand.csv` 欄位、標頭、Hour 慣例~~ → 3 行標頭,hour-ending 1–24
- [x] ~~2002 前後接點~~ → 不成立,§2.1
- [ ] Market Demand 定義在 2002 年以來是否變過
- [ ] Ontario > Market 的 43 個小時原因
- [ ] 2024/2025 年度檔接縫恰為 0.0 MW 的原因
