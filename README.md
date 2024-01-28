在實驗室期間執行的兩大計畫：人口預測論文與小兒心臟M-mode圖即時量測計畫

1. 人口預測論文
老師請一頓黃金海岸，吃完就讓我們動筆寫這篇要投Entropy的論文。
論文主要任務為判斷何類模型較適合用於人口預測，
資料集：台灣六個縣市的每年人口(Y)和每年出生率、市民均收益、該年移入率、該年死亡率、城市該年稅收、城市年度預算
加上我共有四位同學分工合作跑Feature selection、LSTM、Linear regression、XGBoost、ARIMA。其中ARIMA是過往人口預測經常使用的統計模型，另外四個模型則是採用AI機器學習方法，來做比較。
我負責跑XGBoost、ARIMA和統整撰寫論文。
論文連結：https://www.semanticscholar.org/reader/ae85c6c3b255caa61b2e051cc3e82860f4c3a142

2. 小兒心臟M-mode圖即時量測計畫
'檔案處理.ipynb'是與高雄長庚醫院合作的小兒心臟M-mode圖即時量測計畫中，我負責的部分。
我負責M-mode圖片前處理、OS讀取檔案、OpenCV切取圖片、OCR光學辨識字元、並採用正則化方式取出各特徵數值放進不同column。
