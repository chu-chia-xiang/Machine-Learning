# 機器學習實驗與作業總覽

Lab 1｜線性迴歸與 Ridge Regression

- 以 NumPy 從零實作線性迴歸模型，使用梯度下降訓練，並加入 L2 正規化形成 Ridge Regression，搭配 MSE 評估與 loss 曲線觀察收斂。
- 實作 Ridge 的閉形式解，計算 predictive mean / variance，畫出預測曲線與不確定性區間，比較不同模型的擬合能力。
- **功能重點：** 線性迴歸、L2 正規化（Ridge）、梯度下降 vs 閉形式解、MSE 與預測區間視覺化。

---

Lab 2｜Logistic Regression 與 ROC 分析

- 用 NumPy 從零實作 Logistic Regression，自己寫 sigmoid、梯度下降訓練、`predict_proba` 與 `predict`，在二元分類資料集上做實驗。
- 針對不同 learning rate / iteration 組合訓練模型，利用混淆矩陣計算 Accuracy、Precision、Recall、F1。
- 透過改變 decision threshold 繪製 ROC 曲線並計算 AUC，比較不同超參數設定下的分類表現與錯誤型態。
- **功能重點：** Logistic Regression、混淆矩陣與各種評估指標、ROC／AUC、threshold 對分類結果的影響。

---

Lab 3｜全連接深度神經網路（DNN）分類 MNIST

- 以 NumPy 從零實作多層前饋神經網路（含兩層 hidden layer），搭配 ReLU、tanh、Softplus、Leaky ReLU 等激活函數，以及 Softmax + Cross-Entropy loss。
- 在 MNIST 多類別分類任務上，改變 hidden layer 大小與 activation 組合，比較對測試準確率與收斂行為的影響。
- 使用混淆矩陣、ROC 曲線與 Precision/Recall/F1/Accuracy 分析各數字類別的分類效果與常見誤判。
- **功能重點：** DNN 從零實作、各種激活函數比較、多類別 Softmax 分類、混淆矩陣與多指標評估。

---

Lab 4｜多種 SGD 型優化器比較

- 在 MNIST 二元分類上，用 NumPy 分別實作 Mini-batch SGD、SGD with Momentum、Nesterov Momentum、Adam 四種優化器。
- 對同一模型與資料集進行訓練，觀察不同優化器的收斂速度與 loss 震盪情形。
- 紀錄測試集準確率與錯分樣本，分析 learning rate、batch size、momentum / β₁β₂ 等超參數對訓練穩定度與結果的影響。
- **功能重點：** 多種梯度下降變形、學習曲線比較、超參數敏感度分析、訓練穩定性與收斂速度評估。

---

Lab 5｜Regularization：Weight Decay 與 Early Stopping

- 以一層 hidden layer 的 MLP（ReLU + Softmax）做 MNIST 二元分類，實作 L2 weight decay（在 loss 中加入 \(\lambda \|W\|_2^2\)）與 Early Stopping 兩種正規化方法。
- 在反向傳播中加入 L2 梯度項，並在訓練迴圈中引入 validation set 與 patience 機制，自動偵測過擬合點停止訓練。
- 透過多組 \(\lambda\) 與早停設定，畫出 train / val loss & accuracy 曲線，比較不同正規化強度與組合對測試準確率與過擬合程度的影響。
- **功能重點：** L2 正規化、Early Stopping、訓練與驗證曲線分析、控制模型複雜度與過擬合。

---

Lab 6｜卷積神經網路（CNN）分類貓狗圖片

- 使用 Keras/TensorFlow 建立資料前處理流程（重採樣、標準化、資料集切分），並以 Conv2D + MaxPooling 堆疊出 baseline CNN。
- 以 binary cross-entropy + Adam 進行訓練，記錄訓練與驗證的 accuracy / loss 曲線，觀察是否過擬合或欠擬合。
- 進一步加入更深的卷積層、Batch Normalization、Dropout、資料增強與 EarlyStopping，改善模型泛化能力。
- 以測試集結果、混淆矩陣與 classification report（precision/recall/F1）評估模型表現與常見誤分類案例。
- **功能重點：** CNN 模型設計與微調、影像前處理與資料增強、正規化技巧（BN／Dropout／EarlyStopping）、分類效能評估。

---

Lab 7｜手刻 LSTM 模型分類 MNIST 序列

- 不使用現成 `nn.LSTM`，以 PyTorch 自行實作 LSTM cell：包含 forget/input/output gate 與 candidate cell 的前向運算。
- 將 28×28 影像視為 28 步長的序列輸入 LSTM，完成多類別分類（0–9）訓練與測試迴圈，計算整體 accuracy。
- 顯示數張測試圖片的真實標籤與預測結果，說明各 gate 在記憶與遺忘中的角色，並與簡單 RNN 的差異做概念性比較。
- **功能重點：** LSTM 結構與 gate 機制、序列建模、PyTorch 從零實作 recurrent cell、RNN vs LSTM 特性比較。

---

Lab 8｜Vision Transformer (ViT) 工業缺陷分類

- 在工業缺陷影像資料集上，以 PyTorch 從零實作 Vision Transformer：包含 patch embedding、位置編碼、多頭自注意力、Transformer encoder 與最終分類 MLP。
- 完成資料切分（train/test）、Resize / Gray / Normalize 等前處理後訓練 ViT，透過調整 dim、depth、heads、mlp_dim、dropout 等超參數提升測試準確率。
- 實作 evaluation loop 計算 test loss / accuracy，並畫出混淆矩陣與每類別多張測試影像的預測結果。
- **功能重點：** ViT 架構拆解與實作、自注意力機制、工業影像分類、混淆矩陣與視覺化分析。

---

Lab 9｜GAN 與 CycleGAN 影像生成與風格轉換

- 在 FashionMNIST 與 CIFAR-10 上，使用課堂 GAN 架構，針對每個資料集至少三個類別訓練標準 GAN，生成 fake 與 mimic 影像，觀察模式崩潰與多樣性。
- 實作 CycleGAN（含 cycle-consistency loss），在選定類別之間進行影像對映與風格轉換，輸出 real / fake / mimic 影像比較效果。
- 透過視覺品質、多樣性、mimic 正確性等角度比較 GAN 與 CycleGAN 的生成能力與穩定度，並討論可能的改進方向。
- **功能重點：** GAN / CycleGAN 架構與訓練、影像生成與風格轉換、生成品質與模式崩潰分析、視覺化比較不同生成模型。
