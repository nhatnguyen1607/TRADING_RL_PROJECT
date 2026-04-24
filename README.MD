# 📈 Ứng dụng Deep Reinforcement Learning trong Giao dịch Tài chính

Dự án này là một hệ thống nghiên cứu (Research-grade) ứng dụng các thuật toán **Deep Reinforcement Learning (DRL)** để tối ưu hóa chiến lược giao dịch chứng khoán. Dự án so sánh hiệu suất giữa **DQN (Discrete)** và **Actor-Critic (Continuous)**, đồng thời tích hợp các phương pháp State-of-the-Art (SOTA) lấy cảm hứng từ hệ sinh thái **FinRL**.

## 🌟 Điểm nhấn Học thuật (SOTA Features)
Vượt qua các triển khai RL cơ bản, dự án này áp dụng các cải tiến nghiên cứu:
1. **Macro Risk Indicator (VIX):** Tích hợp Chỉ số Biến động CBOE (VIX) vào Không gian Trạng thái (State Space) giúp Agent nhận biết và phòng thủ trong các chu kỳ "thị trường gấu" (Market Crash).
2. **Risk-Adjusted Reward (Sharpe-like Reward):** Thay vì chỉ tối ưu hóa PnL thuần túy, hàm Reward được thiết kế lại: `Reward = Lợi nhuận - (Hệ số * Phương sai)`. Điều này ép Agent học cách giao dịch mang lại lợi nhuận ổn định thay vì lướt sóng rủi ro cao.
3. **Mạng LSTM xử lý Time-Series:** Không sử dụng MLP tiêu chuẩn, hệ thống thiết kế mạng Long Short-Term Memory (LSTM) để nắm bắt chuỗi thời gian của OHLCV và các chỉ báo kỹ thuật.
4. **Ensemble Validation Strategy:** Không chỉ đánh giá trên tập Test, hệ thống sử dụng tập Validation để chọn ra Agent tốt nhất (DQN hoặc Actor-Critic) dựa trên Sharpe Ratio cao nhất trước khi giao dịch thực tế.

## 🏗️ Cấu trúc hệ thống
```text
TRADING_RL_PROJECT/
├── data/               # Dữ liệu tự động sinh qua API Yahoo Finance
├── envs/
│   └── trading_env.py  # Môi trường Gymnasium (Risk-Penalty Reward)
├── models/
│   ├── dqn_net.py      # Mạng Deep Q-Network với LSTM
│   └── ac_net.py       # Mạng Actor-Critic với LSTM
├── agents/
│   ├── dqn_agent.py    # Logic Experience Replay & Target Network
│   └── ac_agent.py     # Logic Policy Gradient & TD Error
├── utils/
│   └── data_loader.py  # Preprocessing (SMA, MACD, RSI, VIX)
├── main.py             # Pipeline Training, Validation & Backtesting
├── requirements.txt    # Quản lý dependency (Python 3.10)
└── .gitignore          
🧠 Formulation MDP (Markov Decision Process)State ($\mathcal{S}$): Ma trận 2D (Window_Size, Features). Features bao gồm OHLCV chuẩn hóa, VIX, RSI, MACD, SMA 20/50, tỷ trọng vốn và số dư hiện tại.Action ($\mathcal{A}$): - DQN: Không gian rời rạc $\{Hold, Buy\_Max, Sell\_All\}$.Actor-Critic: Không gian liên tục $[-1, 1]$ tương ứng với tỷ lệ phân bổ vốn.Reward ($\mathcal{R}$): Lợi nhuận phần trăm trừ đi phương sai (Risk-Adjusted).🚀 Hướng dẫn Cài đặt & Khởi chạy1. Thiết lập môi trường (Bắt buộc dùng Python 3.10)Bash# Tạo và kích hoạt virtual environment
py -3.10 -m venv venv
venv\Scripts\activate  # (Trên Windows)

# Cài đặt thư viện không chứa xung đột
pip install -r requirements.txt
2. Thực thi PipelineHệ thống được gói gọn toàn bộ luồng (Download Data -> Train DQN -> Train AC -> Validate -> Backtest) trong một lệnh duy nhất:

python main.py

📊 Phương pháp Đánh giáWalk-forward Validation: Dữ liệu được chia thành Train (70%), Validation (15%), Test (15%).Metrics: Hiệu suất của Agent được so sánh trực tiếp với chiến lược Buy & Hold trên tập Test thông qua biểu đồ Equity Curve (Đường cong Vốn) và Annualized Sharpe Ratio (Tỷ lệ rủi ro/lợi nhuận).