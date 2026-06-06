import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cấu hình trang ứng dụng giao diện rộng, phù hợp hiển thị danh sách dài 100 game
st.set_page_config(page_title="Boardgame AI Recommender Pro", page_icon="🎮", layout="wide")

st.title("🔮 Hệ Thống Gợi Ý Boardgame Thông Minh - Bản Mở Rộng 100 Game")
st.write("Tìm kiếm những tựa game đỉnh cao phù hợp nhất với gu của bạn dựa trên 8 Bát Đại Bang Phái.")

# =====================================================================
# TẢI BỘ NÃO AI (CHỈ MẤT 0.5 GIÂY)
# =====================================================================
@st.cache_resource
def load_ai_brain():
    # Đã sửa thành định dạng nén siêu sâu .joblib
    df = joblib.load('boardgame_database.joblib')
    matrix = joblib.load('boardgame_vectors.joblib')
    return df, matrix

try:
    df, Final_Matrix = load_ai_brain()
    st.success("✅ Đã kết nối thành công với Bộ Não AI!")
except:
    # Cập nhật lại thông báo lỗi cho khớp định dạng mới
    st.error("❌ Không tìm thấy file '.joblib'. Vui lòng để chung thư mục với file code này.")
    st.stop()

# =====================================================================
# THIẾT KẾ BỘ LỌC GIAO DIỆN VÀ NHẬP INPUT THÔNG MINH
# =====================================================================
st.sidebar.header("⚙️ Cấu Hình Bộ Lọc AI")

# 🔥 ĐÃ NÂNG CẤP: Cho phép kéo thanh slider lên tối đa 100 game
top_n = st.sidebar.slider("Số lượng game muốn gợi ý:", min_value=3, max_value=100, value=10, step=1)

# Các tham số thuật toán ẩn tinh chỉnh độ nhạy
alpha = st.sidebar.slider("Đòn bẩy độ nổi tiếng (Alpha):", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
gamma = st.sidebar.slider("Độ gắt phạt độ khó (Gamma):", min_value=0.0, max_value=0.5, value=0.15, step=0.05)

# Thanh nhập dữ liệu tự động nhắc bài thông minh
all_game_names = sorted(df['Name'].tolist())
selected_game = st.selectbox(
    "👉 Nhập hoặc chọn tên Boardgame bạn thích tại đây (Hệ thống tự động sửa sai chính tả):",
    options=[""] + all_game_names,
    index=0,
    help="Hãy gõ những chữ cái đầu, hệ thống sẽ lọc ra tên game chính xác nhất!"
)

# =====================================================================
# ĐỘNG CƠ XỬ LÝ TOÁN HỌC & XUẤT KẾT QUẢ RA GIAO DIỆN
# =====================================================================
if selected_game != "":
    idx = df[df['Name'] == selected_game].index[0]
    game_cluster = df.loc[idx, 'cluster_id']
    target_complexity = df.loc[idx, 'Complexity Average']
    
    # 🔥 LẤY THÔNG TIN CỦA GAME INPUT ĐỂ HIỂN THỊ THEO YÊU CẦU
    target_year = int(df.loc[idx, 'Year Published'])
    target_rank = df.loc[idx, 'BGG Rank']
    target_rank_display = int(target_rank) if target_rank > 0 else "N/A"
    
    # Khoanh vùng không gian cụm
    cluster_indices = df[df['cluster_id'] == game_cluster].index.tolist()
    
    target_vector = Final_Matrix[idx].reshape(1, -1)
    cluster_vectors = Final_Matrix[cluster_indices]
    cos_sim = cosine_similarity(target_vector, cluster_vectors)[0]
    
    results = []
    for i, cluster_idx in enumerate(cluster_indices):
        if cluster_idx == idx: 
            continue
            
        sim = cos_sim[i]
        bonus = df.loc[cluster_idx, 'Rank_Bonus']
        candidate_complexity = df.loc[cluster_idx, 'Complexity Average']
        
        # Cơ chế phạt độ khó chuẩn của bạn
        delta_complexity = abs(target_complexity - candidate_complexity)
        if delta_complexity <= 1.0:
            penalty_factor = 1.0
        else:
            penalty_factor = 1.0 - gamma * (delta_complexity - 1.0)
            penalty_factor = max(0.1, penalty_factor)
            
        final_score = sim * (1 + alpha * bonus) * penalty_factor
        
        results.append({
            'Tên Game': df.loc[cluster_idx, 'Name'],
            'Năm': df.loc[cluster_idx, 'Year Published'],
            'Độ Khó': round(candidate_complexity, 2),
            'Điểm Khớp (Match)': round(final_score, 4),
            'Hạng BGG': int(df.loc[cluster_idx, 'BGG Rank']) if df.loc[cluster_idx, 'BGG Rank'] > 0 else "N/A"
        })
        
    results_df = pd.DataFrame(results).sort_values(by='Điểm Khớp (Match)', ascending=False).head(top_n)
    
    # =====================================================================
    # HIỂN THỊ THÔNG TIN TỔNG QUAN CỦA GAME INPUT (ĐÃ ĐƯỢC CẢI TIẾN)
    # =====================================================================
    st.write("---")
    st.subheader(f"🎮 Bạn đã chọn: {selected_game}")
    
    # Tăng cấu trúc từ 3 cột lên thành 5 cột để bổ sung thông tin Năm và Hạng BGG của game đầu vào
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.info(f"**Năm Phát Hành:**\n\n Năm {target_year}")
    with col2:
        st.success(f"**Xếp Hạng BGG Rank:**\n\n Hạng {target_rank_display}")
    with col3:
        st.warning(f"**Độ Khó Gốc:**\n\n {round(target_complexity, 2)} / 5")
    with col4:
        st.dark_note = st.metric(label="Bang Phái Số", value=f"Nhóm {game_cluster}")
    with col5:
        st.metric(label="Ứng Viên Cùng Cụm", value=f"{len(cluster_indices)} games")
        
    # Xuất bảng danh sách kết quả trực quan trên Web UI
    st.write("---")
    st.subheader(f"🎯 TOP {top_n} BOARDGAME SIÊU HỢP GU ĐƯỢC AI ĐỀ XUẤT:")
    
    # Đánh số thứ tự từ 1 đến N
    results_df.index = np.arange(1, len(results_df) + 1)
    
    # Sử dụng st.dataframe với chiều cao động để người dùng Windows dễ cuộn chuột khi chọn hiển thị 100 game
    st.dataframe(results_df, use_container_width=True, height=min(40 * top_n + 100, 600))
    
    st.balloons() # Hiệu ứng bắn bóng bay ăn mừng
