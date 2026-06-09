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
    df = joblib.load('boardgame_database.joblib')
    matrix = joblib.load('boardgame_vectors.joblib')
    return df, matrix

try:
    df, Final_Matrix = load_ai_brain()
    st.success("✅ Đã kết nối thành công với Bộ Não AI!")
except:
    st.error("❌ Không tìm thấy file '.joblib'. Vui lòng để chung thư mục với file code này.")
    st.stop()

# =====================================================================
# THIẾT KẾ BỘ LỌC GIAO DIỆN VÀ NHẬP INPUT THÔNG MINH
# =====================================================================
st.sidebar.header("⚙️ Cấu Hình Bộ Lọc AI")

# Cho phép kéo thanh slider lên tối đa 100 game
top_n = st.sidebar.slider("Số lượng game muốn gợi ý:", min_value=3, max_value=100, value=10, step=1)

# Các tham số thuật toán ẩn tinh chỉnh độ nhạy
alpha = st.sidebar.slider("Đòn bẩy độ nổi tiếng (Alpha):", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
gamma = st.sidebar.slider("Độ gắt phạt độ khó (Gamma):", min_value=0.0, max_value=0.5, value=0.15, step=0.05)

# TÍNH NĂNG ĐÓNG/MỞ CỘT CHO GAME OUTPUT (BẢNG KẾT QUẢ)
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Tùy Chọn Đóng/Mở Cột Kết Quả")
extra_cols = st.sidebar.multiselect(
    "Tích chọn để MỞ RỘNG (hiển thị) thêm các cột dữ liệu nâng cao:",
    options=["Số Người Chơi", "Thời Gian Chơi", "Mechanic", "Domain", "Category", "Designer", "Artist"],
    default=["Số Người Chơi", "Thời Gian Chơi"], # Mặc định mở sẵn 2 cột cơ bản cho đẹp giao diện
    help="Bỏ tích chọn để THU GỌN (ẩn) cột tương ứng trong bảng kết quả phía dưới!"
)

# Thanh nhập dữ liệu tự động nhắc bài thông minh
all_game_names = sorted(df['Name'].tolist())
selected_game = st.selectbox(
    "👉 Nhập hoặc chọn tên Boardgame bạn thích tại đây (Hệ thống tự động sửa sai chính tả):",
    options=[""] + all_game_names,
    index=0,
    help="Hãy gõ những chữ cái đầu, hệ thống sẽ lọc ra tên game chính xác nhất!"
)

# Hàm bổ trợ giúp trích xuất an toàn dữ liệu từ dataframe đề phòng lệch phông chữ/viết hoa viết thường
def get_safe_val(dataframe, index, col_candidate, default_val="N/A"):
    for col in dataframe.columns:
        if col.strip().lower() == col_candidate.strip().lower():
            val = dataframe.loc[index, col]
            return val if pd.notna(val) else default_val
    return default_val

# =====================================================================
# ĐỘNG CƠ XỬ LÝ TOÁN HỌC & XUẤT KẾT QUẢ RA GIAO DIỆN
# =====================================================================
if selected_game != "":
    idx = df[df['Name'] == selected_game].index[0]
    game_cluster = df.loc[idx, 'cluster_id']
    target_complexity = df.loc[idx, 'Complexity Average']
    
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
        
        # Cơ chế phạt độ khó chuẩn
        delta_complexity = abs(target_complexity - candidate_complexity)
        if delta_complexity <= 1.0:
            penalty_factor = 1.0
        else:
            penalty_factor = 1.0 - gamma * (delta_complexity - 1.0)
            penalty_factor = max(0.1, penalty_factor)
            
        final_score = sim * (1 + alpha * bonus) * penalty_factor
        
        # Đã đồng bộ chính xác tên các cột văn bản gốc từ mô hình của bạn
        results.append({
            'Tên Game': df.loc[cluster_idx, 'Name'],
            'Năm': df.loc[cluster_idx, 'Year Published'],
            'Độ Khó': round(candidate_complexity, 2),
            'Điểm Khớp (Match)': round(final_score, 4),
            'Hạng BGG': int(df.loc[cluster_idx, 'BGG Rank']) if df.loc[cluster_idx, 'BGG Rank'] > 0 else "N/A",
            'Min Players': get_safe_val(df, cluster_idx, 'Min Players'),
            'Max Players': get_safe_val(df, cluster_idx, 'Max Players'),
            'Play Time': get_safe_val(df, cluster_idx, 'Play Time'),
            'Mechanic': get_safe_val(df, cluster_idx, 'Mechanics'), 
            'Domain': get_safe_val(df, cluster_idx, 'Domains'),
            'Category': get_safe_val(df, cluster_idx, 'boardgamecategory'),
            'Designer': get_safe_val(df, cluster_idx, 'boardgamedesigner'),
            'Artist': get_safe_val(df, cluster_idx, 'boardgameartist')
        })
        
    results_df = pd.DataFrame(results).sort_values(by='Điểm Khớp (Match)', ascending=False).head(top_n)
    
    # =====================================================================
    # HIỂN THỊ THÔNG TIN TỔNG QUAN CỦA GAME INPUT
    # =====================================================================
    st.write("---")
    st.subheader(f"🎮 Bạn đã chọn: {selected_game}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.info(f"**Năm Phát Hành:**\n\n Năm {target_year}")
    with col2:
        st.success(f"**Xếp Hạng BGG Rank:**\n\n Hạng {target_rank_display}")
    with col3:
        st.warning(f"**Độ Khó Gốc:**\n\n {round(target_complexity, 2)} / 5")
    with col4:
        st.metric(label="Bang Phái Số", value=f"Nhóm {game_cluster}")
    with col5:
        st.metric(label="Ứng Viên Cùng Cụm", value=f"{len(cluster_indices)} games")
        
    # THIẾT KẾ ĐÓNG/MỞ THÔNG TIN CHI TIẾT CHO GAME INPUT QUA CÁC EXPANDER
    st.markdown("**🔍 Xem thêm thông tin thuộc tính mở rộng của game đầu vào:**")
    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    
    with exp_col1:
        with st.expander("👥 Số Người & Thời Gian"):
            min_p = get_safe_val(df, idx, 'Min Players')
            max_p = get_safe_val(df, idx, 'Max Players')
            p_time = get_safe_val(df, idx, 'Play Time')
            st.write(f"- **Min player:** {min_p} người")
            st.write(f"- **Max player:** {max_p} người")
            st.write(f"- **Time play:** {p_time} phút")
            
    with exp_col2:
        with st.expander("⚙️ Cơ Chế & Phân Loại"):
            mech = get_safe_val(df, idx, 'Mechanics')
            cat = get_safe_val(df, idx, 'boardgamecategory')
            st.write(f"**Mechanic:**\n{mech}")
            st.write(f"**Category:**\n{cat}")
            
    with exp_col3:
        with st.expander("🌐 Hệ Sinh Thế (Domain)"):
            dom = get_safe_val(df, idx, 'Domains')
            st.write(f"**Domain:**\n{dom}")
            
    with exp_col4:
        with st.expander("🎨 Tác Giả & Họa Sĩ"):
            des = get_safe_val(df, idx, 'boardgamedesigner')
            art = get_safe_val(df, idx, 'boardgameartist')
            st.write(f"**Designer:**\n{des}")
            st.write(f"**Artist:**\n{art}")

    # =====================================================================
    # XỬ LÝ ĐÓNG/MỞ CỘT ĐỘNG CHO BẢNG GAME OUTPUT THEO THỜI GIAN THỰC
    # =====================================================================
    st.write("---")
    st.subheader(f"🎯 TOP {top_n} BOARDGAME SIÊU HỢP GU ĐƯỢC AI ĐỀ XUẤT:")
    
    # Định hình các cột bắt buộc hiển thị mặc định
    final_visible_columns = ['Tên Game', 'Năm', 'Độ Khó', 'Điểm Khớp (Match)', 'Hạng BGG']
    
    # Ma trận ánh xạ từ lựa chọn Sidebar sang các cột dữ liệu ẩn trong DataFrame kết quả
    mapping_dictionary = {
        "Số Người Chơi": ['Min Players', 'Max Players'],
        "Thời Gian Chơi": ['Play Time'],
        "Mechanic": ['Mechanic'],
        "Domain": ['Domain'],
        "Category": ['Category'],
        "Designer": ['Designer'],
        "Artist": ['Artist']
    }
    
    # Nếu người dùng chọn thêm cột nào, ta append cột đó vào danh sách hiển thị
    for chosen_option in extra_cols:
        if chosen_option in mapping_dictionary:
            final_visible_columns.extend(mapping_dictionary[chosen_option])
            
    # Tiến hành lọc các cột dữ liệu theo cấu hình Đóng/Mở động
    filtered_results_df = results_df[final_visible_columns]
    
    # Đánh số thứ tự từ 1 đến N cho bảng kết quả
    filtered_results_df.index = np.arange(1, len(filtered_results_df) + 1)
    
    # Xuất bảng danh sách kết quả trực quan ra UI Web
    st.dataframe(filtered_results_df, use_container_width=True, height=min(40 * top_n + 100, 600))
    
    st.balloons() # Hiệu ứng bắn bóng bay ăn mừng
