import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="FIFA 19 Player Analysis & Prediction",
    page_icon="⚽",
    layout="wide"
)

# Tiêu đề chính
st.title("⚽ FIFA 19 - Phân tích & Dự đoán Chỉ số Cầu thủ 🎯")
st.markdown("---")

# Sidebar
st.sidebar.header("🎮 Điều hướng")
page = st.sidebar.selectbox(
    "Chọn trang:",
    ["🏠 Trang chủ", "🔮 Dự đoán Overall", "📊 So sánh Cầu thủ", "📈 Phân tích Dữ liệu"]
)

# Hàm đọc và xử lý dữ liệu
@st.cache_data
def load_data():
    try:
        # Thử đọc file đã xử lý
        df = pd.read_csv('processed_data.csv')
        return df
    except FileNotFoundError:
        st.error("Không tìm thấy file 'processed_data.csv'. Vui lòng chạy notebook để tạo file này trước.")
        return None

# Hàm hồi quy tuyến tính thủ công
def manual_linear_regression(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X))
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

def manual_predict(X, w):
    X = np.column_stack((np.ones(X.shape[0]), X))
    return X.dot(w)

# Trang chủ
if page == "🏠 Trang chủ":
    st.header("🏠 Chào mừng đến với FIFA 19 Player Analysis!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Tính năng chính:
        - **🔮 Dự đoán Overall**: Dự đoán chỉ số tổng quát của cầu thủ
        - **📊 So sánh Cầu thủ**: So sánh các thuộc tính giữa các cầu thủ
        - **📈 Phân tích Dữ liệu**: Khám phá và phân tích dữ liệu FIFA 19
        
        ### 🎯 Mô hình sử dụng:
        - **Linear Regression**: Hồi quy tuyến tính để dự đoán Overall
        - **Feature Importance**: Phân tích tầm quan trọng của các thuộc tính
        """)
    
    with col2:
        st.markdown("""
        ### ⚽ Dữ liệu FIFA 19:
        - **Số lượng cầu thủ**: 18,129
        - **Số thuộc tính**: 55
        - **Độ chính xác mô hình**: ~91%
        
        ### 🔧 Thuộc tính chính:
        - Age, Potential, International Reputation
        - Skill Moves, Reactions
        - Các chỉ số kỹ thuật (Finishing, Dribbling, etc.)
        """)
    
    # Hiển thị thống kê cơ bản
    df = load_data()
    if df is not None:
        st.markdown("### 📊 Thống kê cơ bản")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tổng số cầu thủ", f"{len(df):,}")
        with col2:
            st.metric("Độ tuổi trung bình", f"{df['Age'].mean():.1f}")
        with col3:
            st.metric("Overall trung bình", f"{df['Overall'].mean():.1f}")
        with col4:
            st.metric("Potential trung bình", f"{df['Potential'].mean():.1f}")

# Trang dự đoán
elif page == "🔮 Dự đoán Overall":
    st.header("🔮 Dự đoán Chỉ số Overall của Cầu thủ")
    st.markdown("Nhập các thông số dưới đây để dự đoán chỉ số Overall của cầu thủ.")
    
    df = load_data()
    if df is not None:
        # Chọn các đặc trưng quan trọng nhất
        important_features = ['Reactions', 'Potential', 'Age', 'International Reputation', 'Skill Moves']
        X = df[important_features]
        y = df['Overall']
        
        # Huấn luyện mô hình
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Mô hình thủ công
        w_manual = manual_linear_regression(X_train.values, y_train.values)
        
        # Mô hình sklearn
        model_sklearn = LinearRegression()
        model_sklearn.fit(X_train, y_train)
        
        # Giao diện nhập liệu
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Nhập thông số cầu thủ")
            reactions = st.slider("Reactions", 0, 100, 70, help="Khả năng phản ứng của cầu thủ")
            potential = st.slider("Potential", 0, 100, 80, help="Tiềm năng phát triển")
            age = st.slider("Age", 15, 50, 25, help="Tuổi của cầu thủ")
            
        with col2:
            international_reputation = st.slider("International Reputation", 1, 5, 3, help="Danh tiếng quốc tế")
            skill_moves = st.slider("Skill Moves", 1, 5, 3, help="Kỹ năng rê bóng")
            
            st.markdown("---")
            if st.button("🔮 Dự đoán", type="primary", use_container_width=True):
                # Dự đoán
                input_data = np.array([[reactions, potential, age, international_reputation, skill_moves]])
                
                # Dự đoán từ mô hình thủ công
                predicted_manual = manual_predict(input_data, w_manual)[0]
                
                # Dự đoán từ mô hình sklearn
                predicted_sklearn = model_sklearn.predict(input_data)[0]
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Mô hình thủ công:** {predicted_manual:.2f}")
                with col2:
                    st.success(f"**Mô hình sklearn:** {predicted_sklearn:.2f}")
                
                # Đánh giá độ chính xác
                y_pred_manual = manual_predict(X_test.values, w_manual)
                y_pred_sklearn = model_sklearn.predict(X_test)
                
                r2_manual = r2_score(y_test, y_pred_manual)
                r2_sklearn = r2_score(y_test, y_pred_sklearn)
                
                rmse_manual = np.sqrt(mean_squared_error(y_test, y_pred_manual))
                rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
                
                st.markdown("### 📊 Độ chính xác mô hình")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score (Manual)", f"{r2_manual:.4f}")
                    st.metric("RMSE (Manual)", f"{rmse_manual:.4f}")
                with col2:
                    st.metric("R² Score (Sklearn)", f"{r2_sklearn:.4f}")
                    st.metric("RMSE (Sklearn)", f"{rmse_sklearn:.4f}")

# Trang so sánh cầu thủ
elif page == "📊 So sánh Cầu thủ":
    st.header("📊 So sánh Cầu thủ")
    st.markdown("Chọn hai hoặc nhiều cầu thủ để so sánh các thuộc tính của họ.")
    
    df = load_data()
    if df is not None:
        # Chọn cầu thủ
        player_names = st.multiselect(
            "Chọn cầu thủ để so sánh:",
            options=df['Name'].unique(),
            default=df['Name'].head(3).tolist()
        )
        
        if len(player_names) > 1:
            # Lọc dữ liệu cho các cầu thủ được chọn
            selected_players = df[df['Name'].isin(player_names)]
            
            # Hiển thị bảng so sánh
            st.subheader("📋 Bảng so sánh")
            comparison_cols = ['Name', 'Overall', 'Potential', 'Age', 'International Reputation', 
                             'Skill Moves', 'Reactions', 'Value', 'Wage']
            st.dataframe(selected_players[comparison_cols].set_index('Name'), use_container_width=True)
            
            # Biểu đồ so sánh
            st.subheader("📈 Biểu đồ so sánh thuộc tính")
            
            # Chọn thuộc tính để so sánh
            attribute_options = ['Overall', 'Potential', 'Age', 'International Reputation', 
                               'Skill Moves', 'Reactions', 'Value', 'Wage']
            selected_attributes = st.multiselect(
                "Chọn thuộc tính để so sánh:",
                options=attribute_options,
                default=['Overall', 'Potential', 'Age']
            )
            
            if selected_attributes:
                fig, ax = plt.subplots(figsize=(12, 6))
                selected_players.set_index('Name')[selected_attributes].plot(kind='bar', ax=ax)
                plt.title("So sánh thuộc tính giữa các cầu thủ")
                plt.ylabel("Giá trị thuộc tính")
                plt.xticks(rotation=45)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Biểu đồ radar (nếu có đủ thuộc tính)
                if len(selected_attributes) >= 3:
                    st.subheader("🎯 Biểu đồ radar")
                    
                    # Chuẩn hóa dữ liệu cho biểu đồ radar
                    radar_data = selected_players[selected_attributes].copy()
                    for col in selected_attributes:
                        if col in ['Value', 'Wage']:
                            radar_data[col] = radar_data[col] / radar_data[col].max() * 100
                    
                    # Tạo biểu đồ radar
                    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                    
                    angles = np.linspace(0, 2 * np.pi, len(selected_attributes), endpoint=False).tolist()
                    angles += angles[:1]  # Đóng vòng tròn
                    
                    for idx, player in radar_data.iterrows():
                        values = player[selected_attributes].values.tolist()
                        values += values[:1]  # Đóng vòng tròn
                        ax.plot(angles, values, 'o-', linewidth=2, label=idx)
                        ax.fill(angles, values, alpha=0.25)
                    
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(selected_attributes)
                    ax.set_ylim(0, 100)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    plt.title("Biểu đồ radar so sánh thuộc tính")
                    st.pyplot(fig)

# Trang phân tích dữ liệu
elif page == "📈 Phân tích Dữ liệu":
    st.header("📈 Phân tích Dữ liệu FIFA 19")
    
    df = load_data()
    if df is not None:
        # Tổng quan dữ liệu
        st.subheader("📋 Tổng quan dữ liệu")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Thông tin dữ liệu:**")
            st.write(f"Kích thước: {df.shape[0]} hàng × {df.shape[1]} cột")
            st.write(f"Bộ nhớ: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        with col2:
            st.write("**Thống kê cơ bản:**")
            st.write(f"Overall trung bình: {df['Overall'].mean():.2f}")
            st.write(f"Overall cao nhất: {df['Overall'].max()}")
            st.write(f"Overall thấp nhất: {df['Overall'].min()}")
        
        # Phân phối Overall
        st.subheader("📊 Phân phối chỉ số Overall")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Overall'], bins=30, kde=True, color='skyblue', edgecolor='black')
        plt.title('Phân phối điểm số tổng quát của cầu thủ')
        plt.xlabel('Overall Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Phân phối tuổi
        st.subheader("📊 Phân phối tuổi")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='Age', ax=ax)
            plt.title('Phân phối tuổi cầu thủ')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, y='Age', ax=ax)
            plt.title('Box plot tuổi cầu thủ')
            plt.ylabel('Age')
            st.pyplot(fig)
        
        # Tương quan giữa các thuộc tính
        st.subheader("🔗 Ma trận tương quan")
        
        # Chọn các thuộc tính số để phân tích tương quan
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_cols = st.multiselect(
            "Chọn thuộc tính để phân tích tương quan:",
            options=numeric_cols.tolist(),
            default=['Overall', 'Potential', 'Age', 'Value', 'Wage', 'International Reputation']
        )
        
        if len(correlation_cols) > 1:
            corr_matrix = df[correlation_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, ax=ax)
            plt.title('Ma trận tương quan giữa các thuộc tính')
            st.pyplot(fig)
            
            # Hiển thị tương quan với Overall
            st.subheader("📊 Tương quan với Overall")
            overall_corr = corr_matrix['Overall'].sort_values(ascending=False)
            st.write(overall_corr)
            
            # Biểu đồ tương quan
            fig, ax = plt.subplots(figsize=(10, 6))
            overall_corr.plot(kind='bar', ax=ax, color='steelblue')
            plt.title('Tương quan của các thuộc tính với Overall')
            plt.xlabel('Thuộc tính')
            plt.ylabel('Hệ số tương quan')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚽ FIFA 19 Player Analysis & Prediction App | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# python run_app.py