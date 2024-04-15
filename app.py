import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Graduation Assistant",
                   layout="wide",
                   page_icon="🎓")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Graduation Prediction System',
                           ['Analytical Intent', 'Visualization', 'Graduation Prediction'],
                        #    menu_icon='university',
                        #    icons=['bar-chart', 'mortar-board'],
                           default_index=0)

# Visualization methods

# IPS
def plot_ips_variance(df):
    st.subheader("Variabilitas Nilai IPS")
    ips_columns = [col for col in df.columns if col.startswith('IPS')]
    ips_df = df[ips_columns]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ips_df)
    # plt.title('Variability Analysis of IPS Scores')
    plt.xlabel('Semester')
    plt.ylabel('IPS Score')
    st.pyplot(plt.gcf())
    
    st.write("**Interpretasi:**")
    st.write("Boxplot diatas menunjukkan distribusi kinerja akademik mahasiswa dari semester ke semester, dengan memperlihatkan kuartil, nilai ekstrem, serta kecenderungan nilai tengah. Garis di dalam kotak mewakili median, sedangkan kotak menunjukkan kuartil pertama (25%) dan ketiga (75%).'Whiskers' di kedua ujung kotak menunjukkan kisaran data, dengan titik-titik di luar 'whiskers' sebagai outliers potensial.")
    
    st.write("**Insight:**")
    st.write("Distribusi kinerja akademik mahasiswa dari semester ke semester cenderung seragam, dengan sebagian besar nilai IPS berada di sekitar median, ditunjukkan oleh lebar kotak yang konsisten. Variasi yang terlihat dalam panjang 'whiskers' antara semester-semester menunjukkan bahwa beberapa semester memiliki kisaran nilai yang lebih luas daripada yang lain, mungkin mengindikasikan tingkat variasi kinerja yang berbeda.")

    st.write("**Actionable Insight:**")
    st.write("""
                - Mengidentifikasi semester-semester yang memiliki variasi kinerja yang lebih besar dan melakukan analisis lebih lanjut untuk memahami faktor-faktor penyebabnya.
                - Mengembangkan strategi untuk memberikan dukungan tambahan kepada mahasiswa selama semester-semester yang memiliki variasi kinerja yang lebih tinggi.
                - Melakukan pemantauan terus-menerus terhadap outlier potensial dan memberikan perhatian khusus pada mahasiswa yang mungkin memerlukan bantuan tambahan untuk meningkatkan kinerja akademik mereka.
                """)

# IPK
def plot_gpa_distribution(df):
    st.subheader("Distribusi IPK")
    fig = px.histogram(df, x='IPK ', nbins=20)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("**Interpretasi:**")
    st.write("Grafik diatas menunjukkan distribusi IPK mahasiswa. Berdasarkan grafik tersebut dapat dilihat bahwa distribusi nilai IPK mahasiswa paling banyak berada dalam rentang 3 - 3.09.")
    
    st.write("**Insight:**")
    st.write("Distribusi IPK mahasiswa yang paling banyak berada dalam rentang 3 - 3.09 menunjukkan bahwa sebagian besar mahasiswa mencapai tingkat kesuksesan akademik yang relatif tinggi.")
    
    st.write("**Actionable Insight:**")
    st.write("Peningkatan upaya dukungan akademik dan strategi pembelajaran tambahan dapat diarahkan kepada mahasiswa di luar rentang IPK 3 - 3.19 untuk membantu mereka mencapai potensi akademik maksimal mereka.")

def plot_gpa_by_status(df):
    st.subheader("Korelasi Antara Status Mahasiswa & Ketepatan Kelulusan")
    crosstab = pd.crosstab(df['STATUS MAHASISWA'].map({0: 'Bekerja', 1: 'Mahasiswa'}), df['STATUS KELULUSAN'].map({1: 'Tepat', 0: 'Terlambat'}))
    # crosstab = pd.crosstab(df['STATUS MAHASISWA'], df['STATUS KELULUSAN'])
    styled_table = crosstab.style.background_gradient(cmap='coolwarm').set_table_attributes("style='display:inline'")
    st.write(styled_table)
    
    st.write("**Interpretasi:**")
    st.write("Tabel di atas menunjukkan bahwa mahasiswa yang bekerja cenderung memiliki jumlah lulus terlambat yang signifikan dibandingkan dengan mahasiswa yang tidak bekerja.")
    
    st.write("**Insight:**")
    st.write("Faktor seperti bekerja dapat berpengaruh pada akurasi kelulusan mahasiswa. Mahasiswa yang bekerja mungkin mengalami tantangan dalam mengelola waktu antara pekerjaan dan studi.")

    st.write("**Actionable Insight:**")
    st.write("Menyediakan dukungan tambahan dan fleksibilitas bagi mahasiswa yang bekerja untuk meningkatkan kesempatan mereka dalam menyelesaikan studi tepat waktu.")
    

def plot_marriage_status(df):
    st.subheader("Korelasi Antara Status Menikah & Ketepatan Kelulusan")
    crosstab = pd.crosstab(df['STATUS NIKAH'].map({0: 'Belum Menikah', 1: 'Menikah'}), df['STATUS KELULUSAN'].map({1: 'Tepat', 0: 'Terlambat'}))
    # crosstab = pd.crosstab(df['STATUS NIKAH'], df['STATUS KELULUSAN'])
    styled_table = crosstab.style.background_gradient(cmap='coolwarm', axis=1).set_table_attributes("style='margin-left: auto; margin-right: auto;'")
    st.write(styled_table)
    
    st.write("**Interpretasi:**")
    st.write("Tabel di atas menunjukkan jumlah mahasiswa yang lulus tepat waktu (TEPAT) dan terlambat (TERLAMBAT) berdasarkan status pernikahan mereka. Dapat diamati bahwa mayoritas mahasiswa yang belum menikah lulus tepat waktu, sementara mayoritas mahasiswa yang menikah cenderung lulus terlambat.")
    
    st.write("**Insight:**")
    st.write("Status pernikahan dapat berpengaruh pada akurasi kelulusan mahasiswa. Mahasiswa yang menikah mungkin memiliki tanggung jawab tambahan yang dapat mempengaruhi fokus dan ketersediaan waktu mereka untuk belajar, yang dapat menyebabkan peningkatan jumlah lulus terlambat.")

    st.write("**Actionable Insight:**")
    st.write("Menyediakan dukungan tambahan dan sumber daya kepada mahasiswa yang menikah untuk membantu mereka mengelola tanggung jawab tambahan dan meningkatkan kesempatan mereka dalam menyelesaikan studi tepat waktu.")


def plot_ipk_correlation(df):
    df_visual = df.copy()  # Salin dataframe untuk visualisasi
    df_visual['STATUS KELULUSAN'] = df_visual['STATUS KELULUSAN'].replace({0: 'Terlambat', 1: 'Tepat'})

    # Bar chart untuk membandingkan rata-rata IPK antara status kelulusan
    st.subheader("Rata-rata IPK berdasarkan Status Kelulusan")
    avg_ipk = df_visual.groupby('STATUS KELULUSAN')['IPK '].mean().reset_index()  # Menggunakan df_visual
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=avg_ipk, x='STATUS KELULUSAN', y='IPK ')
    plt.xlabel('Status Kelulusan')
    plt.ylabel('Rata-rata IPK')
    
    # Menambahkan label nilai pada setiap bar dengan jarak antara angka dan batang
    for index, row in avg_ipk.iterrows():
        ax.text(index, row['IPK '] + 0.05, round(row['IPK '], 2), color='black', ha="center")
    
    st.pyplot(plt.gcf())
    
    st.write("**Interpretasi:**")
    st.write("Grafik diatas menunjukkan rata-rata IPK mahasiswa berdasrkan status kelulusan mereka. Dapat dilihat bahwa mahasiswa yang lulus tepat waktu memiliki rata-rata IPK 3.1 dan mahasiswa yang lulus terlambat memiliki rata-rata IPK 2.91")
    
    st.write("**Insight:**")
    st.write("Meskipun perbedaan antara rata-rata IPK mahasiswa yang lulus tepat waktu dan yang lulus terlambat mungkin terlihat kecil, namun tetap penting untuk mengenali dan mengatasi faktor-faktor yang bisa mempengaruhi kinerja akademik mahasiswa.")

    st.write("**Actionable Insight:**")
    st.write("Melakukan evaluasi terhadap kurikulum dan metode pengajaran untuk memastikan bahwa mereka relevan, menantang, dan memotivasi mahasiswa untuk mencapai potensi mereka secara penuh.")

# Path ke file CSV dalam folder
FILE_PATH = "after.csv"

# Membaca file CSV ke dalam DataFrame
df = pd.read_csv(FILE_PATH)

# Train the KNeighborsClassifier model
X = df.drop(columns=['STATUS KELULUSAN', 'PredicateCategory'])
y = df['STATUS KELULUSAN']
knn_model = KNeighborsClassifier()
knn_model.fit(X, y)

# Visualitation Page
if selected == 'Visualization':

    # Streamlit page setup
    st.title("Graduation Success Analysis Dashboard")
    # st.markdown("_Dashboard Prototype v1.0_")

    # Streamlit layout
    plot_ips_variance(df)
    plot_gpa_distribution(df)
    plot_ipk_correlation(df)
    plot_gpa_by_status(df)
    plot_marriage_status(df)
    

# Graduation Prediction Page
if selected == "Graduation Prediction":

    # page title
    st.title("Graduation Prediction using ML")

    # Box for IPS input
    st.subheader("Input IPS")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        IPS1 = st.number_input('IPS 1', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    with col2:
        IPS2 = st.number_input('IPS 2', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    with col3:
        IPS3 = st.number_input('IPS 3', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    with col4:
        IPS4 = st.number_input('IPS 4', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    with col5:
        IPS5 = st.number_input('IPS 5', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    with col1:
        IPS6 = st.number_input('IPS 6', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    with col2:
        IPS7 = st.number_input('IPS 7', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    with col3:
        IPS8 = st.number_input('IPS 8', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    with col4:
        IPK = st.number_input('IPK ', min_value=0.0, max_value=4.0, step=0.01, format="%.2f")

    # Radio button for status mahasiswa
    st.subheader("Status Mahasiswa")
    status_mahasiswa = st.radio('', ['Bekerja', 'Mahasiswa'], index=0)

    # Radio button for status pernikahan
    st.subheader("Status Pernikahan")
    status_nikah = st.radio('', ['Belum Menikah', 'Menikah'], index=0)

    knn_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Graduation Test Result"):

        # Mapping string values to binary
        status_mahasiswa_map = {'Bekerja': 0, 'Mahasiswa': 1}
        status_nikah_map = {'Belum Menikah': 0, 'Menikah': 1}

        # Get the corresponding binary values
        status_mahasiswa_val = status_mahasiswa_map[status_mahasiswa]
        status_nikah_val = status_nikah_map[status_nikah]

        user_input = [IPS1, IPS2, IPS3, IPS4, IPS5,
                      IPS6, IPS7, IPS8, IPK, status_mahasiswa_val, status_nikah_val]

        user_input = [float(x) for x in user_input]

        # Perform prediction using the trained model
        knn_prediction = knn_model.predict([user_input])

        if knn_prediction[0] == 1:
            knn_diagnosis = "Lulus Tepat Waktu"
        else:
            knn_diagnosis = "Lulus Terlambat" 

    st.success(knn_diagnosis)


# Graduation Prediction Page
if selected == 'Analytical Intent':

    # page title
    st.title('Analyzing Patterns of Student Graduation Accuracy')

    st.subheader("Tujuan Analisis")
    st.write("Tujuan dari Analyzing Patterns of Student Graduation Accuracy adalah untuk memahami faktor-faktor yang mempengaruhi tingkat ketepatan waktu mahasiswa dalam menyelesaikan studi mereka. Analisis ini bertujuan untuk mengidentifikasi pola atau tren dalam data yang berkaitan dengan tingkat kelulusan tepat waktu, serta mengeksplorasi hubungan antara faktor-faktor tersebut. Dengan memahami pola ketepatan kelulusan mahasiswa, lembaga pendidikan dapat mengembangkan strategi yang tepat untuk meningkatkan tingkat kelulusan tepat waktu, memperbaiki pengalaman belajar mahasiswa, dan memastikan kesuksesan akademik mereka.")

    st.subheader("Dataset Asli")
    URL = 'Kelulusan Train.csv'
    df = pd.read_csv(URL)
    st.write(df)
    st.write("Dataset ini merupakan dataset asli yang akan melalui beberapa tahapan untuk membersihkan, mempersiapkan, dan memvalidasi dataset untuk memastikan bahwa data yang digunakan akurat, relevan, dan representatif sehingga dapat membantu meminimalkan kesalahan dan bias dalam hasil analisis serta memastikan bahwa insight yang dihasilkan dari data tersebut dapat diandalkan dan bermanfaat.")

    st.subheader("Dataset Sebelum Mapping")
    URL = 'before.csv'
    df = pd.read_csv(URL)
    st.write(df)
    st.write("Dataset ini merupakan dataset yang telah melalui beberapa tahapan seperti pembersihan data untuk menghapus nilai yang hilang atau duplikat, pemilihan fitur yang relevan dan pembuatan fitur-fitur baru berdasarkan kombinasi atau transformasi variabel yang sudah ada.")

    st.subheader("Dataset Setelah Mapping")
    URL = 'after.csv'
    df = pd.read_csv(URL)
    st.write(df)
    st.write("Dataset ini merupakan dataset yang telah diubah atau disesuaikan formatnya agar sesuai dengan kebutuhan analisis lebih lanjut. Ini melibatkan transformasi variabel-variabel dalam dataset ke dalam bentuk yang dapat diproses oleh algoritma atau teknik analisis yang akan digunakan, seperti konversi variabel kategorikal menjadi numerik. Dataset yang sudah dimapping siap untuk digunakan dalam analisis lebih lanjut untuk membangun model atau mendapatkan insight dari data yang ada.")
