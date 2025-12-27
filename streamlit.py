import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import shutil

# ==============================================================================
# KONFIGURASI CONSTANT & STYLE
# ==============================================================================
plt.style.use('ggplot')

# Default Constants
DEFAULT_FPS = 240
MASSA_KELERENG = 0.0055
K_KELERENG = 0.4
MASSA_PINGPONG = 0.0027
K_PINGPONG = 0.666

# ==============================================================================
# CLASS ENGINE (IDENTIK DENGAN FIX.PY)
# ==============================================================================
class PhysicsAnalyzer:
    def __init__(self, fps=240):
        self.fps = fps
        self.results = {} 

    def calculate_velocity(self, frame_start, frame_end, distance):
        if frame_end <= frame_start: return 0.0
        dt = (frame_end - frame_start) / self.fps
        if dt == 0: return 0.0
        return distance / dt

    def calculate_energy(self, mass, velocity, k=0):
        return 0.5 * mass * (velocity ** 2) * (1 + k)

    def analyze_momentum_case(self, case_name, data_list):
        # ...Logic Momentum (Simplified for Streamlit)...
        results = []
        # Setup Constants based on Case
        if case_name == 'Case 1': 
            m_proj = MASSA_KELERENG; m_tq = MASSA_KELERENG
        elif case_name == 'Case 2':
            m_proj = MASSA_KELERENG; m_tq = MASSA_PINGPONG
        else:
            m_proj = MASSA_KELERENG; m_tq = MASSA_KELERENG

        for d in data_list:
            v_bef = d['v_before']
            v_aft1 = d['v_after1'] # Projectile After
            v_aft2 = d['v_after2'] # Target After
            
            p_awal = m_proj * v_bef
            p_akhir = (m_proj * v_aft1) + (m_tq * v_aft2)
            
            e_awal = 0.5 * m_proj * v_bef**2
            e_akhir = (0.5 * m_proj * v_aft1**2) + (0.5 * m_tq * v_aft2**2)
            
            err_p = abs(p_awal - p_akhir)/p_awal*100 if p_awal > 0 else 0
            
            results.append({
                'P_awal': p_awal, 'P_akhir': p_akhir,
                'E_awal': e_awal, 'E_akhir': e_akhir,
                'Error_P (%)': err_p
            })
        return pd.DataFrame(results)

    def plot_waterfall(self, df_data):
        means = [df_data['E_Ramp_End'].mean(), df_data['E_Pre_Collision'].mean(), df_data['E_Post_Total'].mean()]
        stds = [df_data['E_Ramp_End'].std(), df_data['E_Pre_Collision'].std(), df_data['E_Post_Total'].std()]
        stages = ['Ramp End', 'Before Impact', 'Final Total']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2980b9', '#f39c12', '#27ae60']
        bars = ax.bar(stages, means, yerr=stds, capsize=10, color=colors, alpha=0.9, edgecolor='black', width=0.5)
        
        ax.set_title(f"Waterfall Energi: {df_data['Case'].iloc[0]}", fontsize=14)
        ax.set_ylabel("Energi (Joule)")
        
        # Labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f} J', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        # Arrow Loss
        for i in range(len(means)-1):
            start = means[i]
            end = means[i+1]
            loss = start - end
            loss_pct = (loss/start*100) if start > 0 else 0
            
            mid_x = i + 0.5
            mid_y = (start + end)/2
            
            ax.annotate('', xy=(i+0.75, end), xytext=(i+0.25, start),
                        arrowprops=dict(arrowstyle="-|>", color='red', lw=2, ls='--'))
            
            bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
            ax.text(mid_x, mid_y, f"Loss:\n{loss_pct:.1f}%", ha='center', va='center', fontsize=8, bbox=bbox, color='red')
            
        return fig

# ==============================================================================
# STREAMLIT UI HELPER
# ==============================================================================
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.set_page_config(page_title="Ultimate Physics Analyzer", layout="wide", page_icon="🔬")
    
    st.title("🔬 Ultimate Physics Analyzer (Web Version)")
    st.markdown("Analisis Video Fisika: Profil Kecepatan, Tumbukan, dan Energetika.")

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        param_fps = st.number_input("FPS Kamera (Slo-mo)", value=240, step=10)
        param_dist = st.number_input("Jarak Ukur (meter)", value=0.50, step=0.01)
        
        st.subheader("Massa Benda (kg)")
        m_kelereng = st.number_input("Massa Kelereng", value=MASSA_KELERENG, format="%.5f")
        m_pingpong = st.number_input("Massa Pingpong", value=MASSA_PINGPONG, format="%.5f")

    # --- SESSION STATE SETUP ---
    if 'trials_data' not in st.session_state:
        st.session_state['trials_data'] = []
    if 'current_frames' not in st.session_state:
        st.session_state['current_frames'] = []
    
    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["1. Upload & Picking", "2. Data & Analisis", "3. Laporan Storytelling"])

    # === TAB 1: UPLOAD & PICKING ===
    with tab1:
        st.info("Upload video percobaanmu di sini (Support: MOV, MP4).")
        uploaded_video = st.file_uploader("Pilih Video", type=["mov", "mp4"])
        
        if uploaded_video is not None:
            tfile = save_uploaded_file(uploaded_video)
            
            col_vid, col_ctrl = st.columns([2, 1])
            
            with col_ctrl:
                st.subheader("Frame Picker")
                
                # Load Video Cap
                cap = cv2.VideoCapture(tfile)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_detected = cap.get(cv2.CAP_PROP_FPS)
                
                if fps_detected < 60:
                     st.warning(f"FPS Terdeteksi Rendah ({fps_detected}). Menggunakan Override {param_fps} FPS.")
                
                # Slider Picker
                selected_frame_idx = st.slider("Geser untuk pilih frame", 0, total_frames-1, 0)
                
                # Seek & Read
                cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame, caption=f"Frame {selected_frame_idx}", use_container_width=True)
                    
                    if st.button("📍 Tandai Frame Ini"):
                        st.session_state['current_frames'].append(selected_frame_idx)
                        st.success(f"Frame {selected_frame_idx} ditandai!")
                
                st.write("### Titik Terpilih:")
                st.write(st.session_state['current_frames'])
                
                if st.button("❌ Reset Titik"):
                    st.session_state['current_frames'] = []
                    st.rerun()

                st.markdown("---")
                st.write("#### Petunjuk Picking (Mode 3):")
                labels = [
                    "1. Start Ramp", "2. End Ramp",
                    "3. Start Flat", "4. End Pre-Col",
                    "5. Start Post 1", "6. End Post 1",
                    "7. Start Post 2", "8. End Post 2"
                ]
                for l in labels: st.text(l)
                
                if st.button("💾 SIMPAN TRIAL KE DATASET"):
                    if len(st.session_state['current_frames']) == 8:
                        st.session_state['trials_data'].append({
                            'video': uploaded_video.name,
                            'frames': st.session_state['current_frames']
                        })
                        st.session_state['current_frames'] = [] # Reset for next trial
                        st.success("Trial berhasil disimpan! Silakan upload video lain atau lanjut analisis.")
                    else:
                        st.error(f"Jumlah titik harus 8! (Sekarang: {len(st.session_state['current_frames'])})")

    # === TAB 2: DATA & ANALISIS ===
    with tab2:
        st.header("Dataset Trial")
        if not st.session_state['trials_data']:
            st.warning("Belum ada data trial. Silakan pick frame di Tab 1.")
        else:
            df_input = pd.DataFrame(st.session_state['trials_data'])
            st.dataframe(df_input)
            
            case_type = st.selectbox("Pilih Tipe Kasus", ["Case 1 (Marble-Marble)", "Case 2 (Marble-Pingpong)"])
            
            if st.button("🚀 JALANKAN ANALISIS ULTIMATE"):
                analyzer = PhysicsAnalyzer(fps=param_fps)
                processed_data = []
                
                for idx, row in df_input.iterrows():
                    pts = row['frames']
                    # Velocity Calcs
                    v_ramp = analyzer.calculate_velocity(pts[0], pts[1], param_dist)
                    v_pre = analyzer.calculate_velocity(pts[2], pts[3], param_dist)
                    v_p1 = analyzer.calculate_velocity(pts[4], pts[5], param_dist)
                    v_p2 = analyzer.calculate_velocity(pts[6], pts[7], param_dist)
                    
                    # Energy Calcs
                    m_p = m_kelereng
                    if case_type == 'Case 2':
                        m1 = m_kelereng; m2 = m_pingpong
                    else:
                        m1 = m_kelereng; m2 = m_kelereng
                        
                    e_ramp = analyzer.calculate_energy(m_p, v_ramp, K_KELERENG)
                    e_pre = analyzer.calculate_energy(m_p, v_pre, K_KELERENG)
                    e_post = analyzer.calculate_energy(m1, v_p1, K_KELERENG) + analyzer.calculate_energy(m2, v_p2, 0.4 if m2==m_kelereng else K_PINGPONG)
                    
                    processed_data.append({
                        'Trial': idx+1,
                        'Case': case_type,
                        'E_Ramp_End': e_ramp,
                        'E_Pre_Collision': e_pre, # Merged corner/flat
                        'E_Post_Total': e_post
                    })
                
                df_res = pd.DataFrame(processed_data)
                st.session_state['analysis_result'] = df_res
                st.success("Analisis Selesai! Cek Tab 3.")

    # === TAB 3: LAPORAN ===
    with tab3:
        if 'analysis_result' in st.session_state:
            df_res = st.session_state['analysis_result']
            st.header("📊 Laporan Storytelling Otomatis")
            
            col_metrics, col_chart = st.columns([1, 2])
            
            with col_metrics:
                st.subheader("Ringkasan Statistik")
                st.dataframe(df_res.describe())
                
                avg_eff = (df_res['E_Pre_Collision'].mean() / df_res['E_Ramp_End'].mean()) * 100
                st.metric("Efisiensi Tikungan", f"{avg_eff:.1f}%")
                
                err_var = df_res['E_Ramp_End'].std() / df_res['E_Ramp_End'].mean() * 100
                st.metric("Konsistensi (SD%)", f"{err_var:.2f}%")
            
            with col_chart:
                st.subheader("Grafik Waterfall Ultimate")
                analyzer = PhysicsAnalyzer() # Dummy for plot method
                fig = analyzer.plot_waterfall(df_res)
                st.pyplot(fig)
            
            st.subheader("Interpretasi AI")
            if avg_eff > 80:
                st.success("✅ Lintasan tikungan sangat mulus. Energi terpreservasi dengan baik.")
            else:
                st.warning("⚠️ Ada kehilangan energi signifikan di tikungan/gesekan.")
                
        else:
            st.info("Silakan jalankan analisis di Tab 2 terlebih dahulu.")

if __name__ == "__main__":
    main()
