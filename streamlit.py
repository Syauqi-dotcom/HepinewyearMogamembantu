import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import math

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
# PHYSICS ENGINE
# ==============================================================================
class PhysicsAnalyzer:
    def __init__(self, fps=240):
        self.fps = fps

    def calculate_velocity(self, frame_start, frame_end, distance):
        if frame_end <= frame_start: return 0.0
        dt = (frame_end - frame_start) / self.fps
        if dt == 0: return 0.0
        return distance / dt

    def calculate_momentum(self, mass, velocity):
        return mass * velocity

    def calculate_energy(self, mass, velocity, k=0):
        # E = 1/2 * m * v^2 * (1 + k)
        return 0.5 * mass * (velocity ** 2) * (1 + k)

    def analyze_branch(self, name, p_list, m_obj, k_obj, dist):
        """
        Analisis satu cabang (segmen-per-segmen).
        Return: (last_vals, first_vals, data_dict)
        """
        if len(p_list) < 2: return (0,0,0), (0,0,0), None
        
        plot_frames = []
        plot_v = []
        plot_p = []
        plot_e = []
        
        results = []
        
        v_last, p_last, e_last = 0,0,0
        v_first, p_first, e_first = 0,0,0
        
        for i in range(len(p_list)-1):
            v = self.calculate_velocity(p_list[i], p_list[i+1], dist)
            p = self.calculate_momentum(m_obj, v)
            e = self.calculate_energy(m_obj, v, k_obj)
            
            results.append({
                'Segmen': f"{i+1}-{i+2}",
                'Frames': f"{p_list[i]}-{p_list[i+1]}",
                'Velocity': v,
                'Momentum': p,
                'Energy': e
            })
            
            if i == 0: v_first, p_first, e_first = v, p, e
            v_last, p_last, e_last = v, p, e
            
            # Data for plot (midpoint)
            mid_frame = (p_list[i] + p_list[i+1]) / 2
            plot_frames.append(mid_frame)
            plot_v.append(v)
            plot_p.append(p)
            plot_e.append(e)
            
        return (v_last, p_last, e_last), (v_first, p_first, e_first), {
            'frames': plot_frames, 'v': plot_v, 'p': plot_p, 'e': plot_e, 'table': pd.DataFrame(results)
        }

# ==============================================================================
# UI HELPERS
# ==============================================================================
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def picking_tab_content(key_prefix, video_path, fps_override):
    """Component reusable untuk tab picking dengan Navigasi Button"""
    # 1. Init Session State list points
    if 'frames_' + key_prefix not in st.session_state:
        st.session_state['frames_' + key_prefix] = []
    
    # 2. Init Session State Current Frame Index per Tab
    idx_key = f"idx_{key_prefix}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    col_vid, col_k = st.columns([3, 1])
    
    # --- LOAD VIDEO METADATA ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- LAYOUTING ---
    with col_vid:
        # 1. DISPLAY IMAGE FIRST (Visual Focus)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state[idx_key])
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, use_container_width=True)
            
        # 2. NAVIGATION CONTROLS (BELOW IMAGE)
        st.caption("Navigasi Frame:")
        c_nav1, c_nav2, c_nav3, c_nav4, c_nav5 = st.columns([1,1,2,1,1])
        
        current_idx = st.session_state[idx_key]
        
        if c_nav1.button("⏪ -10", key=f"n10_{key_prefix}"):
            current_idx = max(0, current_idx - 10)
        if c_nav2.button("◀️ Prev", key=f"prev_{key_prefix}"):
            current_idx = max(0, current_idx - 1)
            
        # Slider synced
        current_idx = c_nav3.slider("Seek", 0, total_frames-1, current_idx, key=f"sld_{key_prefix}", label_visibility="collapsed")
        
        if c_nav4.button("Next ▶️", key=f"next_{key_prefix}"):
            current_idx = min(total_frames-1, current_idx + 1)
        if c_nav5.button("+10 ⏩", key=f"p10_{key_prefix}"):
            current_idx = min(total_frames-1, current_idx + 10)
            
        # Update State (Immediate)
        st.session_state[idx_key] = current_idx
        
        # 3. MARK BUTTON (Action)
        st.write("") # Spacer
        if st.button(f"📍 TANDAI FRAME {current_idx}", key=f"add_{key_prefix}", type="primary", use_container_width=True):
            # Insert sorted unique
            current = st.session_state['frames_' + key_prefix]
            if current_idx not in current:
                current.append(current_idx)
                current.sort()
                st.session_state['frames_' + key_prefix] = current
                st.success(f"Frame {current_idx} berhasil disimpan!")
                st.rerun()

    with col_k:
        st.markdown(f"### Daftar Titik ({len(st.session_state['frames_' + key_prefix])})")
        st.write("Frame Index:")
        st.write(st.session_state['frames_' + key_prefix])
        
        if st.button(f"🗑️ HAPUS SEMUA", key=f"rst_{key_prefix}"):
            st.session_state['frames_' + key_prefix] = []
            st.rerun()

# ==============================================================================
# MAIN PAGE
# ==============================================================================
def main():
    st.set_page_config(page_title="Physics Lab: Free Analysis", layout="wide", page_icon="🧪")
    
    st.title("🧪 Physics Lab: Free Analysis (Mode 5 Equivalent)")
    st.markdown("Analisis Eksperimen Fisika dengan sistem **Branching** (Shooter, Target A, Target B) dan **Verifikasi Teoritis**.")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # 1. Video Settings
        param_fps = st.number_input("FPS Kamera (High Speed)", value=240, step=10)
        
        # 2. Calibration
        param_dist = st.number_input("Jarak Ukur per Segmen (m)", value=0.05, format="%.3f", step=0.01)
        
        # 3. Case Type
        st.divider()
        st.subheader("Tipe Kasus")
        case_pick = st.radio("Pilih Case:", 
                             ["A. Case 1 (Marble 1D)", 
                              "B. Case 2 (Marble-Pingpong 1D)",
                              "C. Case 3 (Branching Marble)",
                              "D. Case 4 (Branching Pingpong)"])
        
        case_code = case_pick[0] # A/B/C/D
        st.info(f"Mode Terpilih: {case_code}")
        
        # 4. Mass Config
        st.divider()
        st.subheader("Parameter Benda")
        m_sho = MASSA_KELERENG
        k_sho = K_KELERENG
        
        # Determine Target B mass
        if case_code in ['B', 'D']:
             st.caption("Target B terdeteksi sebagai Pingpong.")
             m_t2_val = MASSA_PINGPONG
             k_t2_val = K_PINGPONG
        else:
             m_t2_val = MASSA_KELERENG
             k_t2_val = K_KELERENG
             
        # Allow Override
        if st.checkbox("Custom Massa/Inersia?"):
            m_sho = st.number_input("Massa Shooter (kg)", value=m_sho, format="%.5f")
            m_t2_val = st.number_input("Massa Target B (kg)", value=m_t2_val, format="%.5f")

    # --- MAIN CONTENT ---
    
    # 1. VIDEO UPLOAD
    uploaded_video = st.file_uploader("📂 Upload Video Eksperimen (.MOV, .MP4)", type=["mov", "mp4"])
    
    if uploaded_video:
        tfile = save_uploaded_file(uploaded_video)
        
        # --- TAB PICKING SECTION ---
        st.subheader("1. Picking Frame (Tandai Titik)")
        
        # Tentukan Struktur Tab berdasarkan Case
        is_branching = case_code in ['C', 'D']
        
        if is_branching:
            tabs = st.tabs(["🔴 JALUR 1: SHOOTER", "🟢 JALUR 2: TARGET A", "🔵 JALUR 3: TARGET B"])
            
            with tabs[0]:
                st.info("Tandai posisi Shooter (Kelereng Penembak) dari awal sampai SEBELUM tumbukan percabangan.")
                picking_tab_content("shooter", tfile, param_fps)
                
            with tabs[1]:
                st.info("Tandai posisi Target A (Jalur Kiri) SETELAH tumbukan.")
                picking_tab_content("targetA", tfile, param_fps)
                
            with tabs[2]:
                st.info("Tandai posisi Target B (Jalur Kanan) SETELAH tumbukan.")
                picking_tab_content("targetB", tfile, param_fps)
                
        else:
            # Linear Case A/B
            tabs = st.tabs(["MAIN PATH"])
            with tabs[0]:
                st.info("Tandai semua titik berurutan (Satu jalur lurus).")
                picking_tab_content("shooter", tfile, param_fps)

        # --- ANALYSIS BUTTON ---
        st.divider()
        if st.button("🚀 ANALISIS DATA SEKARANG", type="primary"):
            analyzer = PhysicsAnalyzer(fps=param_fps)
            
            # Retrieve Points
            pts_sho = st.session_state.get('frames_shooter', [])
            pts_t1 = st.session_state.get('frames_targetA', [])
            pts_t2 = st.session_state.get('frames_targetB', [])
            
            # --- PROCESSING ---
            # 1. Shooter
            (v_s_end, p_s_end, e_s_end), _, d_sho = analyzer.analyze_branch("Shooter", pts_sho, m_sho, k_sho, param_dist)
            
            # 2. Target A (Only if branching)
            d_t1 = None; p_a_start = 0; e_a_start = 0; p_b_start = 0; e_b_start = 0
            
            if is_branching:
                _, (v_a_start, p_a_start, e_a_start), d_t1 = analyzer.analyze_branch("Target A", pts_t1, m_sho, k_sho, param_dist)
                
                # 3. Target B
                _, (v_b_start, p_b_start, e_b_start), d_t2 = analyzer.analyze_branch("Target B", pts_t2, m_t2_val, k_t2_val, param_dist)
                
                # --- RESULTS ---
                st.subheader("2. Hasil Analisis (Branching)")
                
                # Dataframes
                c1, c2, c3 = st.columns(3)
                if d_sho: c1.write("Data Shooter"); c1.dataframe(d_sho['table'])
                if d_t1: c2.write("Data Target A"); c2.dataframe(d_t1['table'])
                if d_t2: c3.write("Data Target B"); c3.dataframe(d_t2['table'])
                
                # Summary Conservation
                p_final = p_a_start + p_b_start
                e_final = e_a_start + e_b_start
                err_p = abs(p_s_end - p_final)/p_s_end*100 if p_s_end else 0
                err_e = abs(e_s_end - e_final)/e_s_end*100 if e_s_end else 0
                
                st.warning("📊 SUMMARY HUKUM KEKEKALAN")
                
                met_col1, met_col2 = st.columns(2)
                met_col1.metric("Error Momentum", f"{err_p:.2f}%", f"{p_s_end:.5f} -> {p_final:.5f}")
                met_col2.metric("Energy Loss", f"{err_e:.2f}%", f"{e_s_end:.5f} -> {e_final:.5f} J")
                
                # --- CHARTS ---
                if d_sho and d_t1 and d_t2:
                    st.subheader("3. Grafik Profil")
                    
                    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
                    
                    # V
                    axs[0].plot(d_sho['frames'], d_sho['v'], 'o-', label='Shooter')
                    axs[0].plot(d_t1['frames'], d_t1['v'], 's-', label='Target A')
                    axs[0].plot(d_t2['frames'], d_t2['v'], '^-', label='Target B')
                    axs[0].set_ylabel("Velocity (m/s)")
                    axs[0].legend()
                    
                    # P
                    axs[1].plot(d_sho['frames'], d_sho['p'], 'o--')
                    axs[1].plot(d_t1['frames'], d_t1['p'], 's--')
                    axs[1].plot(d_t2['frames'], d_t2['p'], '^--')
                    axs[1].set_ylabel("Momentum")
                    
                    # E
                    axs[2].plot(d_sho['frames'], d_sho['e'], 'o:')
                    axs[2].plot(d_t1['frames'], d_t1['e'], 's:')
                    axs[2].plot(d_t2['frames'], d_t2['e'], '^:')
                    axs[2].set_ylabel("Energy (J)")
                    axs[2].set_xlabel("Frame")
                    
                    st.pyplot(fig)
                    
                    # Loss Chart
                    st.subheader("4. Grafik Loss Analysis")
                    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
                    
                    labels = ['In(S)', 'Out(A)', 'Out(B)', 'Loss']
                    p_loss = p_s_end - p_final
                    e_loss = e_s_end - e_final
                    
                    axs2[0].bar(labels, [p_s_end, p_a_start, p_b_start, p_loss], color=['blue','green','red','gray'])
                    axs2[0].set_title("Momentum Balance")
                    
                    axs2[1].bar(labels, [e_s_end, e_a_start, e_b_start, e_loss], color=['blue','green','red','gray'])
                    axs2[1].set_title("Energy Balance")
                    
                    st.pyplot(fig2)
                    
            else:
                # Linear Case
                st.info("Pilih Case Branching (C/D) untuk fitur lengkap. Case Linear hanya menampilkan tabel dasar.")
                if d_sho: st.dataframe(d_sho['table'])

        # --- THEORETICAL CALCULATOR ---
        st.divider()
        with st.expander("📐 Kalkulator Verifikasi Teoritis (Experiment vs Theory)"):
            st.write("Verifikasi kecepatan Shooter berdasarkan rumus menggelinding murni.")
            
            c_teori1, c_teori2 = st.columns(2)
            angle = c_teori1.number_input("Sudut Kemiringan (derajat)", value=30.0)
            length_cm = c_teori2.number_input("Panjang Lintasan (cm)", value=50.0)
            
            if st.button("Hitung Prediksi Teoritis"):
                g = 9.8
                L = length_cm / 100.0
                rad = math.radians(angle)
                h = L * math.sin(rad)
                
                v_theory = math.sqrt((10/7) * g * h)
                
                st.info(f"Ketinggian (h): {h:.3f} m")
                st.success(f"Kecepatan Teoritis: {v_theory:.4f} m/s")
                
                # Check against data if available
                # We need to store last analysis result in session state to compare here, 
                # but simplest is just let user compare manually or re-run analysis.

if __name__ == "__main__":
    main()
