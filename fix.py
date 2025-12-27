# ============================================================================== 
# ADVANCED PHYSICS ANALYSIS SCRIPT (240 FPS SUPPORT)
# ==============================================================================
import cv2
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

# Set style untuk grafik yang lebih cantik (mirip jurnal ilmiah)
plt.style.use('ggplot')

# ==============================================================================
# KONFIGURASI CONSTANT
# ==============================================================================
NAMA_VIDEO = 'IMG_3238.MOV'
FOLDER_HASIL = 'hasil_analisis_frame'
FPS_VIDEO = 240  # Default 240, bisa diubah jika deteksi otomatis berhasil
MASSA_KELERENG = 0.0055  # Kg (Contoh: 5.5 gram)
K_KELERENG = 0.4 # Solid Sphere
MASSA_PINGPONG = 0.0027  # Kg (Contoh: 2.7 gram)
K_PINGPONG = 0.666 # Hollow Sphere
GRAVITASI = 9.81

# ==============================================================================
# BAGIAN 1: EKSTRAKSI FRAME (Sama seperti sebelumnya, efisien)
# ==============================================================================
# ==============================================================================
# BAGIAN 1: EKSTRAKSI FRAME (Sama seperti sebelumnya, efisien)
# ==============================================================================
def proses_video_ke_frame(video_path=NAMA_VIDEO):
    print("\n" + "="*60)
    print(f"TAHAP 1: MEMPROSES VIDEO: {video_path}")
    print("="*60)

    if not os.path.exists(video_path):
        print(f"[ERROR] File '{video_path}' tidak ditemukan!")
        return 0

    # Buat folder khusus untuk video ini agar tidak tercampur
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    specific_folder = os.path.join(FOLDER_HASIL, base_name)

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # FIX: High speed video often reports 30fps. Force 240 if low.
    if fps_video < 60: 
        print(f"[INFO] Detected Low FPS ({fps_video}). Assuming High-Speed 240 FPS.")
        fps_video = 240
    
    # Cek apakah folder hasil sudah ada dan berisi data
    if os.path.exists(specific_folder) and os.path.exists(os.path.join(specific_folder, 'LOG_FRAMES.csv')):
        print(f"[INFO] Data lama untuk '{base_name}' ditemukan.")
        print(f"[INFO] Using Pre-detected FPS: {fps_video}")
        return fps_video

    if os.path.exists(specific_folder):
        shutil.rmtree(specific_folder)
    os.makedirs(specific_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps_video already detected above
    
    # FIX: Force 240 if detected low (common in iPhone Slo-mo)
    if fps_video < 60: fps_video = 240 
    
    print(f"[INFO] Video ditemukan. Total Frame: {total_frames} | FPS: {fps_video}")
    
    # Deteksi Gerakan untuk efisiensi penyimpanan
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    frame_ke = 0
    saved_count = 0
    log_data = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Apply motion detection mask
        fgmask = fgbg.apply(frame)
        motion_pixels = np.count_nonzero(fgmask)

        # Simpan jika ada gerakan signifikan (>500 pixel berubah)
        if motion_pixels > 500:
            filename = f"frame_{frame_ke:04d}.jpg"
            # Info Teks di Gambar
            cv2.rectangle(frame, (5, 5), (350, 40), (0,0,0), -1)
            cv2.putText(frame, f"F:{frame_ke} | T:{frame_ke/fps_video:.3f}s | M:{motion_pixels}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imwrite(os.path.join(specific_folder, filename), frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"      ... Mengestrak {saved_count} frame", end='\r')

        # Log semua frame (atau yang bergerak saja) untuk analisis grafik
        # Kita log semua frame yang bergerak > 100 pixel agar grafik lebih detail
        if motion_pixels > 100:
             log_data.append({
                'Frame ID': frame_ke, 
                'Waktu (s)': frame_ke/fps_video, 
                'Gerakan (pixel)': motion_pixels,
                'Nama File': f"frame_{frame_ke:04d}.jpg" if motion_pixels > 500 else "N/A"
            })

        frame_ke += 1

    cap.release()
    
    if log_data:
        df = pd.DataFrame(log_data)
        csv_path = os.path.join(specific_folder, 'LOG_FRAMES.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[SUKSES] {saved_count} frame tersimpan di subfolder '{base_name}'")
        print(f"[INFO] Log data frame disimpan di {csv_path}")

        # --- SUGGESTION: MOTION GRAPH ---
        plt.figure(figsize=(12, 4))
        plt.plot(df['Frame ID'], df['Gerakan (pixel)'], color='orange', linewidth=1)
        plt.title(f"Grafik Aktivitas Gerakan: {base_name}")
        plt.xlabel("Nomor Frame")
        plt.ylabel("Intensitas Gerakan (pixel)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        saved_frames = df[df['Nama File'] != "N/A"]
        plt.scatter(saved_frames['Frame ID'], saved_frames['Gerakan (pixel)'], color='red', s=5, label='Frame Disimpan')
        plt.legend()
        
        graph_path = os.path.join(specific_folder, 'BANTUAN_ANALISIS_MOTION_GRAPH.png')
        plt.savefig(graph_path)
        print(f"[SARAN] Cek '{graph_path}' untuk melihat kapan benda mulai bergerak.")
        
        return fps_video
    return 0

# ==============================================================================
# BAGIAN 2: KELAS ANALISIS DATA (STATISTIK & GRAFIK)
# ==============================================================================
class PhysicsAnalyzer:
    def __init__(self, fps=240):
        self.fps = fps
        self.trials_data = [] 
        self.results = {} 

    def calculate_velocity(self, frame_start, frame_end, distance):
        if frame_end <= frame_start: return 0.0
        dt = (frame_end - frame_start) / self.fps
        if dt == 0: return 0.0
        return distance / dt

    def calculate_momentum(self, mass, velocity):
        return mass * velocity

    def calculate_energy(self, mass, velocity, k=0):
        """
        Energi Kinetik Total (Translasi + Rotasi)
        E = 1/2 * m * v^2 * (1 + k)
        k = 0 untuk benda yang tidak berotasi atau rotasinya diabaikan
        k = 2/5 untuk bola pejal (solid sphere)
        k = 2/3 untuk bola berongga (hollow sphere)
        """
        return 0.5 * mass * (velocity ** 2) * (1 + k)

    def add_trial_segment_frames(self, frames_list, start_distance, interval_distance):
        """
        Input: frames_list = [f_start, f_line1, f_line2, ... f_end]
        Menghitung kecepatan antar interval untuk satu trial.
        """
        velocities = []
        times = []
        distances = []
        
        current_dist = start_distance
        
        for i in range(len(frames_list) - 1):
            f1 = frames_list[i]
            f2 = frames_list[i+1]
            dt = (f2 - f1) / self.fps
            
            if dt > 0:
                v = interval_distance / dt
                velocities.append(v)
                times.append((f1+f2)/2 / self.fps) # Waktu tengah interval
                distances.append(current_dist + (interval_distance/2))
            else:
                velocities.append(0)
                times.append(0)
                distances.append(current_dist)
            
            current_dist += interval_distance
            
        return velocities, distances

    def analyze_momentum_case(self, case_name, data_list):
        """
        Menganalisis momentum dan energi berdasarkan kasus.
        data_list: list of dict {'v_before', 'v_after1', 'v_after2'}
        """
        results = []
        
        # Tentukan Massa & Inersia berdasarkan Kasus
        if case_name == 'Case 1': # Marble -> Marble
            m_proj = MASSA_KELERENG; k_proj = K_KELERENG
            m_targ1 = MASSA_KELERENG; k_targ1 = K_KELERENG
            m_targ2 = MASSA_KELERENG; k_targ2 = K_KELERENG 
            
        elif case_name == 'Case 2': # Marble -> Pingpong
            # Prompts: v_after1 is Object A (Marble), v_after2 is Object B (Pingpong)
            m_proj = MASSA_KELERENG; k_proj = K_KELERENG
            m_targ1 = MASSA_KELERENG; k_targ1 = K_KELERENG # A (Marble) retains some momentum
            m_targ2 = MASSA_PINGPONG; k_targ2 = K_PINGPONG # B (Pingpong) moves
            
        elif case_name == 'Case 3': # Marble -> 2 Marbles
            m_proj = MASSA_KELERENG; k_proj = K_KELERENG
            m_targ1 = MASSA_KELERENG; k_targ1 = K_KELERENG
            m_targ2 = MASSA_KELERENG; k_targ2 = K_KELERENG
            
        elif case_name == 'Case 4': # Marble -> Marble + Pingpong
            # Asumsi Prompt: Object 1 = Marble, Object 2 = Pingpong
            m_proj = MASSA_KELERENG; k_proj = K_KELERENG
            m_targ1 = MASSA_KELERENG; k_targ1 = K_KELERENG  
            m_targ2 = MASSA_PINGPONG; k_targ2 = K_PINGPONG
        
        else: # Default
            m_proj = MASSA_KELERENG; k_proj = 0.4
            m_targ1 = MASSA_KELERENG; k_targ1 = 0.4
            m_targ2 = MASSA_KELERENG; k_targ2 = 0.4

        for d in data_list:
            v_bef = d['v_before']
            v_aft1 = d['v_after1']
            v_aft2 = d['v_after2']
            
            # --- MOMENTUM (Linear p = mv) ---
            # Asumsi 1D Sederhana atau Magnitudo Vektor
            p_awal = self.calculate_momentum(m_proj, v_bef)
            p_akhir = self.calculate_momentum(m_targ1, v_aft1) + self.calculate_momentum(m_targ2, v_aft2)
            
            # --- ENERGI (Rotasi + Translasi) ---
            e_awal = self.calculate_energy(m_proj, v_bef, k_proj)
            e_akhir = self.calculate_energy(m_targ1, v_aft1, k_targ1) + self.calculate_energy(m_targ2, v_aft2, k_targ2)
            
            # Error
            err_p = abs(p_awal - p_akhir) / p_awal * 100 if p_awal > 0 else 0
            err_e = abs(e_awal - e_akhir) / e_awal * 100 if e_awal > 0 else 0
            
            results.append({
                'v_before': v_bef,
                'v_after1': v_aft1,
                'v_after2': v_aft2,
                'P_awal': p_awal,
                'P_akhir': p_akhir,
                'E_awal': e_awal,
                'E_akhir': e_akhir,
                'Error_P (%)': err_p,
                'Error_E (%)': err_e
            })
            
        df = pd.DataFrame(results)
        self.results[case_name] = df
        return df

    def print_automated_analysis(self, df):
        """Mencetak analisis otomatis dalam Bahasa Indonesia."""
        print("\n" + "="*60)
        print("INTERPRETASI OTOMATIS (AI PHYSICS ANALYST)")
        print("="*60)
        
        # 1. Analisis Error Momentum
        err_p = df['Error_P (%)'].mean()
        if err_p < 10:
            print(f"[SUCCESS] Hukum Kekekalan Momentum TERBUKTI SANGAT BAIK.")
            print(f"          Error rata-rata hanya {err_p:.2f}% (<10%).")
        elif err_p < 20:
            print(f"[OK] Hukum Kekekalan Momentum CUKUP TERBUKTI.")
            print(f"     Error {err_p:.2f}% mungkin karena gesekan meja atau tracking.")
        else:
            print(f"[WARNING] Error Momentum TINGGI ({err_p:.2f}%).")
            print("          Cek kembali tracking frame atau kemungkinan meja miring.")

        # 2. Analisis Energi (Tumbukan Lenting/Tak Lenting)
        err_e = df['Error_E (%)'].mean()
        print("-" * 60)
        if err_e < 10:
             print(f"[TYPE] Tumbukan mendekati LENTING SEMPURNA (Elastic).")
             print(f"       Energi kinetik terjaga dengan baik (Loss: {err_e:.2f}%).")
        elif err_e < 50:
             print(f"[TYPE] Tumbukan TIDAK LENTING SEBAGIAN (Inelastic).")
             print(f"       Sebagian energi ({err_e:.2f}%) hilang menjadi bunyi/panas.")
             print("       Ini wajar untuk bola pejal/kelereng.")
        else:
             print(f"[TYPE] Tumbukan TIDAK LENTING SAMA SEKALI (Perfectly Inelastic)?")
             print(f"       Energi hilang sngat besar ({err_e:.2f}%).")
        print("="*60)
    
    # --- [NEW] MODE 3 FEATURES ADDED HERE ---
    def print_storytelling_report(self, df_data):
        print("\n" + "="*60)
        print("LAPORAN NARASI FISIKA (MODE 3 - MULTI TRIAL)")
        print("="*60)
        
        # Calculate Means
        e_in = df_data['E_Ramp_End'].mean()
        e_corner = df_data['E_Corner_Exit'].mean()
        e_final = df_data['E_Post_Total'].mean()
        
        eff = (e_corner/e_in*100) if e_in > 0 else 0
        loss_c = e_in - e_corner
        loss_f = e_corner - e_corner # Simplified model (merged)
        
        print(f"1. ANALISIS SIKU (Rata-rata): Efisiensi Tikungan {eff:.1f}%.")
        print(f"   Energi hilang di tikungan: {loss_c:.4f} J.")
        print(f"2. ANALISIS TUMBUKAN: Total Energi Akhir {e_final:.4f} J.")
        
        # Error Analysis
        err_ramp = df_data['E_Ramp_End'].std() / e_in * 100 if e_in > 0 else 0
        print(f"3. KONSISTENSI DATA: Variasi Percobaan (Error) {err_ramp:.2f}%.")
        print("="*60)

    def plot_velocity_profile(self, all_velocities, segment_labels):
        data_np = np.array(all_velocities)
        mean_v = np.mean(data_np, axis=0) # Rata-rata per segmen
        std_v = np.std(data_np, axis=0)   # Standar Deviasi per segmen
        x_pos = np.arange(len(segment_labels))
        
        plt.figure(figsize=(10, 6))
        
        # Plot INDIVIDUAL TRIALS (Thin lines)
        colors = plt.cm.jet(np.linspace(0, 1, len(all_velocities)))
        for i, velocities in enumerate(all_velocities):
             plt.plot(x_pos, velocities, marker='.', linestyle=':', linewidth=1, 
                      color=colors[i], alpha=0.7, label=f'Trial {i+1}')

        # Plot AVERAGE (Thick Blue line) with Error Bars
        plt.errorbar(x_pos, mean_v, yerr=std_v, fmt='o-', capsize=5, color='blue', 
                     label='Rata-rata', linewidth=2.5, ecolor='black')

        # --- TRENDLINE (Analisis Gesekan) ---
        # Fit linear (y = mx + c)
        z = np.polyfit(x_pos, mean_v, 1)
        p = np.poly1d(z)
        plt.plot(x_pos, p(x_pos), "r--", linewidth=2, label=f'Trendline (Slope={z[0]:.4f})')
        
        # Anotasi
        txt_gesek = "Perlambatan Terdeteksi (Gesekan)" if z[0] < 0 else "Kecepatan Konstan/Naik"
        plt.text(x_pos[len(x_pos)//2], max(mean_v), txt_gesek, 
                 fontsize=12, color='red', fontweight='bold', ha='center')

        plt.xticks(x_pos, segment_labels, rotation=45)
        plt.title(f"Perbandingan Profil Kecepatan ({len(all_velocities)} Trial)")
        plt.xlabel("Segmen Lintasan")
        plt.ylabel("Kecepatan (m/s)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        filename = os.path.join(FOLDER_HASIL, 'GradeA_Profil_Kecepatan_Komparasi.png')
        plt.savefig(filename)
        print(f"[GRAFIK] Disimpan: {filename}")
        # plt.show()

    def plot_conservation(self, case_name):
        df = self.results[case_name]
        
        mean_p = [df['P_awal'].mean(), df['P_akhir'].mean()]
        std_p  = [df['P_awal'].std(), df['P_akhir'].std()]
        
        mean_ek = [df['E_awal'].mean(), df['E_akhir'].mean()]
        std_ek  = [df['E_awal'].std(), df['E_akhir'].std()]
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Grafik Momentum
        ax[0].bar(['Awal', 'Akhir'], mean_p, yerr=std_p, capsize=10, color=['#3498db', '#2ecc71'], alpha=0.8)
        ax[0].set_title(f"{case_name}: Kekekalan Momentum\n(Error Rata2: {df['Error_P (%)'].mean():.2f}%)")
        ax[0].set_ylabel("Momentum (kg.m/s)")
        
        # Grafik Energi
        ax[1].bar(['Awal', 'Akhir'], mean_ek, yerr=std_ek, capsize=10, color=['#e74c3c', '#f1c40f'], alpha=0.8)
        ax[1].set_title(f"{case_name}: Kekekalan Energi\n(Error Rata2: {df['Error_E (%)'].mean():.2f}%)")
        ax[1].set_ylabel("Energi Kinetik (Joule)")
        
        plt.tight_layout()
        filename = os.path.join(FOLDER_HASIL, f'Grafik_{case_name.replace(" ", "_")}.png')
        plt.savefig(filename)
        print(f"[GRAFIK] Disimpan: {filename}")

    # --- [NEW] MODE 3 PLOT ---
    def plot_advanced_waterfall(self, df_data):
        # Calculate Stats
        means = [
            df_data['E_Ramp_End'].mean(), 
            df_data['E_Pre_Collision'].mean(), 
            df_data['E_Post_Total'].mean()
        ]
        stds = [
            df_data['E_Ramp_End'].std(), 
            df_data['E_Pre_Collision'].std(), 
            df_data['E_Post_Total'].std()
        ]
        
        stages = ['Ramp End', 'Before Impact', 'Final Total']
        
        plt.figure(figsize=(10, 6))
        # Colors: Blue Start, Orange Impact, Green Final
        colors = ['#2980b9', '#f39c12', '#27ae60']
        
        # Plot Bar with Error Caps
        bars = plt.bar(stages, means, yerr=stds, capsize=10, 
                       color=colors, alpha=0.8, edgecolor='black', width=0.5)
        
        plt.title(f"Waterfall Energi: {df_data['Case'].iloc[0]} (Mean ± SD)", fontsize=14)
        plt.ylabel("Energi Kinetik (Joule)")
        
        # Add Value Labels on top of bars
        for bar, err in zip(bars, stds):
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + err + (max(means)*0.02), 
                     f'{h:.4f} J', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Draw Arrows and Annotations for LOSS/EFFICIENCY
        for i in range(len(means)-1):
            start_e = means[i]
            end_e = means[i+1]
            loss = start_e - end_e
            
            if start_e > 0:
                percent_loss = (loss / start_e) * 100
                efficiency = (end_e / start_e) * 100
            else:
                percent_loss = 0
                efficiency = 0
            
            # Position for arrow (between bars)
            mid_x = i + 0.5
            mid_y = (start_e + end_e) / 2
            
            # Draw Arrow
            plt.annotate('', xy=(i+0.75, end_e), xytext=(i+0.25, start_e),
                         arrowprops=dict(arrowstyle="-|>", color='red', lw=1.5, ls='--'))
            
            # Draw Text Box
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
            txt_label = f"Loss: {loss:.4f} J\n(-{percent_loss:.1f}%)"
            
            plt.text(mid_x, mid_y, txt_label, ha='center', va='center', 
                     color='red', fontsize=9, bbox=bbox_props)

        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FOLDER_HASIL, 'Grafik_Mode3_Waterfall.png'))
        print("[GRAFIK] Disimpan: Grafik_Mode3_Waterfall.png")

# ==============================================================================
# BAGIAN 3: INTERACTIVE PICKER (ALAT BANTU PILIH FRAME)
# ==============================================================================
def interactive_frame_picker(n_points_needed=0, point_labels=None, context="Pilih Frame", video_path=NAMA_VIDEO):
    """
    Membuka jendela GUI untuk memilih frame secara visual.
    point_labels: List string nama titik yang harus dicari (opsional)
    video_path: Path video yang akan digunakan (default: constant global)
    """
    # If point_labels is provided, use it.
    target_labels = point_labels if point_labels else [f"Point {i+1}" for i in range(n_points_needed)]
    n_needed = len(target_labels)

    print(f"\n[INFO] Membuka Interactive Picker untuk: {context}")
    print(f"[INFO] Video Source: {video_path}")
    print("KEYS: [A/D] Gerak | [Spasi] Pilih | [Enter] Selesai | [R] Reset")
    print(">>> TOLONG KLIK JENDELA POPUP 'Frame Picker' AGAR BISA TEKAN TOMBOL <<<")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video '{NAMA_VIDEO}' di dalam picker!")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    marked_frames = []
    
    cv2.namedWindow("Frame Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame Picker", 1000, 600)
    
    # Bawa window ke depan (Windows only trick)
    cv2.setWindowProperty("Frame Picker", cv2.WND_PROP_TOPMOST, 1)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret: 
            print("[DEBUG] Gagal membaca frame. End of video?")
            break
        
        # Gambar Info di Layar
        info_text = f"FRAME: {current_frame} / {total_frames} | Time: {current_frame/FPS_VIDEO:.3f}s"
        cv2.putText(frame, info_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # INSTRUKSI TARGET
        if len(marked_frames) < n_needed:
            target_msg = f"CARI: {target_labels[len(marked_frames)]}"
            cv2.putText(frame, target_msg, (20, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
             cv2.putText(frame, "SELESAI! TEKAN ENTER.", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # List Marked
        cv2.putText(frame, f"Marked [{len(marked_frames)}]: {marked_frames}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Frame Picker", frame)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d'): # Next
            current_frame = min(total_frames-1, current_frame + 1)
        elif key == ord('a'): # Prev
            current_frame = max(0, current_frame - 1)
        elif key == ord('w'): # Fast Fwd
            current_frame = min(total_frames-1, current_frame + 20)
        elif key == ord('s'): # Fast Rewind
            current_frame = max(0, current_frame - 20)
        elif key == 32: # SPACE to Mark
            if len(marked_frames) < n_needed:
                if current_frame not in marked_frames:
                    marked_frames.append(current_frame)
                    marked_frames.sort()
                    print(f"   -> Frame {current_frame} ditandai.")
        elif key == ord('r'): # Reset
            marked_frames = []
            print("   -> Reset selection.")
        elif key == 13: # ENTER
            if len(marked_frames) == n_needed:
                break
            else:
                print(f"[INFO] Belum selesai. Butuh {n_needed} titik.")
        elif key == 27: # ESC
            exit()
            
    cv2.destroyAllWindows()
    cap.release()
    return marked_frames

# ==============================================================================
# BAGIAN 4: MAIN INTERFACE
# ==============================================================================
def main():
    # 1. Init Analyzer (FPS awal placeholder, nanti diupdate per video)
    experiment = PhysicsAnalyzer(fps=240)
    
    print("\n" + "="*60)
    print("TAHAP 2: INPUT DATA & ANALISIS")
    print("="*60)
    
    print("Pilih Mode:")
    print("1. Analisis Profil Kecepatan (Multi-Segmen)")
    print("2. Analisis Tumbukan (Case 1/2/3/4) dengan Multi-Trial")
    print("3. Analisis LENGKAP (Siku + Gesekan + Tumbukan) [NEW!]")
    
    mode = input(">> Masukkan Pilihan (1/2/3): ")
    
    if mode == '1':
        # --- ANALISIS KECEPATAN ---
        n_trials = int(input("Jumlah Trial (contoh: 3): "))
        n_segments = int(input("Jumlah Segmen Garis (contoh: 20): "))     
        dist_per_seg = float(input("Jarak antar garis (meter, cth 0.1): "))
        
        # Custom Labels
        labels = [f"Segmen {i+1}" for i in range(n_segments)]
        print(f"\n[OPSI] Default label: {labels[:3]}...")
        cust_label = input("Apakah ingin mengubah nama segmen? (y/n): ")
        if cust_label.lower() == 'y':
            print("Masukkan nama untuk setiap segmen (Tekan Enter untuk skip/default):")
            for i in range(n_segments):
                nama = input(f"  - Nama Segmen {i+1} (Default 'Segmen {i+1}'): ")
                if nama.strip():
                    labels[i] = nama
        
        all_velocities = []
        
        all_velocities = []
        
        # Ask Video Strategy
        vid_strategy = input("Video Source? (1=Sama utk semua, 2=Beda tiap trial): ")
        current_vid = NAMA_VIDEO 

        for t in range(n_trials):
            print(f"\n--- TRIAL {t+1} ---")
            
            # Handle Multi-Video
            if vid_strategy == '2':
                 v_input = input(f"Masukkan nama video Trial {t+1} (Default: {NAMA_VIDEO}): ")
                 if v_input.strip(): current_vid = v_input
                 if not os.path.exists(current_vid):
                     print(f"[ERROR] File {current_vid} tidak ditemukan! Menggunakan default.")
                     current_vid = NAMA_VIDEO
            
            print(f"[INFO] Using Video: {current_vid}")
            
            # --- PROCESS VIDEO JUST IN TIME ---
            fps_trial = proses_video_ke_frame(current_vid)
            if fps_trial > 0: experiment.fps = fps_trial
            print(f"[INFO] FPS Set to: {experiment.fps}")
            # ----------------------------------

            print(f"[INFO] Silakan pilih frame saat benda melewati setiap garis.")
            
            # Construct Point Labels for GUI
            point_labels_gui = []
            point_labels_gui.append(f"START dari {labels[0]}")
            for i in range(len(labels)-1):
                point_labels_gui.append(f"END {labels[i]} / START {labels[i+1]}")
            point_labels_gui.append(f"FINISH dari {labels[-1]}")

            # PANGGIL PICKER
            frames = interactive_frame_picker(
                point_labels=point_labels_gui,
                context=f"Trial {t+1}",
                video_path=current_vid
            )
            
            if len(frames) < n_segments + 1:
                print(f"[WARNING] Jumlah titik kurang. Skip.")
                continue
            
            # Slice frames
            frames = frames[:n_segments+1]
            print(f"Frame terpilih: {frames}")
            
            try:
                v_list, _ = experiment.add_trial_segment_frames(frames, 0, dist_per_seg)
                all_velocities.append(v_list)
                
                # PRINT HASIL KE TERMINAL LANGSUNG
                print(f"\n[HASIL TRIAL {t+1}]")
                print(f"{'SEGMENT':<20} | {'KECEPATAN (m/s)':<15}")
                print("-" * 40)
                for i_seg, vel in enumerate(v_list):
                    print(f"{labels[i_seg]:<20} | {vel:.4f}")
                print("-" * 40)
                
            except ValueError:
                print("Input error, skip trial ini.")
        
        if all_velocities:
            if len(all_velocities[0]) == len(labels):
                experiment.plot_velocity_profile(all_velocities, labels)
            else:
                 temp_labels = [f"Seg {i+1}" for i in range(len(all_velocities[0]))]
                 experiment.plot_velocity_profile(all_velocities, temp_labels)
            
    elif mode == '2':
        # --- ANALISIS TUMBUKAN ---
        print("\nPilih Kasus:")
        print("  A. Case 1 (2 Kelereng Sama - 1 Dimensi)")
        print("  B. Case 2 (Kelereng vs Pingpong - 1 Dimensi)")
        print("  C. Case 3 (1 Kelereng -> 2 Kelereng - Branching)")
        print("  D. Case 4 (1 Kelereng -> 1 Kelereng + 1 Pingpong - Branching)")
        
        pick_map = {'A': 'Case 1', 'B': 'Case 2', 'C': 'Case 3', 'D': 'Case 4'}
        case_pick = input(">> Pilihan (A/B/C/D): ").upper()
        case_name = pick_map.get(case_pick, 'Case 1')
        
        print(f"\n[INFO] Mode Terpilih: {case_name}")
        
        case_labels = {
            'Case 1': ["Start Awal (A)", "Sesaat SEBELUM Tabrakan (A)", "Start Setelah (A)", "End Setelah (A)", "Start Setelah (B)", "End Setelah (B)"],
            'Case 2': ["Start Awal (Marble)", "Sesaat SEBELUM Tabrakan (Marble)", "Start Setelah (Marble)", "End Setelah (Marble)", "Start Setelah (Pingpong)", "End Setelah (Pingpong)"],
            'Case 3': ["Start Awal (Penembak)", "Sesaat SEBELUM Tabrakan (Penembak)", "Start Setelah (Target Kiri)", "End Setelah (Target Kiri)", "Start Setelah (Target Kanan)", "End Setelah (Target Kanan)"],
            'Case 4': ["Start Awal (Kelereng)", "Sesaat SEBELUM Tabrakan (Kelereng)", "Start Setelah (Kelereng)", "End Setelah (Kelereng)", "Start Setelah (Pingpong)", "End Setelah (Pingpong)"]
        }
        
        current_labels = case_labels.get(case_name, case_labels['Case 1'])
        
        n_trials = int(input(f"Jumlah Trial untuk {case_name}: "))
        data_trials = []
        
        jarak_ukur = float(input("Jarak ukur untuk kecepatan (m): "))
        
        # Ask Video Strategy
        vid_strategy = input("Video Source? (1=Sama utk semua, 2=Beda tiap trial): ")
        current_vid = NAMA_VIDEO 

        for t in range(n_trials):
            print(f"\n--- TRIAL {t+1} ---")
            
            # Handle Multi-Video
            if vid_strategy == '2':
                 print(f"Masukkan nama video untuk Trial {t+1} (Default: {NAMA_VIDEO})")
                 v_input = input(">> Filename: ")
                 if v_input.strip(): current_vid = v_input
                 if not os.path.exists(current_vid):
                     print(f"[ERROR] File {current_vid} tidak ditemukan! Menggunakan default.")
                     current_vid = NAMA_VIDEO
            
            print(f"[INFO] Using Video: {current_vid}")
            
            # --- PROCESS VIDEO JUST IN TIME ---
            fps_trial = proses_video_ke_frame(current_vid)
            if fps_trial > 0: experiment.fps = fps_trial
            # ----------------------------------
            
            print("Pilih 6 TITIK UTAMA sesuai instruksi di layar:")
            for idx, txt in enumerate(current_labels):
                 print(f"  {idx+1}. {txt}")

            frames = interactive_frame_picker(n_points_needed=6, point_labels=current_labels, context=f"{case_name} T{t+1}", video_path=current_vid)
            
            if len(frames) >= 6:
                fa1, fa2 = frames[0], frames[1]
                fb1_start, fb1_end = frames[2], frames[3]
                fb2_start, fb2_end = frames[4], frames[5]
                
                v_before = 0
                if fa2 > fa1: 
                    v_before = jarak_ukur / ((fa2 - fa1) / experiment.fps)
                
                v_after1 = 0
                if fb1_end > fb1_start:
                    v_after1 = jarak_ukur / ((fb1_end - fb1_start) / experiment.fps)
                    
                v_after2 = 0
                if fb2_end > fb2_start:
                    v_after2 = jarak_ukur / ((fb2_end - fb2_start) / experiment.fps)
                
                data_trials.append({
                    'v_before': v_before,
                    'v_after1': v_after1,
                    'v_after2': v_after2
                })
                
                print(f"[HASIL TRIAL {t+1}]")
                print(f"  v_before : {v_before:.4f} m/s")
                print(f"  v_after_1: {v_after1:.4f} m/s")
                print(f"  v_after_2: {v_after2:.4f} m/s")
                
            else:
                print("[ERROR] Point kurang dari 6. Skip Trial ini.")
                
        if data_trials:
            df_result = experiment.analyze_momentum_case(case_name, data_trials)
            print("\n" + "="*50)
            print(f"HASIL ANALISIS STATISTIK: {case_name}")
            print("="*50)
            print(df_result[['v_before', 'P_awal', 'P_akhir', 'Error_P (%)']].to_string(index=False))
            print("-" * 50)
            print(f"Rata-rata Error Momentum: {df_result['Error_P (%)'].mean():.2f}%")
            
            experiment.print_automated_analysis(df_result)
            experiment.plot_conservation(case_name)

    elif mode == '3':
        # --- [NEW] MODE 3: ULTIMATE ANALYSIS ---
        print("\n--- MODE 3: ULTIMATE ANALYSIS ---")
        print("Analisis Alur: Ramp -> Siku -> Datar -> Tumbukan")
        print("Pilih Tumbukan Akhir: A(Case 1), B(Case 2), C(Case 3), D(Case 4)")
        cmap = {'A':'Case 1','B':'Case 2','C':'Case 3','D':'Case 4'}
        cname = cmap.get(input(">> Pilihan: ").upper(), 'Case 1')
        dist = float(input("Jarak ukur standar (m): "))
        
        n_trials = int(input("Jumlah Trial Mode 3 (cth: 3): "))
        all_mode3_data = []

        # Ask Video Strategy
        vid_strategy = input("Video Source? (1=Sama utk semua, 2=Beda tiap trial): ")
        current_vid = NAMA_VIDEO
        
        for t in range(n_trials):
            print(f"\n--- TRIAL {t+1} / {n_trials} ---")
            
            if vid_strategy == '2':
                 print(f"Masukkan nama video untuk Trial {t+1} (Default: {NAMA_VIDEO})")
                 v_input = input(">> Filename: ")
                 if v_input.strip(): current_vid = v_input
                 if not os.path.exists(current_vid):
                     print(f"[ERROR] File {current_vid} tidak ditemukan! Menggunakan default.")
                     current_vid = NAMA_VIDEO
    
            # --- PROCESS VIDEO JUST IN TIME ---
            fps_trial = proses_video_ke_frame(current_vid)
            if fps_trial > 0: experiment.fps = fps_trial
            print(f"[INFO] FPS Set to: {experiment.fps}")
            # ----------------------------------
    
            # 8 Points Picking Logic (Simplified: Ramp -> Flat/Pre-Col -> Post 1 -> Post 2)
            labels = [
                "Start Ramp", "End Ramp (Pre-Corner)",
                "Start Flat (Post-Corner)", "End Flat (Pre-Col)",
                "Start Post 1", "End Post 1",
                "Start Post 2", "End Post 2"
            ]
            
            # Use simple adaptation for picker
            pts = interactive_frame_picker(point_labels=labels, context=f"Ultimate T{t+1}", video_path=current_vid)
            
            if len(pts) == 8:
                v_ramp = experiment.calculate_velocity(pts[0], pts[1], dist)
                v_pre = experiment.calculate_velocity(pts[2], pts[3], dist) # Combined Corner Exit / Pre-Col
                v_p1 = experiment.calculate_velocity(pts[4], pts[5], dist)
                v_p2 = experiment.calculate_velocity(pts[6], pts[7], dist)
                
                # Setup Mass based on Case
                m_p = MASSA_KELERENG; k_p = K_KELERENG
                if cname in ['Case 2', 'Case 4']: 
                     m1=MASSA_KELERENG; k1=K_KELERENG; m2=MASSA_PINGPONG; k2=K_PINGPONG
                else:
                     m1=MASSA_KELERENG; k1=K_KELERENG; m2=MASSA_KELERENG; k2=K_KELERENG
                
                # Energy Calc (Simplified)
                all_mode3_data.append({
                    'Case': cname,
                    'E_Ramp_End': experiment.calculate_energy(m_p, v_ramp, k_p),
                    'E_Corner_Exit': experiment.calculate_energy(m_p, v_pre, k_p),
                    'E_Pre_Collision': experiment.calculate_energy(m_p, v_pre, k_p),
                    'E_Post_Total': experiment.calculate_energy(m1, v_p1, k1) + experiment.calculate_energy(m2, v_p2, k2)
                })
            else:
                print("[ERROR] Data tidak lengkap (Wajib 8 titik). Skip trial ini.")

        if all_mode3_data:
            df_res = pd.DataFrame(all_mode3_data)
            experiment.print_storytelling_report(df_res)
            experiment.plot_advanced_waterfall(df_res)
        else:
            print("[INFO] Tidak ada data valid untuk dianalisis.")

if __name__ == "__main__":
    main()
