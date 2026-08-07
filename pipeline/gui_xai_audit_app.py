import os
import sys
import threading
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Adiciona o caminho base para importação dos modelos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipeline.xai_gradcam_analysis import GradCAM, CustomCNN, get_resnet50, get_convnext_tiny, get_swin_tiny

def infer_architecture_type(model_filename):
    name = model_filename.lower()
    if "convnext" in name:
        return "convnext"
    elif "resnet" in name or "transfer" in name:
        return "resnet"
    elif "swin" in name:
        return "swin"
    elif "cnn" in name:
        return "cnn"
    return "convnext"

def load_model_instance(pth_path, arch_type, num_classes=5, device="cpu"):
    if arch_type == "convnext":
        model = get_convnext_tiny(num_classes)
        target_layer = model.features[-1]
    elif arch_type == "resnet":
        model = get_resnet50(num_classes)
        target_layer = model.layer4[-1]
    elif arch_type == "cnn":
        model = CustomCNN(num_classes)
        target_layer = model.features[12]
    elif arch_type == "swin":
        model = get_swin_tiny(num_classes)
        target_layer = model.norm
    else:
        model = get_convnext_tiny(num_classes)
        target_layer = model.features[-1]

    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, target_layer

class XAIGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auditoria XAI - Reconhecimento de Plásticos")
        self.root.geometry("900x720")
        self.root.minsize(800, 600)
        self.root.configure(bg="#1e1e2e")

        self.selected_models = []
        self.selected_images = []
        self.default_output_dir = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\Resultados_Modelos\v5_benchmark_expanded_results_v2\xai_gui_output"

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Cores modernas
        self.style.configure("TFrame", background="#1e1e2e")
        self.style.configure("Header.TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 16, "bold"))
        self.style.configure("SubHeader.TLabel", background="#1e1e2e", foreground="#a6adc8", font=("Segoe UI", 10))
        self.style.configure("Card.TLabelframe", background="#252538", foreground="#89b4fa", font=("Segoe UI", 11, "bold"))
        self.style.configure("Card.TLabelframe.Label", background="#252538", foreground="#89b4fa")
        
        self.style.configure("Primary.TButton", background="#89b4fa", foreground="#11111b", font=("Segoe UI", 10, "bold"), borderwidth=0)
        self.style.map("Primary.TButton", background=[("active", "#b4befe")])

        self.style.configure("Action.TButton", background="#a6e3a1", foreground="#11111b", font=("Segoe UI", 12, "bold"), borderwidth=0)
        self.style.map("Action.TButton", background=[("active", "#94e2d5")])

    def create_widgets(self):
        main_container = tk.Frame(self.root, bg="#1e1e2e", padx=20, pady=15)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Cabeçalho
        lbl_title = ttk.Label(main_container, text="🔬 Auditoria Visual XAI por Mapas de Calor (Grad-CAM)", style="Header.TLabel")
        lbl_title.pack(anchor="w", pady=(0, 2))
        lbl_sub = ttk.Label(main_container, text="Selecione os modelos (.pth) e as imagens de teste para gerar automaticamente as grades comparativas lado a lado.", style="SubHeader.TLabel")
        lbl_sub.pack(anchor="w", pady=(0, 15))

        # Painel Superior: Modelos e Imagens
        panels_frame = tk.Frame(main_container, bg="#1e1e2e")
        panels_frame.pack(fill=tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # QUADRO 1: MODELOS (.PTH)
        # -------------------------------------------------------------
        card_models = ttk.LabelFrame(panels_frame, text=" 🤖 1. Modelos Neurais (.pth) ", style="Card.TLabelframe", padding=10)
        card_models.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        btn_box_m = tk.Frame(card_models, bg="#252538")
        btn_box_m.pack(fill=tk.X, pady=(0, 8))

        btn_add_models = ttk.Button(btn_box_m, text="➕ Selecionar Arquivos (.pth)", style="Primary.TButton", command=self.add_model_files)
        btn_add_models.pack(side=tk.LEFT, padx=(0, 5))

        btn_add_model_dir = ttk.Button(btn_box_m, text="📁 Adicionar Pasta", style="Primary.TButton", command=self.add_model_folder)
        btn_add_model_dir.pack(side=tk.LEFT, padx=5)

        btn_clear_models = tk.Button(btn_box_m, text="Limpar", bg="#f38ba8", fg="#11111b", font=("Segoe UI", 9, "bold"), relief="flat", command=self.clear_models)
        btn_clear_models.pack(side=tk.RIGHT)

        self.listbox_models = tk.Listbox(card_models, bg="#181825", fg="#cdd6f4", selectbackground="#45475a", font=("Consolas", 9), relief="flat", highlightthickness=1, highlightbackground="#313244")
        self.listbox_models.pack(fill=tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # QUADRO 2: IMAGENS DE TESTE (.JPG, .PNG)
        # -------------------------------------------------------------
        card_imgs = ttk.LabelFrame(panels_frame, text=" 🖼️ 2. Imagens de Teste ", style="Card.TLabelframe", padding=10)
        card_imgs.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        btn_box_i = tk.Frame(card_imgs, bg="#252538")
        btn_box_i.pack(fill=tk.X, pady=(0, 8))

        btn_add_imgs = ttk.Button(btn_box_i, text="➕ Selecionar Imagens", style="Primary.TButton", command=self.add_image_files)
        btn_add_imgs.pack(side=tk.LEFT, padx=(0, 5))

        btn_add_img_dir = ttk.Button(btn_box_i, text="📁 Adicionar Pasta", style="Primary.TButton", command=self.add_image_folder)
        btn_add_img_dir.pack(side=tk.LEFT, padx=5)

        btn_clear_imgs = tk.Button(btn_box_i, text="Limpar", bg="#f38ba8", fg="#11111b", font=("Segoe UI", 9, "bold"), relief="flat", command=self.clear_images)
        btn_clear_imgs.pack(side=tk.RIGHT)

        self.listbox_imgs = tk.Listbox(card_imgs, bg="#181825", fg="#cdd6f4", selectbackground="#45475a", font=("Consolas", 9), relief="flat", highlightthickness=1, highlightbackground="#313244")
        self.listbox_imgs.pack(fill=tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # QUADRO INFERIOR: PASTA DE SAÍDA E EXECUÇÃO
        # -------------------------------------------------------------
        card_out = tk.Frame(main_container, bg="#1e1e2e", pady=10)
        card_out.pack(fill=tk.X)

        lbl_out = tk.Label(card_out, text="📁 Pasta de Destino das Grades:", bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 10, "bold"))
        lbl_out.pack(anchor="w", pady=(0, 3))

        out_box = tk.Frame(card_out, bg="#1e1e2e")
        out_box.pack(fill=tk.X)

        self.entry_out = tk.Entry(out_box, bg="#181825", fg="#a6e3a1", font=("Segoe UI", 10), relief="flat", highlightthickness=1, highlightbackground="#313244")
        self.entry_out.insert(0, self.default_output_dir)
        self.entry_out.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 8))

        btn_browse_out = ttk.Button(out_box, text="Procurar...", style="Primary.TButton", command=self.browse_output_dir)
        btn_browse_out.pack(side=tk.LEFT)

        # Barra de Progresso e Status
        self.progress_bar = ttk.Progressbar(main_container, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(10, 5))

        self.lbl_status = tk.Label(main_container, text="Aguardando seleção de modelos e imagens...", bg="#1e1e2e", fg="#89dceb", font=("Segoe UI", 10))
        self.lbl_status.pack(anchor="w", pady=(0, 10))

        # Botões de Ação
        action_box = tk.Frame(main_container, bg="#1e1e2e")
        action_box.pack(fill=tk.X)

        self.btn_run = ttk.Button(action_box, text="🚀 GERAR GRADES COMPARATIVAS LADO A LADO", style="Action.TButton", command=self.start_processing_thread)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))

        self.btn_open_folder = tk.Button(action_box, text="📂 Abrir Pasta de Resultados", bg="#fab387", fg="#11111b", font=("Segoe UI", 11, "bold"), relief="flat", command=self.open_output_folder)
        self.btn_open_folder.pack(side=tk.RIGHT, ipady=6, padx=(10, 0))

        # Pré-carrega modelos padrão se existirem
        self.auto_load_default_models()

    def auto_load_default_models(self):
        default_dir = r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\modelos0408"
        if os.path.exists(default_dir):
            for f in sorted(os.listdir(default_dir)):
                if f.endswith(".pth"):
                    full_p = os.path.join(default_dir, f)
                    if full_p not in self.selected_models:
                        self.selected_models.append(full_p)
                        self.listbox_models.insert(tk.END, f"[{infer_architecture_type(f).upper()}] {f}")

    def add_model_files(self):
        files = filedialog.askopenfilenames(
            title="Selecione os arquivos de modelo (.pth)",
            filetypes=[("Modelos PyTorch", "*.pth"), ("Todos os arquivos", "*.*")],
            initialdir=r"C:\Users\Vinicius\Desktop\MestradoCodeAnti"
        )
        for f in files:
            if f not in self.selected_models:
                self.selected_models.append(f)
                fname = os.path.basename(f)
                self.listbox_models.insert(tk.END, f"[{infer_architecture_type(fname).upper()}] {fname}")

    def add_model_folder(self):
        folder = filedialog.askdirectory(title="Selecione a pasta com arquivos .pth", initialdir=r"C:\Users\Vinicius\Desktop\MestradoCodeAnti")
        if folder:
            for root, _, files in os.walk(folder):
                for f in sorted(files):
                    if f.endswith(".pth") and not f.startswith("."):
                        full_p = os.path.join(root, f)
                        if full_p not in self.selected_models:
                            self.selected_models.append(full_p)
                            self.listbox_models.insert(tk.END, f"[{infer_architecture_type(f).upper()}] {f}")

    def clear_models(self):
        self.selected_models.clear()
        self.listbox_models.delete(0, tk.END)

    def add_image_files(self):
        files = filedialog.askopenfilenames(
            title="Selecione as imagens de teste",
            filetypes=[("Imagens", "*.jpg;*.jpeg;*.png"), ("Todos os arquivos", "*.*")],
            initialdir=r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\Datasets"
        )
        for f in files:
            if f not in self.selected_images:
                self.selected_images.append(f)
                self.listbox_imgs.insert(tk.END, os.path.basename(f))

    def add_image_folder(self):
        folder = filedialog.askdirectory(title="Selecione a pasta de imagens", initialdir=r"C:\Users\Vinicius\Desktop\MestradoCodeAnti\Datasets")
        if folder:
            valid_exts = ('.jpg', '.jpeg', '.png')
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(valid_exts):
                    full_p = os.path.join(folder, f)
                    if full_p not in self.selected_images:
                        self.selected_images.append(full_p)
                        self.listbox_imgs.insert(tk.END, f)

    def clear_images(self):
        self.selected_images.clear()
        self.listbox_imgs.delete(0, tk.END)

    def browse_output_dir(self):
        folder = filedialog.askdirectory(title="Selecione a pasta de saída", initialdir=self.default_output_dir)
        if folder:
            self.entry_out.delete(0, tk.END)
            self.entry_out.insert(0, folder)

    def open_output_folder(self):
        out_dir = self.entry_out.get().strip()
        if os.path.exists(out_dir):
            os.startfile(out_dir)
        else:
            messagebox.showinfo("Informação", f"A pasta ainda não foi criada:\n{out_dir}")

    def start_processing_thread(self):
        if not self.selected_models:
            messagebox.showwarning("Aviso", "Por favor, selecione pelo menos 1 modelo (.pth).")
            return
        if not self.selected_images:
            messagebox.showwarning("Aviso", "Por favor, selecione pelo menos 1 imagem de teste.")
            return

        self.btn_run.config(state=tk.DISABLED)
        threading.Thread(target=self.run_processing, daemon=True).start()

    def run_processing(self):
        out_dir = self.entry_out.get().strip()
        os.makedirs(out_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        classes = ['Other', 'PET', 'PE_HD', 'PP', 'PS']

        total_images = len(self.selected_images)
        self.progress_bar["maximum"] = total_images
        self.progress_bar["value"] = 0

        # Carrega instâncias de modelos uma única vez para performance
        self.lbl_status.config(text="Carregando modelos selecionados na memória...")
        loaded_models = []
        for pth_path in self.selected_models:
            fname = os.path.basename(pth_path)
            arch_t = infer_architecture_type(fname)
            try:
                model, target_layer = load_model_instance(pth_path, arch_t, num_classes=5, device=device)
                display_name = fname.replace(".pth", "")[:22]
                loaded_models.append((display_name, model, target_layer))
            except Exception as e:
                print(f"Erro ao carregar {fname}: {e}")

        # Itera sobre cada imagem
        for idx, img_path in enumerate(self.selected_images):
            img_name = os.path.basename(img_path)
            self.lbl_status.config(text=f"Processando imagem ({idx+1}/{total_images}): {img_name}")

            try:
                orig_pil = Image.open(img_path).convert("RGB")
                orig_np = np.array(orig_pil.resize((224, 224)))
                input_tensor = transform(orig_pil).unsqueeze(0).to(device)

                panels = []
                # Bloco 1: Imagem Original
                blk_orig = np.zeros((224 + 32, 224, 3), dtype=np.uint8)
                lbl_orig = np.zeros((32, 224, 3), dtype=np.uint8)
                cv2.putText(lbl_orig, "Imagem Original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                blk_orig[:32, :] = lbl_orig
                blk_orig[32:, :] = orig_np
                panels.append(blk_orig)

                # Gera o mapa de calor para cada modelo carregado
                for display_name, model, target_layer in loaded_models:
                    gradcam = GradCAM(model, target_layer)
                    heatmap, pred_idx = gradcam.generate_heatmap(input_tensor)
                    pred_class = classes[pred_idx] if pred_idx < len(classes) else f"Classe_{pred_idx}"

                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)

                    blk = np.zeros((224 + 32, 224, 3), dtype=np.uint8)
                    lbl = np.zeros((32, 224, 3), dtype=np.uint8)
                    cv2.putText(lbl, f"{display_name}", (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(lbl, f"Pred: {pred_class}", (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
                    blk[:32, :] = lbl
                    blk[32:, :] = overlay
                    panels.append(blk)

                # Monta a grade (máximo 4 colunas por linha)
                cols_per_row = min(len(panels), 4)
                rows = []
                for p_idx in range(0, len(panels), cols_per_row):
                    row_p = panels[p_idx : p_idx + cols_per_row]
                    if len(row_p) < cols_per_row:
                        dummy = np.zeros((224 + 32, 224 * (cols_per_row - len(row_p)), 3), dtype=np.uint8)
                        row_p.append(dummy)
                    rows.append(np.hstack(row_p))

                full_grid = np.vstack(rows)
                grid_filename = f"GRADE_{idx+1:03d}_{img_name}"
                out_grid_path = os.path.join(out_dir, grid_filename)
                cv2.imwrite(out_grid_path, cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Erro ao processar imagem {img_name}: {e}")

            self.progress_bar["value"] = idx + 1

        self.lbl_status.config(text=f"✅ Concluído! {total_images} grades salvas em: {out_dir}")
        self.btn_run.config(state=tk.NORMAL)
        messagebox.showinfo("Sucesso", f"Grades comparativas geradas com sucesso!\n\nSalvas em:\n{out_dir}")

def main():
    root = tk.Tk()
    app = XAIGuiApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
