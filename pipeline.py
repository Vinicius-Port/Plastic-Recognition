import os
import sys
import argparse

# Add pipeline module directory to sys.path
pipeline_module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline")
if pipeline_module_dir not in sys.path:
    sys.path.insert(0, pipeline_module_dir)

import ingest_dataset
import ingest_other_background_swap
import train_pipeline
import model_registry
import simulate_belt_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="🚀 PIPELINE AUTOMATIZADO DE RECONHECIMENTO DE PLÁSTICOS (MLOps & AI Engineering)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis do pipeline")

    # 1. Ingestão Padrão
    ingest_parser = subparsers.add_parser("ingest", help="Ingerir e padronizar novas imagens no formato WaDaBa")
    ingest_parser.add_argument("--incoming", type=str, required=True, help="Diretório com novas imagens")
    ingest_parser.add_argument("--dataset_dir", type=str, default="./Dataset_Wadaba", help="Diretório alvo do dataset")
    ingest_parser.add_argument("--obj_id", type=str, default=None, help="ID do objeto (ex: 0105)")
    ingest_parser.add_argument("--class_name", type=str, default=None, help="Classe (PET, PE_HD, PP, PS, Other)")

    # 1b. Ingestão com Substituição de Fundo (Qualquer Classe)
    swap_parser = subparsers.add_parser("ingest_swap", help="Substituir fundo branco/claro de novas imagens pela esteira")
    swap_parser.add_argument("--incoming", type=str, required=True, help="Diretório com novas imagens de fundo branco")
    swap_parser.add_argument("--dataset_dir", type=str, default="./Dataset_Wadaba", help="Diretório alvo do dataset")
    swap_parser.add_argument("--class_name", type=str, default="Other", help="Classe de destino (PET, PE_HD, PP, PS, Other)")
    swap_parser.add_argument("--texture", type=str, default="data/esteira_textura.jpg", help="Textura da esteira")

    # 2. Quality Gate
    subparsers.add_parser("quality", help="Executar Data Quality Gate no dataset")

    # 3. Treino
    train_parser = subparsers.add_parser("train", help="Treinar modelo individual ou todos com salvamento de métricas")
    train_parser.add_argument("--model", type=str, default=None, help="Modelo específico (ex: resnet_looo, convnext_looo, 2)")
    train_parser.add_argument("--data_dir", type=str, default="./Dataset_Wadaba", help="Caminho do dataset")
    train_parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")

    # 4. Leaderboard
    subparsers.add_parser("leaderboard", help="Compilar Leaderboard dos modelos treinados em outputs/")

    # 5. Exportação ONNX
    export_parser = subparsers.add_parser("export", help="Exportar modelo PyTorch (.pth) para ONNX")
    export_parser.add_argument("--model_path", type=str, required=True, help="Caminho do modelo (.pth)")
    export_parser.add_argument("--arch", type=str, required=True, choices=["cnn", "resnet", "convnext", "swin"], help="Arquitetura")

    # 6. Simulação na Esteira
    sim_parser = subparsers.add_parser("simulate", help="Executar simulação da esteira em tempo real com Gabarito")
    sim_parser.add_argument("--model_path", type=str, default="outputs/modelo_resnet_looo/model.pth", help="Caminho dos pesos (.pth)")
    sim_parser.add_argument("--arch", type=str, default="resnet", choices=["cnn", "resnet", "convnext", "swin"], help="Arquitetura")
    sim_parser.add_argument("--video", type=str, default="data/simulation_belt.mp4", help="Vídeo da esteira")
    sim_parser.add_argument("--gt", type=str, default="data/belt_ground_truth.json", help="Arquivo de Gabarito (Ground Truth)")
    sim_parser.add_argument("--headless", action="store_true", help="Executar sem janela gráfica")

    # 7. Run All
    subparsers.add_parser("run_all", help="Executar pipeline completo de ponta a ponta")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        ingest_dataset.ingest_batch_directory(args.incoming, args.dataset_dir, args.obj_id, args.class_name)
        ingest_dataset.run_quality_gate(args.dataset_dir)

    elif args.command == "ingest_swap":
        ingest_other_background_swap.process_white_bg_directory(args.incoming, args.dataset_dir, args.texture, args.class_name)
        ingest_dataset.run_quality_gate(args.dataset_dir)

    elif args.command == "quality":
        ingest_dataset.run_quality_gate("./Dataset_Wadaba")

    elif args.command == "train":
        experiments = [
            {"name": "modelo_cnn_looo", "split": "looo", "arch": "cnn"},
            {"name": "modelo_resnet_looo", "split": "looo", "arch": "resnet"},
            {"name": "modelo_convnext_looo", "split": "looo", "arch": "convnext"},
            {"name": "modelo_swin_looo", "split": "looo", "arch": "swin"},
            {"name": "modelo_cnn_random", "split": "random", "arch": "cnn"},
            {"name": "modelo_resnet_random", "split": "random", "arch": "resnet"},
            {"name": "modelo_convnext_random", "split": "random", "arch": "convnext"},
            {"name": "modelo_swin_random", "split": "random", "arch": "swin"}
        ]
        if args.model:
            target = args.model.lower().strip()
            filtered = [e for i, e in enumerate(experiments, 1) if target == str(i) or target == e["name"] or target in e["name"]]
            if filtered:
                experiments = filtered

        for exp in experiments:
            train_pipeline.run_single_experiment(exp, args.data_dir, args.epochs)
        model_registry.compile_leaderboard()

    elif args.command == "leaderboard":
        model_registry.compile_leaderboard()

    elif args.command == "export":
        model_registry.export_model_onnx(args.model_path, args.arch)

    elif args.command == "simulate":
        simulate_belt_pipeline.run_belt_simulation(args.model_path, args.arch, video_path=args.video, gt_path=args.gt, show_window=not args.headless)

    elif args.command == "run_all":
        ingest_dataset.run_quality_gate("./Dataset_Wadaba")
        train_pipeline.run_single_experiment({"name": "modelo_resnet_looo", "split": "looo", "arch": "resnet"}, "./Dataset_Wadaba", epochs=2)
        model_registry.compile_leaderboard()
        simulate_belt_pipeline.run_belt_simulation("outputs/modelo_resnet_looo/model.pth", "resnet", video_path="data/simulation_belt.mp4", gt_path="data/belt_ground_truth.json", show_window=False)

if __name__ == '__main__':
    main()
