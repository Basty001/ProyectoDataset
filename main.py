import pandas as pd
from pathlib import Path
from src.audit import audit_data
from src.optimization import optimize_memory, process_in_chunks
from src.pipeline import build_preprocessing_pipeline


def main():
    """Main orchestration script for the Car Price Data Science project."""
    print("--- 🚀 Starting Data Pipeline ---\n")

    try:
        # 1. Auditoría de integridad SHA-256
        if not audit_data():
            print("\n🛑 Pipeline stopped due to audit failure.")
            return

        # 2. Demostración de chunk processing (escalabilidad para grandes volúmenes)
        raw_dir = Path("data/raw")
        csv_file = list(raw_dir.glob("*.csv"))[0]
        process_in_chunks(csv_file, chunk_size=5000)

        # 3. Carga del dataset
        print(f"\n📥 Loading raw data from {csv_file.name}...")
        df_raw = pd.read_csv(csv_file)

        # 4. Optimización de memoria (downcasting de tipos numéricos)
        print("\n⚙️ Optimizing memory...")
        df_opt = optimize_memory(df_raw)

        # 5. Construcción y aplicación del pipeline
        print("\n🏗️ Building and applying preprocessing pipeline...")
        # 'Model' se descarta: 915 valores únicos generarían 915 columnas con OHE
        columns_to_drop = ['Model']
        pipeline = build_preprocessing_pipeline(df_opt, columns_to_drop=columns_to_drop)

        processed_matrix = pipeline.fit_transform(df_opt)

        # 6. Guardado del dataset procesado
        print("\n💾 Saving processed dataset...")
        feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

        df_processed = pd.DataFrame(processed_matrix, columns=feature_names)

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / "processed_data.csv"

        df_processed.to_csv(output_path, index=False)
        print(f"✅ SUCCESS: Processed dataset saved at {output_path}")
        print(f"📊 Original shape: {df_raw.shape} → Final shape: {df_processed.shape}")

    except IndexError:
        print("\n❌ CRITICAL ERROR: No se encontró ningún archivo CSV en 'data/raw'.")
    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: Archivo no encontrado: {e}")
    except pd.errors.EmptyDataError:
        print("\n❌ CRITICAL ERROR: El archivo CSV está vacío.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: El pipeline falló inesperadamente: {e}")


if __name__ == "__main__":
    main()
