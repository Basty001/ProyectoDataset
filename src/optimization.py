"""
Memory Optimization Module.
Reduces DataFrame memory footprint and processes large files in chunks.
"""

import pandas as pd
import numpy as np


def optimize_memory(df):
    """
    Reduces the memory usage of a DataFrame by downcasting numeric types.
    
    Técnica: en lugar de int64 (8 bytes) usa el tipo entero más pequeño posible.
    Por ejemplo, Year (1990-2017) cabe en int16 (2 bytes), ahorrando 75% por columna.
    Incluye manejo de excepciones para saltar columnas problemáticas de forma segura.
    """
    try:
        original_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f"💾 Original memory usage: {original_mem:.2f} MB")

        df_opt = df.copy()

        for col in df_opt.select_dtypes(include=['int', 'float']).columns:
            try:
                orig_type = df_opt[col].dtype
                c_min = df_opt[col].min()
                c_max = df_opt[col].max()

                # Conversión de enteros al tipo más pequeño posible
                if str(orig_type).startswith('int'):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_opt[col] = df_opt[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_opt[col] = df_opt[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_opt[col] = df_opt[col].astype(np.int32)

                # Conversión de decimales a float32 (4 bytes en vez de 8)
                elif str(orig_type).startswith('float'):
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_opt[col] = df_opt[col].astype(np.float32)

            except BaseException as e:
                # Si una columna falla, advertimos pero el ciclo continúa
                print(f"⚠️ WARNING: Could not optimize column '{col}': {e}")
                continue

        final_mem = df_opt.memory_usage(deep=True).sum() / 1024**2
        savings = 100 * (original_mem - final_mem) / original_mem

        print(f"🚀 Optimized memory usage: {final_mem:.2f} MB")
        print(f"📉 Total savings: {savings:.1f}%")
        return df_opt

    except Exception as e:
        print(f"❌ CRITICAL ERROR in memory optimization: {e}")
        # En caso de fallo total, devolvemos el dataframe original intacto
        return df


def process_in_chunks(file_path, chunk_size=5000):
    """
    Demonstrates chunk processing for large files.
    
    En producción con datasets de millones de filas, cargar todo en RAM falla.
    pd.read_csv con chunksize crea un iterador: lee N filas, procesa, descarta,
    luego lee las siguientes N filas. Nunca tiene todo en memoria a la vez.
    """
    print(f"\n📦 Processing '{file_path}' in chunks of {chunk_size} rows...")
    total_rows = 0
    try:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            total_rows += len(chunk)
            # Aquí iría la lógica de transformación por bloque en producción
        print(f"✅ Total rows processed: {total_rows}")
        return total_rows
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {file_path}")
        return 0
